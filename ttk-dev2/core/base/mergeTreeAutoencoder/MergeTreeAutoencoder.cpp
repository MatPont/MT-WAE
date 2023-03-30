#include <MergeTreeAutoencoder.h>
#include <MergeTreeAutoencoderUtils.h>
#include <cmath>

#ifdef TTK_ENABLE_TORCH
using namespace torch::indexing;
#endif

ttk::MergeTreeAutoencoder::MergeTreeAutoencoder() {
  // inherited from Debug: prefix will be printed at the beginning of every msg
  this->setDebugMsgPrefix("MergeTreeAutoencoder");
#ifdef TTK_ENABLE_OPENMP
  omp_set_nested(1);
#endif
}

#ifdef TTK_ENABLE_TORCH
//  ---------------------------------------------------------------------------
//  --- Init
//  ---------------------------------------------------------------------------
void ttk::MergeTreeAutoencoder::initOutputBasisTreeStructure(
  TorchUtils::TorchMergeTree<float> &originPrime,
  bool isJT,
  TorchUtils::TorchMergeTree<float> &baseOrigin) {
  // ----- Create scalars vector
  std::vector<float> scalarsVector(
    originPrime.tensor.data_ptr<float>(),
    originPrime.tensor.data_ptr<float>() + originPrime.tensor.numel());
  unsigned int noNodes = scalarsVector.size() / 2;
  std::vector<std::vector<ftm::idNode>> childrenFinal(noNodes);

  // ----- Init tree structure and modify scalars if necessary
  if(isPersistenceDiagram_) {
    for(unsigned int i = 2; i < scalarsVector.size(); i += 2)
      childrenFinal[0].emplace_back(i / 2);
  } else {
    // --- Fix or swap min-max pair
    float maxPers = std::numeric_limits<float>::lowest();
    unsigned int indMax = 0;
    for(unsigned int i = 0; i < scalarsVector.size(); i += 2) {
      if(maxPers < (scalarsVector[i + 1] - scalarsVector[i])) {
        maxPers = (scalarsVector[i + 1] - scalarsVector[i]);
        indMax = i;
      }
    }
    if(indMax != 0) {
      float temp = scalarsVector[0];
      scalarsVector[0] = scalarsVector[indMax];
      scalarsVector[indMax] = temp;
      temp = scalarsVector[1];
      scalarsVector[1] = scalarsVector[indMax + 1];
      scalarsVector[indMax + 1] = temp;
    }
    ftm::idNode refNode = 0;
    for(unsigned int i = 2; i < scalarsVector.size(); i += 2) {
      ftm::idNode node = i / 2;
      adjustNestingScalars(scalarsVector, node, refNode);
    }

    if(not initOriginPrimeStructByCopy_
       or (int) noNodes > baseOrigin.mTree.tree.getRealNumberOfNodes()) {
      // --- Get possible children and parent relations
      std::vector<std::vector<ftm::idNode>> parents(noNodes), children(noNodes);
      for(unsigned int i = 0; i < scalarsVector.size(); i += 2) {
        for(unsigned int j = i; j < scalarsVector.size(); j += 2) {
          if(i == j)
            continue;
          unsigned int iN = i / 2, jN = j / 2;
          if(scalarsVector[i] <= scalarsVector[j]
             and scalarsVector[i + 1] >= scalarsVector[j + 1]) {
            // - i is parent of j
            parents[jN].emplace_back(iN);
            children[iN].emplace_back(jN);
          } else if(scalarsVector[i] >= scalarsVector[j]
                    and scalarsVector[i + 1] <= scalarsVector[j + 1]) {
            // - j is parent of i
            parents[iN].emplace_back(jN);
            children[jN].emplace_back(iN);
          }
        }
      }
      ttk::createBalancedBDT(parents, children, scalarsVector, childrenFinal);
    } else {
      ftm::MergeTree<float> mTreeTemp
        = ftm::copyMergeTree<float>(baseOrigin.mTree);
      bool useBD = true;
      keepMostImportantPairs<float>(&(mTreeTemp.tree), noNodes, useBD);
      torch::Tensor reshaped = torch::tensor(scalarsVector).reshape({-1, 2});
      torch::Tensor order = torch::argsort(
        (reshaped.index({Slice(), 1}) - reshaped.index({Slice(), 0})), -1,
        true);
      std::vector<unsigned int> nodeCorr(mTreeTemp.tree.getNumberOfNodes(), 0);
      unsigned int nodeNum = 1;
      std::queue<ftm::idNode> queue;
      queue.emplace(mTreeTemp.tree.getRoot());
      while(!queue.empty()) {
        ftm::idNode node = queue.front();
        queue.pop();
        std::vector<ftm::idNode> children;
        mTreeTemp.tree.getChildren(node, children);
        for(auto &child : children) {
          queue.emplace(child);
          unsigned int tNode = nodeCorr[node];
          nodeCorr[child] = order[nodeNum].item<int>();
          ++nodeNum;
          unsigned int tChild = nodeCorr[child];
          childrenFinal[tNode].emplace_back(tChild);
          adjustNestingScalars(scalarsVector, tChild, tNode);
        }
      }
    }
  }

  // ----- Create new tree
  originPrime.mTree = ftm::createEmptyMergeTree<float>(scalarsVector.size());
  ftm::FTMTree_MT *tree = &(originPrime.mTree.tree);
  if(isJT) {
    for(unsigned int i = 0; i < scalarsVector.size(); i += 2) {
      float temp = scalarsVector[i];
      scalarsVector[i] = scalarsVector[i + 1];
      scalarsVector[i + 1] = temp;
    }
  }
  ftm::setTreeScalars<float>(originPrime.mTree, scalarsVector);

  // ----- Create tree structure
  originPrime.nodeCorr.clear();
  originPrime.nodeCorr.assign(
    scalarsVector.size(), std::numeric_limits<unsigned int>::max());
  for(unsigned int i = 0; i < scalarsVector.size(); i += 2) {
    tree->makeNode(i);
    tree->makeNode(i + 1);
    tree->getNode(i)->setOrigin(i + 1);
    tree->getNode(i + 1)->setOrigin(i);
    originPrime.nodeCorr[i] = (unsigned int)(i / 2);
  }
  for(unsigned int i = 0; i < scalarsVector.size(); i += 2) {
    unsigned int node = i / 2;
    for(auto &child : childrenFinal[node])
      tree->makeSuperArc(child * 2, i);
  }
  TorchUtils::getParentsVector(originPrime.mTree, originPrime.parentsOri);

  if(isTreeHasBigValues(originPrime.mTree, bigValuesThreshold_)) {
    std::cout << originPrime.mTree.tree.printPairsFromTree<float>(true).str()
              << std::endl;
    std::cout << "isTreeHasBigValues(originPrime.mTree)" << std::endl;
    std::cout << "pause" << std::endl;
    std::cin.get();
  }
}

void ttk::MergeTreeAutoencoder::initOutputBasis(unsigned int l,
                                                unsigned int dim,
                                                unsigned int dim2) {
  unsigned int originSize = origins_[l].tensor.sizes()[0];
  unsigned int origin2Size = 0;
  if(useDoubleInput_)
    origin2Size = origins2_[l].tensor.sizes()[0];

  // --- Compute output basis origin
  printMsg("Compute output basis origin", debug::Priority::DETAIL);
  auto initOutputBasisOrigin = [this, &l](
                                 torch::Tensor &w,
                                 TorchUtils::TorchMergeTree<float> &tmt,
                                 TorchUtils::TorchMergeTree<float> &baseTmt) {
    // - Create scalars
    torch::nn::init::xavier_normal_(w);
    // torch::nn::init::orthogonal_(w);
    torch::Tensor baseTmtTensor = baseTmt.tensor;
    if(normalizedWasserstein_)
      // Work on unnormalized tensor
      TorchUtils::mergeTreeToTorchTensor(baseTmt.mTree, baseTmtTensor, false);
    torch::Tensor b = torch::fill(torch::zeros({w.sizes()[0], 1}), 0.01);
    tmt.tensor = (torch::matmul(w, baseTmtTensor) + b);
    // - Shift to keep mean birth and max pers
    meanBirthMaxPersShift(tmt.tensor, baseTmtTensor);
    // - Shift to avoid diagonal points
    belowDiagonalPointsShift(tmt.tensor, baseTmtTensor);
    //
    auto endLayer
      = (trackingLossDecoding_ ? noLayers_ : getLatentLayerIndex() + 1);
    if(trackingLossWeight_ != 0 and l < endLayer) {
      auto baseTensor
        = (l == 0 ? origins_[0].tensor : originsPrime_[l - 1].tensor);
      auto baseTensorDiag = baseTensor.reshape({-1, 2});
      auto basePersDiag = (baseTensorDiag.index({Slice(), 1})
                           - baseTensorDiag.index({Slice(), 0}));
      auto tmtTensorDiag = tmt.tensor.reshape({-1, 2});
      auto persDiag = (tmtTensorDiag.index({Slice(1, None), 1})
                       - tmtTensorDiag.index({Slice(1, None), 0}));
      int noK = std::min(baseTensorDiag.sizes()[0], tmtTensorDiag.sizes()[0]);
      auto topVal = baseTensorDiag.index({std::get<1>(basePersDiag.topk(noK))});
      auto indexes = std::get<1>(persDiag.topk(noK - 1)) + 1;
      indexes = torch::cat({torch::zeros(1), indexes}).to(torch::kLong);
      if(trackingLossInitRandomness_ != 0) {
        topVal = (1 - trackingLossInitRandomness_) * topVal
                 + trackingLossInitRandomness_ * tmtTensorDiag.index({indexes});
      }
      tmtTensorDiag.index_put_({indexes}, topVal);
    }
    // - Create tree structure
    initOutputBasisTreeStructure(
      tmt, baseTmt.mTree.tree.isJoinTree<float>(), baseTmt);
    if(normalizedWasserstein_)
      // Normalize tensor
      TorchUtils::mergeTreeToTorchTensor(tmt.mTree, tmt.tensor, true);
    // - Projection
    interpolationProjection(tmt);
  };
  torch::Tensor w = torch::zeros({dim, originSize});
  initOutputBasisOrigin(w, originsPrime_[l], origins_[l]);
  torch::Tensor w2;
  if(useDoubleInput_) {
    w2 = torch::zeros({dim2, origin2Size});
    initOutputBasisOrigin(w2, origins2Prime_[l], origins2_[l]);
  }

  // --- Compute output basis vectors
  printMsg("Compute output basis vectors", debug::Priority::DETAIL);
  initOutputBasisVectors(l, w, w2);
}

void ttk::MergeTreeAutoencoder::initOutputBasisVectors(unsigned int l,
                                                       torch::Tensor &w,
                                                       torch::Tensor &w2) {
  vSPrimeTensor_[l] = torch::matmul(w, vSTensor_[l]);
  if(useDoubleInput_)
    vS2PrimeTensor_[l] = torch::matmul(w2, vS2Tensor_[l]);
  if(normalizedWasserstein_) {
    normalizeVectors(originsPrime_[l].tensor, vSPrimeTensor_[l]);
    if(useDoubleInput_)
      normalizeVectors(origins2Prime_[l].tensor, vS2PrimeTensor_[l]);
  }
}

void ttk::MergeTreeAutoencoder::initOutputBasisVectors(unsigned int l,
                                                       unsigned int dim,
                                                       unsigned int dim2) {
  unsigned int originSize = origins_[l].tensor.sizes()[0];
  unsigned int origin2Size = 0;
  if(useDoubleInput_)
    origin2Size = origins2_[l].tensor.sizes()[0];
  torch::Tensor w = torch::zeros({dim, originSize});
  torch::nn::init::xavier_normal_(w);
  torch::Tensor w2 = torch::zeros({dim2, origin2Size});
  torch::nn::init::xavier_normal_(w2);
  initOutputBasisVectors(l, w, w2);
}

void ttk::MergeTreeAutoencoder::initInputBasisOrigin(
  std::vector<ftm::MergeTree<float>> &treesToUse,
  std::vector<ftm::MergeTree<float>> &trees2ToUse,
  double barycenterSizeLimitPercent,
  unsigned int barycenterMaxNoPairs,
  unsigned int barycenterMaxNoPairs2,
  TorchUtils::TorchMergeTree<float> &origin,
  TorchUtils::TorchMergeTree<float> &origin2,
  std::vector<double> &inputToBaryDistances,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &baryMatchings,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &baryMatchings2) {
  computeOneBarycenter<float>(treesToUse, origin.mTree, baryMatchings,
                              inputToBaryDistances, barycenterSizeLimitPercent,
                              useDoubleInput_);
  if(barycenterMaxNoPairs > 0)
    keepMostImportantPairs<float>(
      &(origin.mTree.tree), barycenterMaxNoPairs, true);
  // printMsg(origins_[l].mTree.tree.printTreeStats().str());
  if(useDoubleInput_) {
    std::vector<double> baryDistances2;
    computeOneBarycenter<float>(trees2ToUse, origin2.mTree, baryMatchings2,
                                baryDistances2, barycenterSizeLimitPercent,
                                useDoubleInput_, false);
    if(barycenterMaxNoPairs2 > 0)
      keepMostImportantPairs<float>(
        &(origin2.mTree.tree), barycenterMaxNoPairs2, true);
    // printMsg(origins2_[l].mTree.tree.printTreeStats().str());
    for(unsigned int i = 0; i < inputToBaryDistances.size(); ++i)
      inputToBaryDistances[i]
        = mixDistances(inputToBaryDistances[i], baryDistances2[i]);
    // verifyMinMaxPair(origin, origin2);
  }

  TorchUtils::getParentsVector(origin.mTree, origin.parentsOri);
  TorchUtils::mergeTreeToTorchTensor<float>(
    origin.mTree, origin.tensor, origin.nodeCorr, normalizedWasserstein_);
  if(useDoubleInput_) {
    TorchUtils::getParentsVector(origin2.mTree, origin2.parentsOri);
    TorchUtils::mergeTreeToTorchTensor<float>(
      origin2.mTree, origin2.tensor, origin2.nodeCorr, normalizedWasserstein_);
  }
}

void ttk::MergeTreeAutoencoder::initInputBasisVectors(
  std::vector<TorchUtils::TorchMergeTree<float>> &tmTreesToUse,
  std::vector<TorchUtils::TorchMergeTree<float>> &tmTrees2ToUse,
  std::vector<ftm::MergeTree<float>> &treesToUse,
  std::vector<ftm::MergeTree<float>> &trees2ToUse,
  TorchUtils::TorchMergeTree<float> &origin,
  TorchUtils::TorchMergeTree<float> &origin2,
  unsigned int noVectors,
  std::vector<std::vector<torch::Tensor>> &allAlphasInit,
  unsigned int l,
  std::vector<double> &inputToBaryDistances,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &baryMatchings,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &baryMatchings2,
  torch::Tensor &vSTensor,
  torch::Tensor &vS2Tensor) {
  // --- Initialized vectors projection function to avoid collinearity
  auto initializedVectorsProjection
    = [=](int ttkNotUsed(_geodesicNumber),
          ftm::MergeTree<float> &ttkNotUsed(_barycenter),
          std::vector<std::vector<double>> &_v,
          std::vector<std::vector<double>> &ttkNotUsed(_v2),
          std::vector<std::vector<std::vector<double>>> &_vS,
          std::vector<std::vector<std::vector<double>>> &ttkNotUsed(_v2s),
          ftm::MergeTree<float> &ttkNotUsed(_barycenter2),
          std::vector<std::vector<double>> &ttkNotUsed(_trees2V),
          std::vector<std::vector<double>> &ttkNotUsed(_trees2V2),
          std::vector<std::vector<std::vector<double>>> &ttkNotUsed(_trees2Vs),
          std::vector<std::vector<std::vector<double>>> &ttkNotUsed(_trees2V2s),
          bool ttkNotUsed(_useSecondInput),
          unsigned int ttkNotUsed(_noProjectionStep)) {
        std::vector<double> scaledV, scaledVSi;
        Geometry::flattenMultiDimensionalVector(_v, scaledV);
        Geometry::scaleVector(
          scaledV, 1.0 / Geometry::magnitude(scaledV), scaledV);
        for(unsigned int i = 0; i < _vS.size(); ++i) {
          Geometry::flattenMultiDimensionalVector(_vS[i], scaledVSi);
          Geometry::scaleVector(
            scaledVSi, 1.0 / Geometry::magnitude(scaledVSi), scaledVSi);
          auto prod = Geometry::dotProduct(scaledV, scaledVSi);
          double tol = 0.01;
          if(prod <= -1.0 + tol or prod >= 1.0 - tol) {
            // Reset vector to initialize it again
            for(unsigned int j = 0; j < _v.size(); ++j)
              for(unsigned int k = 0; k < _v[j].size(); ++k)
                _v[j][k] = 0;
            break;
          }
        }
        return 0;
      };

  // --- Init vectors
  std::vector<std::vector<double>> inputToGeodesicsDistances;
  std::vector<std::vector<std::vector<double>>> vS, v2s, trees2Vs, trees2V2s;
  std::stringstream ss;
  for(unsigned int vecNum = 0; vecNum < noVectors; ++vecNum) {
    ss.str("");
    ss << "Compute vectors " << vecNum;
    printMsg(ss.str(), debug::Priority::VERBOSE);
    std::vector<std::vector<double>> v1, v2, trees2V1, trees2V2;
    int newVectorOffset = 0;
    bool projectInitializedVectors = true;
    int bestIndex = MergeTreeAxesAlgorithmBase::initVectors<float>(
      vecNum, origin.mTree, treesToUse, origin2.mTree, trees2ToUse, v1, v2,
      trees2V1, trees2V2, newVectorOffset, inputToBaryDistances, baryMatchings,
      baryMatchings2, inputToGeodesicsDistances, vS, v2s, trees2Vs, trees2V2s,
      projectInitializedVectors, initializedVectorsProjection);
    vS.emplace_back(v1);
    v2s.emplace_back(v2);
    trees2Vs.emplace_back(trees2V1);
    trees2V2s.emplace_back(trees2V2);

    ss.str("");
    ss << "bestIndex = " << bestIndex;
    printMsg(ss.str(), debug::Priority::VERBOSE);

    // Update inputToGeodesicsDistances
    printMsg("Update inputToGeodesicsDistances", debug::Priority::VERBOSE);
    inputToGeodesicsDistances.resize(1, std::vector<double>(treesToUse.size()));
    // torch::Tensor vSTensorT, vS2TensorT;
    if(bestIndex == -1 and normalizedWasserstein_) {
      normalizeVectors(origin, vS[vS.size() - 1]);
      if(useDoubleInput_)
        normalizeVectors(origin2, trees2Vs[vS.size() - 1]);
    }
    TorchUtils::geodesicVectorsToTorchTensor(origin.mTree, vS, vSTensor);
    if(useDoubleInput_) {
      TorchUtils::geodesicVectorsToTorchTensor(
        origin2.mTree, trees2Vs, vS2Tensor);
    }
    TorchUtils::TorchMergeTree<float> dummyTmt;
    std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>
      dummyBaryMatching2;
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) \
  num_threads(this->threadNumber_) if(parallelize_)
#endif
    for(unsigned int i = 0; i < treesToUse.size(); ++i) {
      auto &tmt2ToUse = (not useDoubleInput_ ? dummyTmt : tmTrees2ToUse[i]);
      if(not euclideanVectorsInit_) {
        unsigned int k = k_;
        auto newAlpha = torch::ones({1, 1});
        if(bestIndex == -1) {
          newAlpha = torch::zeros({1, 1});
          // newAlpha = torch::randn({1, 1});
          // newAlpha = torch::ones({1, 1});
        }
        allAlphasInit[i][l] = (allAlphasInit[i][l].defined()
                                 ? torch::cat({allAlphasInit[i][l], newAlpha})
                                 : newAlpha);
        torch::Tensor bestAlphas;
        bool isCalled = true;
        inputToGeodesicsDistances[0][i] = assignmentOneData(
          tmTreesToUse[i], origin, vSTensor, tmt2ToUse, origin2, vS2Tensor, k,
          allAlphasInit[i][l], bestAlphas, isCalled);
        allAlphasInit[i][l] = bestAlphas.detach();
      } else {
        auto &baryMatching2ToUse
          = (not useDoubleInput_ ? dummyBaryMatching2 : baryMatchings2[i]);
        torch::Tensor alphas;
        computeAlphas(tmTreesToUse[i], origin, vSTensor, origin,
                      baryMatchings[i], tmt2ToUse, origin2, vS2Tensor, origin2,
                      baryMatching2ToUse, alphas);
        TorchUtils::TorchMergeTree<float> interpolated, interpolated2;
        getMultiInterpolation(origin, vSTensor, alphas, interpolated);
        if(useDoubleInput_)
          getMultiInterpolation(origin2, vS2Tensor, alphas, interpolated2);
        torch::Tensor tensorDist;
        bool doSqrt = true;
        getDifferentiableDistanceFromMatchings(
          interpolated, tmTreesToUse[i], interpolated2, tmt2ToUse,
          baryMatchings[i], baryMatching2ToUse, tensorDist, doSqrt);
        inputToGeodesicsDistances[0][i] = tensorDist.item<double>();
        allAlphasInit[i][l] = alphas.detach();
      }
    }
  }
}

void ttk::MergeTreeAutoencoder::initClusteringLossParameters() {
  unsigned int l = getLatentLayerIndex();
  unsigned int noCentroids
    = std::set<unsigned int>(clusterAsgn_.begin(), clusterAsgn_.end()).size();
  latentCentroids_.resize(noCentroids);
  for(unsigned int c = 0; c < noCentroids; ++c) {
    unsigned int firstIndex = std::numeric_limits<unsigned int>::max();
    for(unsigned int i = 0; i < clusterAsgn_.size(); ++i) {
      if(clusterAsgn_[i] == c) {
        firstIndex = i;
        break;
      }
    }
    if(firstIndex >= allAlphas_.size()) {
      printWrn("no data found for cluster " + std::to_string(c));
      // TODO init random centroid
    }
    latentCentroids_[c] = allAlphas_[firstIndex][l].detach().clone();
    float noData = 1;
    for(unsigned int i = 0; i < allAlphas_.size(); ++i) {
      if(i == firstIndex)
        continue;
      if(clusterAsgn_[i] == c) {
        latentCentroids_[c] += allAlphas_[i][l];
        ++noData;
      }
    }
    latentCentroids_[c] /= torch::tensor(noData);
    latentCentroids_[c] = latentCentroids_[c].detach();
    latentCentroids_[c].requires_grad_(true);
  }
}

float ttk::MergeTreeAutoencoder::initParameters(
  std::vector<TorchUtils::TorchMergeTree<float>> &trees,
  std::vector<TorchUtils::TorchMergeTree<float>> &trees2,
  bool computeReconstructionError) {
  // ----- Init variables
  // noLayers_ = number of encoder layers + number of decoder layers + the
  // latent layer + the output layer
  noLayers_ = encoderNoLayers_ * 2 + 1 + 1;
  if(encoderNoLayers_ <= -1)
    noLayers_ = 1;
  std::vector<double> layersOriginPrimeSizePercent(noLayers_);
  std::vector<unsigned int> layersNoGeodesics(noLayers_);
  if(noLayers_ <= 2) {
    layersNoGeodesics[0] = numberOfGeodesics_;
    layersOriginPrimeSizePercent[0] = latentSpaceOriginPrimeSizePercent_;
    if(noLayers_ == 2) {
      layersNoGeodesics[1] = inputNumberOfGeodesics_;
      layersOriginPrimeSizePercent[1] = barycenterSizeLimitPercent_;
    }
  } else {
    for(unsigned int l = 0; l < noLayers_ / 2; ++l) {
      double alpha = (double)(l) / (noLayers_ / 2 - 1);
      unsigned int noGeodesics
        = (1 - alpha) * inputNumberOfGeodesics_ + alpha * numberOfGeodesics_;
      layersNoGeodesics[l] = noGeodesics;
      layersNoGeodesics[noLayers_ - 1 - l] = noGeodesics;
      double originPrimeSizePercent
        = (1 - alpha) * inputOriginPrimeSizePercent_
          + alpha * latentSpaceOriginPrimeSizePercent_;
      layersOriginPrimeSizePercent[l] = originPrimeSizePercent;
      layersOriginPrimeSizePercent[noLayers_ - 1 - l] = originPrimeSizePercent;
    }
    if(scaleLayerAfterLatent_)
      layersNoGeodesics[noLayers_ / 2]
        = (layersNoGeodesics[noLayers_ / 2 - 1]
           + layersNoGeodesics[noLayers_ / 2 + 1])
          / 2.0;
  }

  std::vector<ftm::FTMTree_MT *> ftmTrees(trees.size()),
    ftmTrees2(trees2.size());
  for(unsigned int i = 0; i < trees.size(); ++i)
    ftmTrees[i] = &(trees[i].mTree.tree);
  for(unsigned int i = 0; i < trees2.size(); ++i)
    ftmTrees2[i] = &(trees2[i].mTree.tree);
  auto sizeMetric = getSizeLimitMetric(ftmTrees);
  auto sizeMetric2 = getSizeLimitMetric(ftmTrees2);
  auto getDim = [](double _sizeMetric, double _percent) {
    unsigned int dim = std::max((int)(_sizeMetric * _percent / 100.0), 2) * 2;
    return dim;
  };

  // ----- Resize parameters
  origins_.resize(noLayers_);
  originsPrime_.resize(noLayers_);
  vSTensor_.resize(noLayers_);
  vSPrimeTensor_.resize(noLayers_);
  if(trees2.size() != 0) {
    origins2_.resize(noLayers_);
    origins2Prime_.resize(noLayers_);
    vS2Tensor_.resize(noLayers_);
    vS2PrimeTensor_.resize(noLayers_);
  }

  // ----- Compute parameters of each layer
  bool fullSymmetricAE = fullSymmetricAE_;
  bool outputBasisActivation = activateOutputInit_;

  std::vector<TorchUtils::TorchMergeTree<float>> recs, recs2;
  std::vector<std::vector<torch::Tensor>> allAlphasInit(
    trees.size(), std::vector<torch::Tensor>(noLayers_));
  for(unsigned int l = 0; l < noLayers_; ++l) {
    printMsg(debug::Separator::L2, debug::Priority::DETAIL);
    std::stringstream ss;
    ss << "Init Layer " << l;
    printMsg(ss.str(), debug::Priority::DETAIL);

    // --- Init Input Basis
    if(l < (unsigned int)(noLayers_ / 2) or not fullSymmetricAE
       or (noLayers_ <= 2 and not fullSymmetricAE)) {
      // TODO is there a way to avoid copy of merge trees?
      std::vector<ftm::MergeTree<float>> treesToUse, trees2ToUse;
      for(unsigned int i = 0; i < trees.size(); ++i) {
        treesToUse.emplace_back((l == 0 ? trees[i].mTree : recs[i].mTree));
        if(trees2.size() != 0)
          trees2ToUse.emplace_back((l == 0 ? trees2[i].mTree : recs2[i].mTree));
      }

      // - Compute origin
      printMsg("Compute origin...", debug::Priority::DETAIL);
      Timer t_origin;
      std::vector<double> inputToBaryDistances;
      std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
        baryMatchings, baryMatchings2;
      if(l != 0 or not origins_[0].tensor.defined()) {
        double sizeLimit = (l == 0 ? barycenterSizeLimitPercent_ : 0);
        unsigned int maxNoPairs
          = (l == 0 ? 0 : originsPrime_[l - 1].tensor.sizes()[0] / 2);
        unsigned int maxNoPairs2
          = (l == 0 or not useDoubleInput_
               ? 0
               : origins2Prime_[l - 1].tensor.sizes()[0] / 2);
        initInputBasisOrigin(treesToUse, trees2ToUse, sizeLimit, maxNoPairs,
                             maxNoPairs2, origins_[l], origins2_[l],
                             inputToBaryDistances, baryMatchings,
                             baryMatchings2);
        if(l == 0) {
          baryMatchings_L0_ = baryMatchings;
          baryMatchings2_L0_ = baryMatchings2;
          inputToBaryDistances_L0_ = inputToBaryDistances;
        }
      } else {
        baryMatchings = baryMatchings_L0_;
        baryMatchings2 = baryMatchings2_L0_;
        inputToBaryDistances = inputToBaryDistances_L0_;
      }
      printMsg("Compute origin time", 1, t_origin.getElapsedTime(),
               threadNumber_, debug::LineMode::NEW, debug::Priority::DETAIL);

      // - Compute vectors
      printMsg("Compute vectors...", debug::Priority::DETAIL);
      Timer t_vectors;
      auto &tmTreesToUse = (l == 0 ? trees : recs);
      auto &tmTrees2ToUse = (l == 0 ? trees2 : recs2);
      initInputBasisVectors(tmTreesToUse, tmTrees2ToUse, treesToUse,
                            trees2ToUse, origins_[l], origins2_[l],
                            layersNoGeodesics[l], allAlphasInit, l,
                            inputToBaryDistances, baryMatchings, baryMatchings2,
                            vSTensor_[l], vS2Tensor_[l]);
      printMsg("Compute vectors time", 1, t_vectors.getElapsedTime(),
               threadNumber_, debug::LineMode::NEW, debug::Priority::DETAIL);
    } else {
      // - Copy output tensors of the opposite layer (full symmetric init)
      printMsg(
        "Copy output tensors of the opposite layer", debug::Priority::DETAIL);
      unsigned int middle = noLayers_ / 2;
      unsigned int l_opp = middle - (l - middle + 1);
      TorchUtils::copyTorchMergeTree(originsPrime_[l_opp], origins_[l]);
      TorchUtils::copyTensor(vSPrimeTensor_[l_opp], vSTensor_[l]);
      if(trees2.size() != 0) {
        if(fullSymmetricAE) {
          TorchUtils::copyTorchMergeTree(origins2Prime_[l_opp], origins2_[l]);
          TorchUtils::copyTensor(vS2PrimeTensor_[l_opp], vS2Tensor_[l]);
        }
      }
      for(unsigned int i = 0; i < trees.size(); ++i)
        allAlphasInit[i][l] = allAlphasInit[i][l_opp];
      // allAlphasInit[i][l] = torch::zeros_like(allAlphasInit[i][l_opp]);
    }

    // --- Init Output Basis
    auto initOutputBasisSpecialCase
      = [this, &l, &layersNoGeodesics, &trees, &trees2]() {
          // - Compute Origin
          printMsg("Compute output basis origin", debug::Priority::DETAIL);
          TorchUtils::copyTorchMergeTree(origins_[0], originsPrime_[l]);
          if(useDoubleInput_)
            TorchUtils::copyTorchMergeTree(origins2_[0], origins2Prime_[l]);
          // - Compute vectors
          printMsg("Compute output basis vectors", debug::Priority::DETAIL);
          if(layersNoGeodesics[l] != layersNoGeodesics[0]) {
            // TODO is there a way to avoid copy of merge trees?
            std::vector<ftm::MergeTree<float>> treesToUse, trees2ToUse;
            for(unsigned int i = 0; i < trees.size(); ++i) {
              treesToUse.emplace_back(trees[i].mTree);
              if(useDoubleInput_)
                trees2ToUse.emplace_back(trees2[i].mTree);
            }
            std::vector<std::vector<torch::Tensor>> allAlphasInitT(
              trees.size(), std::vector<torch::Tensor>(noLayers_));
            initInputBasisVectors(
              trees, trees2, treesToUse, trees2ToUse, originsPrime_[l],
              origins2Prime_[l], layersNoGeodesics[l], allAlphasInitT, l,
              inputToBaryDistances_L0_, baryMatchings_L0_, baryMatchings2_L0_,
              vSPrimeTensor_[l], vS2PrimeTensor_[l]);
          } else {
            TorchUtils::copyTensor(vSTensor_[0], vSPrimeTensor_[l]);
            if(useDoubleInput_)
              TorchUtils::copyTensor(vS2Tensor_[0], vS2PrimeTensor_[l]);
          }
        };

    if((noLayers_ == 2 and l == 1) or noLayers_ == 1) {
      // -- Special case
      initOutputBasisSpecialCase();
    } else if(l < (unsigned int)(noLayers_ / 2)) {
      initOutputBasis(l, getDim(sizeMetric, layersOriginPrimeSizePercent[l]),
                      getDim(sizeMetric2, layersOriginPrimeSizePercent[l]));
    } else {
      // - Copy input tensors of the opposite layer (symmetric init)
      printMsg(
        "Copy input tensors of the opposite layer", debug::Priority::DETAIL);
      unsigned int middle = noLayers_ / 2;
      unsigned int l_opp = middle - (l - middle + 1);
      TorchUtils::copyTorchMergeTree(origins_[l_opp], originsPrime_[l]);
      if(trees2.size() != 0)
        TorchUtils::copyTorchMergeTree(origins2_[l_opp], origins2Prime_[l]);
      if(l == (unsigned int)(noLayers_) / 2 and scaleLayerAfterLatent_) {
        unsigned int dim2
          = (trees2.size() != 0 ? origins2Prime_[l].tensor.sizes()[0] : 0);
        initOutputBasisVectors(l, originsPrime_[l].tensor.sizes()[0], dim2);
      } else {
        TorchUtils::copyTensor(vSTensor_[l_opp], vSPrimeTensor_[l]);
        if(trees2.size() != 0)
          TorchUtils::copyTensor(vS2Tensor_[l_opp], vS2PrimeTensor_[l]);
      }
    }

    // --- Get reconstructed
    printMsg("Get reconstructed", debug::Priority::DETAIL);
    recs.resize(trees.size());
    recs2.resize(trees.size());
    unsigned int i = 0;
    unsigned int noReset = 0;
    while(i < trees.size()) {
      outputBasisReconstruction(originsPrime_[l], vSPrimeTensor_[l],
                                origins2Prime_[l], vS2PrimeTensor_[l],
                                allAlphasInit[i][l], recs[i], recs2[i],
                                outputBasisActivation);
      if(recs[i].mTree.tree.getRealNumberOfNodes() == 0) {
        printMsg("Reset output basis", debug::Priority::DETAIL);
        if((noLayers_ == 2 and l == 1) or noLayers_ == 1) {
          initOutputBasisSpecialCase();
        } else if(l < (unsigned int)(noLayers_ / 2)) {
          initOutputBasis(l,
                          getDim(sizeMetric, layersOriginPrimeSizePercent[l]),
                          getDim(sizeMetric2, layersOriginPrimeSizePercent[l]));
        } else {
          // TODO fix this
          printErr("recs[i].mTree.tree.getRealNumberOfNodes() == 0");
          std::cout << "layer " << l << std::endl;
          return std::numeric_limits<float>::max();
        }
        i = 0;
        ++noReset;
        if(noReset >= 100) {
          printWrn("[initParameters] noReset >= 100");
          return std::numeric_limits<float>::max();
        }
      }
      ++i;
    }
  }
  allAlphas_ = allAlphasInit;

  // Init clustering parameters if needed
  if(clusteringLossWeight_ != 0)
    initClusteringLossParameters();

  // Compute error
  float error = 0.0, recLoss = 0.0;
  if(computeReconstructionError) {
    printMsg("Compute error", debug::Priority::DETAIL);
    std::vector<unsigned int> indexes(trees.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    // TODO forward only if necessary
    // if(not outputBasisActivation and activate_) {
    unsigned int k = k_;
    std::vector<std::vector<torch::Tensor>> bestAlphas;
    std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> layersOuts,
      layersOuts2;
    std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
      matchings, matchings2;
    bool reset
      = forwardStep(trees, trees2, indexes, k, allAlphasInit,
                    computeReconstructionError, recs, recs2, bestAlphas,
                    layersOuts, layersOuts2, matchings, matchings2, recLoss);
    if(reset) {
      printWrn("[initParameters] forwardStep reset");
      return std::numeric_limits<float>::max();
    }
    // allAlphas_ = bestAlphas;
    // }
    error = recLoss * reconstructionLossWeight_;
    if(metricLossWeight_ != 0) {
      torch::Tensor metricLoss;
      computeMetricLoss(layersOuts, layersOuts2, allAlphas_, distanceMatrix_,
                        indexes, metricLoss);
      baseRecLoss_ = std::numeric_limits<double>::max();
      metricLoss *= metricLossWeight_
                    * getCustomLossDynamicWeight(recLoss, baseRecLoss_);
      error += metricLoss.item<float>();
    }
    if(clusteringLossWeight_ != 0) {
      torch::Tensor clusteringLoss, asgn;
      computeClusteringLoss(allAlphas_, indexes, clusteringLoss, asgn);
      baseRecLoss_ = std::numeric_limits<double>::max();
      clusteringLoss *= clusteringLossWeight_
                        * getCustomLossDynamicWeight(recLoss, baseRecLoss_);
      error += clusteringLoss.item<float>();
    }
    if(trackingLossWeight_ != 0) {
      torch::Tensor trackingLoss;
      computeTrackingLoss(trackingLoss);
      // baseRecLoss_ = std::numeric_limits<double>::max();
      trackingLoss *= trackingLossWeight_;
      // * getCustomLossDynamicWeight(recLoss, baseRecLoss_);
      error += trackingLoss.item<float>();
    }
  }
  return error;
}

void ttk::MergeTreeAutoencoder::initStep(
  std::vector<TorchUtils::TorchMergeTree<float>> &trees,
  std::vector<TorchUtils::TorchMergeTree<float>> &trees2) {
  origins_.clear();
  originsPrime_.clear();
  vSTensor_.clear();
  vSPrimeTensor_.clear();
  origins2_.clear();
  origins2Prime_.clear();
  vS2Tensor_.clear();
  vS2PrimeTensor_.clear();

  float bestError = std::numeric_limits<float>::max();
  std::vector<torch::Tensor> bestVSTensor, bestVSPrimeTensor, bestVS2Tensor,
    bestVS2PrimeTensor, bestLatentCentroids;
  std::vector<TorchUtils::TorchMergeTree<float>> bestOrigins, bestOriginsPrime,
    bestOrigins2, bestOrigins2Prime;
  std::vector<std::vector<torch::Tensor>> bestAlphasInit;
  for(unsigned int n = 0; n < noInit_; ++n) {
    // Init parameters
    float error = initParameters(trees, trees2, (noInit_ != 1));
    // Save best parameters
    if(noInit_ != 1) {
      std::stringstream ss;
      ss << "Init error = " << error;
      printMsg(ss.str());
      if(error < bestError) {
        bestError = error;
        copyParams(origins_, originsPrime_, vSTensor_, vSPrimeTensor_,
                   origins2_, origins2Prime_, vS2Tensor_, vS2PrimeTensor_,
                   allAlphas_, bestOrigins, bestOriginsPrime, bestVSTensor,
                   bestVSPrimeTensor, bestOrigins2, bestOrigins2Prime,
                   bestVS2Tensor, bestVS2PrimeTensor, bestAlphasInit);
        bestLatentCentroids.resize(latentCentroids_.size());
        for(unsigned int i = 0; i < latentCentroids_.size(); ++i)
          ttk::TorchUtils::copyTensor(
            latentCentroids_[i], bestLatentCentroids[i]);
      }
    }
  }
  // TODO this copy can be avoided if initParameters takes dummy tensors to fill
  // as parameters and then copy to the member tensors when a better init is
  // found.
  if(noInit_ != 1) {
    // Put back best parameters
    std::stringstream ss;
    ss << "Best init error = " << bestError;
    printMsg(ss.str());
    copyParams(bestOrigins, bestOriginsPrime, bestVSTensor, bestVSPrimeTensor,
               bestOrigins2, bestOrigins2Prime, bestVS2Tensor,
               bestVS2PrimeTensor, bestAlphasInit, origins_, originsPrime_,
               vSTensor_, vSPrimeTensor_, origins2_, origins2Prime_, vS2Tensor_,
               vS2PrimeTensor_, allAlphas_);
    latentCentroids_.resize(bestLatentCentroids.size());
    for(unsigned int i = 0; i < bestLatentCentroids.size(); ++i)
      ttk::TorchUtils::copyTensor(bestLatentCentroids[i], latentCentroids_[i]);
  }

  for(unsigned int l = 0; l < noLayers_; ++l) {
    origins_[l].tensor.requires_grad_(true);
    originsPrime_[l].tensor.requires_grad_(true);
    vSTensor_[l].requires_grad_(true);
    vSPrimeTensor_[l].requires_grad_(true);
    if(trees2.size() != 0) {
      origins2_[l].tensor.requires_grad_(true);
      origins2Prime_[l].tensor.requires_grad_(true);
      vS2Tensor_[l].requires_grad_(true);
      vS2PrimeTensor_[l].requires_grad_(true);
    }

    // Print
    printMsg(debug::Separator::L2);
    std::stringstream ss;
    ss << "Layer " << l;
    printMsg(ss.str());
    /*std::cout << origins_[l].tensor << std::endl;
    std::cout << originsPrime_[l].tensor << std::endl;
    std::cout << vSTensor_[l] << std::endl;
    std::cout << vSPrimeTensor_[l] << std::endl;*/
    if(isTreeHasBigValues(origins_[l].mTree, bigValuesThreshold_)) {
      std::cout << "origins_[" << l << "] has big values!" << std::endl;
      ttk::printPairs(origins_[l].mTree);
    }
    if(isTreeHasBigValues(originsPrime_[l].mTree, bigValuesThreshold_)) {
      std::cout << "originsPrime_[" << l << "] has big values!" << std::endl;
      ttk::printPairs(originsPrime_[l].mTree);
    }
    ss.str("");
    ss << "vS size   = " << vSTensor_[l].sizes();
    printMsg(ss.str());
    ss.str("");
    ss << "vS' size  = " << vSPrimeTensor_[l].sizes();
    printMsg(ss.str());
    if(trees2.size() != 0) {
      ss.str("");
      ss << "vS2 size  = " << vS2Tensor_[l].sizes();
      printMsg(ss.str());
      ss.str("");
      ss << "vS2' size = " << vS2PrimeTensor_[l].sizes();
      printMsg(ss.str());
    }
    /*for(unsigned int i = 0; i < trees.size(); ++i) {
      std::cout << "Alphas layer " << l << " tree " << i << std::endl;
      std::cout << allAlphas_[i][l] << std::endl;
    }*/
  }

  // Init Clustering Loss Parameters
  if(clusteringLossWeight_ != 0)
    initClusteringLossParameters();
}

//  ---------------------------------------------------------------------------
//  --- Interpolation
//  ---------------------------------------------------------------------------
void ttk::MergeTreeAutoencoder::interpolationDiagonalProjection(
  TorchUtils::TorchMergeTree<float> &interpolation) {
  torch::Tensor diagTensor = interpolation.tensor.reshape({-1, 2});
  if(interpolation.tensor.requires_grad())
    diagTensor = diagTensor.detach();

  torch::Tensor birthTensor = diagTensor.index({Slice(), 0});
  torch::Tensor deathTensor = diagTensor.index({Slice(), 1});

  torch::Tensor indexer = (birthTensor > deathTensor);

  torch::Tensor allProj = (birthTensor + deathTensor) / 2.0;
  allProj = allProj.index({indexer});
  allProj = allProj.reshape({-1, 1});

  diagTensor.index_put_({indexer}, allProj);
}

void ttk::MergeTreeAutoencoder::interpolationNestingProjection(
  TorchUtils::TorchMergeTree<float> &interpolation) {
  torch::Tensor diagTensor = interpolation.tensor.reshape({-1, 2});
  if(interpolation.tensor.requires_grad())
    diagTensor = diagTensor.detach();

  torch::Tensor birthTensor = diagTensor.index({Slice(1, None), 0});
  torch::Tensor deathTensor = diagTensor.index({Slice(1, None), 1});

  torch::Tensor birthIndexer = (birthTensor < 0);
  torch::Tensor deathIndexer = (deathTensor < 0);
  birthTensor.index_put_(
    {birthIndexer}, torch::zeros_like(birthTensor.index({birthIndexer})));
  deathTensor.index_put_(
    {deathIndexer}, torch::zeros_like(deathTensor.index({deathIndexer})));

  birthIndexer = (birthTensor > 1);
  deathIndexer = (deathTensor > 1);
  birthTensor.index_put_(
    {birthIndexer}, torch::ones_like(birthTensor.index({birthIndexer})));
  deathTensor.index_put_(
    {deathIndexer}, torch::ones_like(deathTensor.index({deathIndexer})));
}

void ttk::MergeTreeAutoencoder::interpolationProjection(
  TorchUtils::TorchMergeTree<float> &interpolation) {
  interpolationDiagonalProjection(interpolation);
  if(normalizedWasserstein_)
    interpolationNestingProjection(interpolation);

  ftm::MergeTree<float> interpolationNew;
  bool noRoot = TorchUtils::torchTensorToMergeTree<float>(
    interpolation, normalizedWasserstein_, interpolationNew);
  if(noRoot)
    printWrn("[interpolationProjection] no root found");
  interpolation.mTree = copyMergeTree(interpolationNew);

  persistenceThresholding<float>(&(interpolation.mTree.tree), 0.001);
  // TODO cleanMergeTree ?

  if(isThereMissingPairs(interpolation) and isPersistenceDiagram_)
    printWrn("[getMultiInterpolation] missing pairs");
}

void ttk::MergeTreeAutoencoder::getMultiInterpolation(
  TorchUtils::TorchMergeTree<float> &origin,
  torch::Tensor &vS,
  torch::Tensor &alphas,
  TorchUtils::TorchMergeTree<float> &interpolation) {
  TorchUtils::copyTorchMergeTree<float>(origin, interpolation);
  interpolation.tensor = origin.tensor + torch::matmul(vS, alphas);
  interpolationProjection(interpolation);
}

//  ---------------------------------------------------------------------------
//  --- Forward
//  ---------------------------------------------------------------------------
void ttk::MergeTreeAutoencoder::getAlphasOptimizationTensors(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &origin,
  torch::Tensor &vSTensor,
  TorchUtils::TorchMergeTree<float> &interpolated,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
  torch::Tensor &reorderedTreeTensor,
  torch::Tensor &deltaOrigin,
  torch::Tensor &deltaA,
  torch::Tensor &originTensor_f,
  torch::Tensor &vSTensor_f) {
  // Create matching indexing
  std::vector<int> tensorMatching;
  TorchUtils::getTensorMatching(interpolated, tree, matching, tensorMatching);

  torch::Tensor indexes = torch::tensor(tensorMatching);
  torch::Tensor projIndexer = (indexes == -1).reshape({-1, 1});

  dataReorderingGivenMatching(
    origin, tree, projIndexer, indexes, reorderedTreeTensor, deltaOrigin);

  // Create axes projection given matching
  deltaA = vSTensor.transpose(0, 1).reshape({vSTensor.sizes()[1], -1, 2});
  deltaA = (deltaA.index({Slice(), Slice(), 0})
            + deltaA.index({Slice(), Slice(), 1}))
           / 2.0;
  deltaA = torch::stack({deltaA, deltaA}, 2);
  deltaA = deltaA * projIndexer;
  deltaA = deltaA.reshape({vSTensor.sizes()[1], -1}).transpose(0, 1);

  //
  originTensor_f = origin.tensor;
  vSTensor_f = vSTensor;
}

void ttk::MergeTreeAutoencoder::computeAlphas(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &origin,
  torch::Tensor &vSTensor,
  TorchUtils::TorchMergeTree<float> &interpolated,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
  TorchUtils::TorchMergeTree<float> &tree2,
  TorchUtils::TorchMergeTree<float> &origin2,
  torch::Tensor &vS2Tensor,
  TorchUtils::TorchMergeTree<float> &interpolated2,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching2,
  torch::Tensor &alphasOut) {
  torch::Tensor reorderedTreeTensor, deltaOrigin, deltaA, originTensor_f,
    vSTensor_f;
  getAlphasOptimizationTensors(tree, origin, vSTensor, interpolated, matching,
                               reorderedTreeTensor, deltaOrigin, deltaA,
                               originTensor_f, vSTensor_f);

  if(useDoubleInput_) {
    torch::Tensor reorderedTree2Tensor, deltaOrigin2, deltaA2, origin2Tensor_f,
      vS2Tensor_f;
    getAlphasOptimizationTensors(tree2, origin2, vS2Tensor, interpolated2,
                                 matching2, reorderedTree2Tensor, deltaOrigin2,
                                 deltaA2, origin2Tensor_f, vS2Tensor_f);
    vSTensor_f = torch::cat({vSTensor_f, vS2Tensor_f});
    deltaA = torch::cat({deltaA, deltaA2});
    reorderedTreeTensor
      = torch::cat({reorderedTreeTensor, reorderedTree2Tensor});
    originTensor_f = torch::cat({originTensor_f, origin2Tensor_f});
    deltaOrigin = torch::cat({deltaOrigin, deltaOrigin2});
  }

  torch::Tensor r_axes = vSTensor_f - deltaA;
  torch::Tensor r_data = reorderedTreeTensor - originTensor_f + deltaOrigin;

  // alphasOut = torch::matmul(torch::linalg::pinv(r_axes), r_data);
  // auto driver = c10::nullopt;
  // auto driver = (deterministic_ ? "gelsd" : "gelsy");
  auto driver = "gelsd";
  alphasOut
    = std::get<0>(torch::linalg::lstsq(r_axes, r_data, c10::nullopt, driver));

  alphasOut.reshape({-1, 1});
}

float ttk::MergeTreeAutoencoder::assignmentOneData(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &origin,
  torch::Tensor &vSTensor,
  TorchUtils::TorchMergeTree<float> &tree2,
  TorchUtils::TorchMergeTree<float> &origin2,
  torch::Tensor &vS2Tensor,
  unsigned int k,
  torch::Tensor &alphasInit,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &bestMatching,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &bestMatching2,
  torch::Tensor &bestAlphas,
  bool isCalled) {
  torch::Tensor alphas, oldAlphas;
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> matching, matching2;
  float bestDistance = std::numeric_limits<float>::max();
  TorchUtils::TorchMergeTree<float> interpolated, interpolated2;
  unsigned int i = 0;
  auto reset = [&]() {
    alphasInit = torch::randn_like(alphas);
    i = 0;
  };
  unsigned int noUpdate = 0;
  unsigned int noReset = 0;
  while(i < k) {
    if(i == 0) {
      if(alphasInit.defined())
        alphas = alphasInit;
      else
        alphas = torch::zeros({vSTensor.sizes()[1], 1});
    } else {
      computeAlphas(tree, origin, vSTensor, interpolated, matching, tree2,
                    origin2, vS2Tensor, interpolated2, matching2, alphas);
      if(oldAlphas.defined() and alphas.defined() and alphas.equal(oldAlphas)
         and i != 1) {
        break;
      }
    }
    TorchUtils::copyTensor(alphas, oldAlphas);
    getMultiInterpolation(origin, vSTensor, alphas, interpolated);
    if(useDoubleInput_)
      getMultiInterpolation(origin2, vS2Tensor, alphas, interpolated2);
    if(interpolated.mTree.tree.getRealNumberOfNodes() == 0
       or (useDoubleInput_
           and interpolated2.mTree.tree.getRealNumberOfNodes() == 0)) {
      ++noReset;
      if(noReset >= 100)
        printWrn("[assignmentOneData] noReset >= 100");
      reset();
      continue;
    }
    float distance;
    computeOneDistance<float>(interpolated.mTree, tree.mTree, matching,
                              distance, isCalled, useDoubleInput_);
    if(useDoubleInput_) {
      float distance2;
      computeOneDistance<float>(interpolated2.mTree, tree2.mTree, matching2,
                                distance2, isCalled, useDoubleInput_, false);
      distance = mixDistances<float>(distance, distance2);
    }
    if(distance < bestDistance and i != 0) {
      bestDistance = distance;
      bestMatching = matching;
      bestMatching2 = matching2;
      bestAlphas = alphas;
      noUpdate += 1;
    }
    i += 1;
  }
  if(noUpdate == 0)
    printErr("[assignmentOneData] noUpdate ==  0");
  return bestDistance;
}

float ttk::MergeTreeAutoencoder::assignmentOneData(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &origin,
  torch::Tensor &vSTensor,
  TorchUtils::TorchMergeTree<float> &tree2,
  TorchUtils::TorchMergeTree<float> &origin2,
  torch::Tensor &vS2Tensor,
  unsigned int k,
  torch::Tensor &alphasInit,
  torch::Tensor &bestAlphas,
  bool isCalled) {
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> bestMatching,
    bestMatching2;
  return assignmentOneData(tree, origin, vSTensor, tree2, origin2, vS2Tensor, k,
                           alphasInit, bestMatching, bestMatching2, bestAlphas,
                           isCalled);
}

torch::Tensor ttk::MergeTreeAutoencoder::activation(torch::Tensor &in) {
  torch::Tensor act;
  switch(activationFunction_) {
    case 1:
      act = torch::nn::LeakyReLU()(in);
      break;
    case 0:
    default:
      act = torch::nn::ReLU()(in);
  }
  return act;
}

void ttk::MergeTreeAutoencoder::outputBasisReconstruction(
  TorchUtils::TorchMergeTree<float> &originPrime,
  torch::Tensor &vSPrimeTensor,
  TorchUtils::TorchMergeTree<float> &origin2Prime,
  torch::Tensor &vS2PrimeTensor,
  torch::Tensor &alphas,
  TorchUtils::TorchMergeTree<float> &out,
  TorchUtils::TorchMergeTree<float> &out2,
  bool activate) {
  if(not activate_)
    activate = false;
  torch::Tensor act = (activate ? activation(alphas) : alphas);
  getMultiInterpolation(originPrime, vSPrimeTensor, act, out);
  if(useDoubleInput_)
    getMultiInterpolation(origin2Prime, vS2PrimeTensor, act, out2);
}

bool ttk::MergeTreeAutoencoder::forwardOneLayer(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &origin,
  torch::Tensor &vSTensor,
  TorchUtils::TorchMergeTree<float> &originPrime,
  torch::Tensor &vSPrimeTensor,
  TorchUtils::TorchMergeTree<float> &tree2,
  TorchUtils::TorchMergeTree<float> &origin2,
  torch::Tensor &vS2Tensor,
  TorchUtils::TorchMergeTree<float> &origin2Prime,
  torch::Tensor &vS2PrimeTensor,
  unsigned int k,
  torch::Tensor &alphasInit,
  TorchUtils::TorchMergeTree<float> &out,
  TorchUtils::TorchMergeTree<float> &out2,
  torch::Tensor &bestAlphas) {
  bool goodOutput = false;
  int noReset = 0;
  while(not goodOutput) {
    bool isCalled = true;
    assignmentOneData(tree, origin, vSTensor, tree2, origin2, vS2Tensor, k,
                      alphasInit, bestAlphas, isCalled);
    outputBasisReconstruction(originPrime, vSPrimeTensor, origin2Prime,
                              vS2PrimeTensor, bestAlphas, out, out2);
    goodOutput = (out.mTree.tree.getRealNumberOfNodes() != 0
                  and (not useDoubleInput_
                       or out2.mTree.tree.getRealNumberOfNodes() != 0));
    if(not goodOutput) {
      ++noReset;
      if(noReset >= 100) {
        printWrn("[forwardOneLayer] noReset >= 100");
        return true;
      }
      alphasInit = torch::randn_like(alphasInit);
    }
  }
  return false;
}

bool ttk::MergeTreeAutoencoder::forwardOneData(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &tree2,
  unsigned int treeIndex,
  unsigned int k,
  std::vector<torch::Tensor> &alphasInit,
  TorchUtils::TorchMergeTree<float> &out,
  TorchUtils::TorchMergeTree<float> &out2,
  std::vector<torch::Tensor> &dataAlphas,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs2) {
  outs.resize(noLayers_ - 1);
  outs2.resize(noLayers_ - 1);
  dataAlphas.resize(noLayers_);
  for(unsigned int l = 0; l < noLayers_; ++l) {
    auto &treeToUse = (l == 0 ? tree : outs[l - 1]);
    auto &tree2ToUse = (l == 0 ? tree2 : outs2[l - 1]);
    auto &outToUse = (l != noLayers_ - 1 ? outs[l] : out);
    auto &out2ToUse = (l != noLayers_ - 1 ? outs2[l] : out2);
    bool reset = forwardOneLayer(
      treeToUse, origins_[l], vSTensor_[l], originsPrime_[l], vSPrimeTensor_[l],
      tree2ToUse, origins2_[l], vS2Tensor_[l], origins2Prime_[l],
      vS2PrimeTensor_[l], k, alphasInit[l], outToUse, out2ToUse, dataAlphas[l]);
    if(reset)
      return true;
    // Update recs
    auto updateRecs
      = [this, &treeIndex, &l](
          std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &recs,
          TorchUtils::TorchMergeTree<float> &outT) {
          if(recs[treeIndex].size() > noLayers_)
            TorchUtils::copyTorchMergeTree<float>(outT, recs[treeIndex][l + 1]);
          else {
            TorchUtils::TorchMergeTree<float> tmt;
            TorchUtils::copyTorchMergeTree<float>(outT, tmt);
            recs[treeIndex].emplace_back(tmt);
          }
        };
    updateRecs(recs_, outToUse);
    if(useDoubleInput_)
      updateRecs(recs2_, out2ToUse);
  }
  return false;
}

bool ttk::MergeTreeAutoencoder::forwardStep(
  std::vector<TorchUtils::TorchMergeTree<float>> &trees,
  std::vector<TorchUtils::TorchMergeTree<float>> &trees2,
  std::vector<unsigned int> &indexes,
  unsigned int k,
  std::vector<std::vector<torch::Tensor>> &allAlphasInit,
  bool computeReconstructionError,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs2,
  std::vector<std::vector<torch::Tensor>> &bestAlphas,
  std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &layersOuts,
  std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &layersOuts2,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &matchings,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &matchings2,
  float &loss) {
  loss = 0;
  outs.resize(trees.size());
  outs2.resize(trees.size());
  bestAlphas.resize(trees.size());
  layersOuts.resize(trees.size());
  layersOuts2.resize(trees.size());
  matchings.resize(trees.size());
  if(useDoubleInput_)
    matchings2.resize(trees2.size());
  TorchUtils::TorchMergeTree<float> dummyTMT;
  bool reset = false;
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(this->threadNumber_) \
  if(parallelize_) reduction(||: reset) reduction(+:loss)
#endif
  for(unsigned int ind = 0; ind < indexes.size(); ++ind) {
    unsigned int i = indexes[ind];
    auto &tree2ToUse = (trees2.size() == 0 ? dummyTMT : trees2[i]);
    bool dReset
      = forwardOneData(trees[i], tree2ToUse, i, k, allAlphasInit[i], outs[i],
                       outs2[i], bestAlphas[i], layersOuts[i], layersOuts2[i]);
    if(computeReconstructionError) {
      float iLoss = computeOneLoss(
        trees[i], outs[i], trees2[i], outs2[i], matchings[i], matchings2[i]);
      loss += iLoss;
    }
    if(dReset)
      reset = reset || dReset;
  }
  loss /= indexes.size();
  return reset;
}

bool ttk::MergeTreeAutoencoder::forwardStep(
  std::vector<TorchUtils::TorchMergeTree<float>> &trees,
  std::vector<TorchUtils::TorchMergeTree<float>> &trees2,
  std::vector<unsigned int> &indexes,
  unsigned int k,
  std::vector<std::vector<torch::Tensor>> &allAlphasInit,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs2,
  std::vector<std::vector<torch::Tensor>> &bestAlphas) {
  std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> layersOuts,
    layersOuts2;
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    matchings, matchings2;
  bool computeReconstructionError = false;
  float loss;
  return forwardStep(trees, trees2, indexes, k, allAlphasInit,
                     computeReconstructionError, outs, outs2, bestAlphas,
                     layersOuts, layersOuts2, matchings, matchings2, loss);
}

//  ---------------------------------------------------------------------------
//  --- Backward
//  ---------------------------------------------------------------------------
bool ttk::MergeTreeAutoencoder::backwardStep(
  std::vector<TorchUtils::TorchMergeTree<float>> &trees,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &matchings,
  std::vector<TorchUtils::TorchMergeTree<float>> &trees2,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs2,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &matchings2,
  torch::optim::Optimizer &optimizer,
  std::vector<unsigned int> &indexes,
  torch::Tensor &metricLoss,
  torch::Tensor &clusteringLoss,
  torch::Tensor &trackingLoss) {
  double totalLoss = 0;
  bool retainGraph = (metricLossWeight_ != 0 or clusteringLossWeight_ != 0
                      or trackingLossWeight_ != 0);
  std::vector<torch::Tensor> outTensors(indexes.size()),
    reorderedTensors(indexes.size());
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) \
  num_threads(this->threadNumber_) if(parallelize_)
#endif
  for(unsigned int ind = 0; ind < indexes.size(); ++ind) {
    unsigned int i = indexes[ind];
    torch::Tensor reorderedTensor;
    dataReorderingGivenMatching(
      outs[i], trees[i], matchings[i], reorderedTensor);
    auto outTensor = outs[i].tensor;
    if(useDoubleInput_) {
      // TODO The min max pair is counted twice, and the loss is not exactly
      // like mixDistances
      torch::Tensor reorderedTensor2;
      dataReorderingGivenMatching(
        outs2[i], trees2[i], matchings2[i], reorderedTensor2);
      outTensor = torch::cat({outTensor, outs2[i].tensor});
      reorderedTensor = torch::cat({reorderedTensor, reorderedTensor2});
    }
    outTensors[ind] = outTensor;
    reorderedTensors[ind] = reorderedTensor;
  }
  for(unsigned int ind = 0; ind < indexes.size(); ++ind) {
    auto loss
      = torch::nn::functional::mse_loss(outTensors[ind], reorderedTensors[ind]);
    totalLoss += loss.item<float>();
    loss *= reconstructionLossWeight_;
    loss.backward({}, retainGraph);
  }
  // TODO backward on metricLoss is non deterministic
  if(metricLossWeight_ != 0) {
    bool retainGraphMetricLoss
      = (clusteringLossWeight_ != 0 or trackingLossWeight_ != 0);
    metricLoss *= metricLossWeight_
                  * getCustomLossDynamicWeight(
                    totalLoss / indexes.size(), baseRecLoss2_);
    metricLoss.backward({}, retainGraphMetricLoss);
  }
  if(clusteringLossWeight_ != 0) {
    bool retainGraphClusteringLoss = (trackingLossWeight_ != 0);
    clusteringLoss *= clusteringLossWeight_
                      * getCustomLossDynamicWeight(
                        totalLoss / indexes.size(), baseRecLoss2_);
    clusteringLoss.backward({}, retainGraphClusteringLoss);
  }
  if(trackingLossWeight_ != 0) {
    trackingLoss *= trackingLossWeight_;
    trackingLoss.backward();
  }

  for(unsigned int l = 0; l < noLayers_; ++l) {
    if(not origins_[l].tensor.grad().defined()
       or not origins_[l].tensor.grad().count_nonzero().is_nonzero())
      ++originsNoZeroGrad_[l];
    if(not originsPrime_[l].tensor.grad().defined()
       or not originsPrime_[l].tensor.grad().count_nonzero().is_nonzero())
      ++originsPrimeNoZeroGrad_[l];
    if(not vSTensor_[l].grad().defined()
       or not vSTensor_[l].grad().count_nonzero().is_nonzero())
      ++vSNoZeroGrad_[l];
    if(not vSPrimeTensor_[l].grad().defined()
       or not vSPrimeTensor_[l].grad().count_nonzero().is_nonzero())
      ++vSPrimeNoZeroGrad_[l];
    if(useDoubleInput_) {
      if(not origins2_[l].tensor.grad().defined()
         or not origins2_[l].tensor.grad().count_nonzero().is_nonzero())
        ++origins2NoZeroGrad_[l];
      if(not origins2Prime_[l].tensor.grad().defined()
         or not origins2Prime_[l].tensor.grad().count_nonzero().is_nonzero())
        ++origins2PrimeNoZeroGrad_[l];
      if(not vS2Tensor_[l].grad().defined()
         or not vS2Tensor_[l].grad().count_nonzero().is_nonzero())
        ++vS2NoZeroGrad_[l];
      if(not vS2PrimeTensor_[l].grad().defined()
         or not vS2PrimeTensor_[l].grad().count_nonzero().is_nonzero())
        ++vS2PrimeNoZeroGrad_[l];
    }
  }

  optimizer.step();
  optimizer.zero_grad();
  // TODO update MergeTree scalars (not really needed because projectionStep
  // does that and is called just after this)
  return false;
}

//  ---------------------------------------------------------------------------
//  --- Projection
//  ---------------------------------------------------------------------------
void ttk::MergeTreeAutoencoder::projectionStep() {
  auto projectTree = [this](TorchUtils::TorchMergeTree<float> &tmt) {
    interpolationProjection(tmt);
    tmt.tensor = tmt.tensor.detach();
    tmt.tensor.requires_grad_(true);
  };
  for(unsigned int l = 0; l < noLayers_; ++l) {
    projectTree(origins_[l]);
    projectTree(originsPrime_[l]);
    if(useDoubleInput_) {
      projectTree(origins2_[l]);
      projectTree(origins2Prime_[l]);
    }
  }
}

//  ---------------------------------------------------------------------------
//  --- Convergence
//  ---------------------------------------------------------------------------
float ttk::MergeTreeAutoencoder::computeOneLoss(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &out,
  TorchUtils::TorchMergeTree<float> &tree2,
  TorchUtils::TorchMergeTree<float> &out2,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching2) {
  float loss = 0;
  bool isCalled = true;
  float distance;
  computeOneDistance<float>(
    out.mTree, tree.mTree, matching, distance, isCalled, useDoubleInput_);
  if(useDoubleInput_) {
    float distance2;
    computeOneDistance<float>(out2.mTree, tree2.mTree, matching2, distance2,
                              isCalled, useDoubleInput_, false);
    distance = mixDistances<float>(distance, distance2);
  }
  loss += distance * distance;
  return loss;
}

float ttk::MergeTreeAutoencoder::computeLoss(
  std::vector<TorchUtils::TorchMergeTree<float>> &trees,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs,
  std::vector<TorchUtils::TorchMergeTree<float>> &trees2,
  std::vector<TorchUtils::TorchMergeTree<float>> &outs2,
  std::vector<unsigned int> &indexes,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &matchings,
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
    &matchings2) {
  float loss = 0;
  matchings.resize(trees.size());
  if(useDoubleInput_)
    matchings2.resize(trees2.size());
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(this->threadNumber_) \
  if(parallelize_) reduction(+:loss)
#endif
  for(unsigned int ind = 0; ind < indexes.size(); ++ind) {
    unsigned int i = indexes[ind];
    float iLoss = computeOneLoss(
      trees[i], outs[i], trees2[i], outs2[i], matchings[i], matchings2[i]);
    loss += iLoss;
  }
  return loss / indexes.size();
}

bool ttk::MergeTreeAutoencoder::isBestLoss(float loss,
                                           float &minLoss,
                                           unsigned int &cptBlocked) {
  bool isBestEnergy = false;
  if(loss + ENERGY_COMPARISON_TOLERANCE < minLoss) {
    minLoss = loss;
    cptBlocked = 0;
    isBestEnergy = true;
  }
  return isBestEnergy;
}

bool ttk::MergeTreeAutoencoder::convergenceStep(float loss,
                                                float &oldLoss,
                                                float &minLoss,
                                                unsigned int &cptBlocked) {
  double tol = oldLoss / 125.0;
  bool converged = std::abs(loss - oldLoss) < std::abs(tol);
  oldLoss = loss;
  if(not converged) {
    cptBlocked += (minLoss < loss) ? 1 : 0;
    converged = (cptBlocked >= 10 * 10);
    if(converged)
      printMsg("Blocked!", debug::Priority::DETAIL);
  }
  return converged;
}

//  ---------------------------------------------------------------------------
//  --- Main Functions
//  ---------------------------------------------------------------------------
void ttk::MergeTreeAutoencoder::fit(
  std::vector<ftm::MergeTree<float>> &trees,
  std::vector<ftm::MergeTree<float>> &trees2) {
  // torch::set_num_threads(this->threadNumber_);
  torch::set_num_threads(1);
  //  ----- Determinism
  if(deterministic_) {
    int m_seed = 0;
    bool m_torch_deterministic = true;
    srand(m_seed);
    torch::manual_seed(m_seed);
    at::globalContext().setDeterministicCuDNN(m_torch_deterministic ? true
                                                                    : false);
    at::globalContext().setDeterministicAlgorithms(
      m_torch_deterministic ? true : false, true);
  }

  //  ----- Testing
  for(unsigned int i = 0; i < trees.size(); ++i) {
    for(unsigned int n = 0; n < trees[i].tree.getNumberOfNodes(); ++n) {
      if(trees[i].tree.isNodeAlone(n))
        continue;
      auto birthDeath = trees[i].tree.template getBirthDeath<float>(n);
      bigValuesThreshold_
        = std::max(std::abs(std::get<0>(birthDeath)), bigValuesThreshold_);
      bigValuesThreshold_
        = std::max(std::abs(std::get<1>(birthDeath)), bigValuesThreshold_);
    }
  }
  bigValuesThreshold_ *= 100;

  // ----- Convert MergeTree to TorchMergeTree
  std::vector<TorchUtils::TorchMergeTree<float>> torchTrees, torchTrees2;
  mergeTreesToTorchTrees(trees, torchTrees, normalizedWasserstein_);
  mergeTreesToTorchTrees(trees2, torchTrees2, normalizedWasserstein_);

  auto initRecs
    = [](std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &recs,
         std::vector<TorchUtils::TorchMergeTree<float>> &torchTreesT) {
        recs.clear();
        recs.resize(torchTreesT.size());
        for(unsigned int i = 0; i < torchTreesT.size(); ++i) {
          TorchUtils::TorchMergeTree<float> tmt;
          TorchUtils::copyTorchMergeTree<float>(torchTreesT[i], tmt);
          recs[i].emplace_back(tmt);
        }
      };
  initRecs(recs_, torchTrees);
  if(useDoubleInput_)
    initRecs(recs2_, torchTrees2);

  // ----- Init Metric Loss
  if(metricLossWeight_ != 0)
    getDistanceMatrix(torchTrees, torchTrees2, distanceMatrix_);

  // ----- Init Model Parameters
  // printMsg("Init", 0, 0, threadNumber_, debug::LineMode::REPLACE);
  Timer t_init;
  initStep(torchTrees, torchTrees2);
  // initParameters(torchTrees, torchTrees2);
  printMsg("Init", 1, t_init.getElapsedTime(), threadNumber_);

  // --- Init optimizer
  std::vector<torch::Tensor> parameters;
  for(unsigned int l = 0; l < noLayers_; ++l) {
    parameters.emplace_back(origins_[l].tensor);
    parameters.emplace_back(originsPrime_[l].tensor);
    parameters.emplace_back(vSTensor_[l]);
    parameters.emplace_back(vSPrimeTensor_[l]);
    if(trees2.size() != 0) {
      parameters.emplace_back(origins2_[l].tensor);
      parameters.emplace_back(origins2Prime_[l].tensor);
      parameters.emplace_back(vS2Tensor_[l]);
      parameters.emplace_back(vS2PrimeTensor_[l]);
    }
  }
  if(clusteringLossWeight_ != 0)
    for(unsigned int i = 0; i < latentCentroids_.size(); ++i)
      parameters.emplace_back(latentCentroids_[i]);

  torch::optim::Optimizer *optimizer;
  // - Init Adam
  auto adamOptions = torch::optim::AdamOptions(gradientStepSize_);
  adamOptions.betas(std::make_tuple(beta1_, beta2_));
  auto adamOptimizer = torch::optim::Adam(parameters, adamOptions);
  // - Init SGD optimizer
  auto sgdOptions = torch::optim::SGDOptions(gradientStepSize_);
  auto sgdOptimizer = torch::optim::SGD(parameters, sgdOptions);
  // -Init RMSprop optimizer
  auto rmspropOptions = torch::optim::RMSpropOptions(gradientStepSize_);
  auto rmspropOptimizer = torch::optim::RMSprop(parameters, rmspropOptions);
  // - Set optimizer pointer
  switch(optimizer_) {
    case 1:
      optimizer = &sgdOptimizer;
      break;
    case 2:
      optimizer = &rmspropOptimizer;
      break;
    case 0:
    default:
      optimizer = &adamOptimizer;
  }

  // --- Init batches indexes
  unsigned int batchSize = std::min(
    std::max((int)(trees.size() * batchSize_), 1), (int)trees.size());
  std::stringstream ssBatch;
  ssBatch << "batchSize = " << batchSize;
  printMsg(ssBatch.str());
  unsigned int noBatch
    = trees.size() / batchSize + ((trees.size() % batchSize) != 0 ? 1 : 0);
  std::vector<std::vector<unsigned int>> allIndexes(noBatch);
  if(noBatch == 1) {
    allIndexes[0].resize(trees.size());
    std::iota(allIndexes[0].begin(), allIndexes[0].end(), 0);
  }
  auto rng = std::default_random_engine{};

  // ----- Testing
  originsNoZeroGrad_.resize(noLayers_);
  originsPrimeNoZeroGrad_.resize(noLayers_);
  vSNoZeroGrad_.resize(noLayers_);
  vSPrimeNoZeroGrad_.resize(noLayers_);
  for(unsigned int l = 0; l < noLayers_; ++l) {
    originsNoZeroGrad_[l] = 0;
    originsPrimeNoZeroGrad_[l] = 0;
    vSNoZeroGrad_[l] = 0;
    vSPrimeNoZeroGrad_[l] = 0;
  }
  if(useDoubleInput_) {
    origins2NoZeroGrad_.resize(noLayers_);
    origins2PrimeNoZeroGrad_.resize(noLayers_);
    vS2NoZeroGrad_.resize(noLayers_);
    vS2PrimeNoZeroGrad_.resize(noLayers_);
    for(unsigned int l = 0; l < noLayers_; ++l) {
      origins2NoZeroGrad_[l] = 0;
      origins2PrimeNoZeroGrad_[l] = 0;
      vS2NoZeroGrad_[l] = 0;
      vS2PrimeNoZeroGrad_[l] = 0;
    }
  }

  // ----- Init Variables
  baseRecLoss_ = std::numeric_limits<double>::max();
  baseRecLoss2_ = std::numeric_limits<double>::max();
  unsigned int k = k_;
  float oldLoss, minLoss, minRecLoss, minMetricLoss, minClustLoss, minTrackLoss;
  unsigned int cptBlocked, iteration = 0;
  auto initLoop = [&]() {
    oldLoss = -1;
    minLoss = std::numeric_limits<float>::max();
    minRecLoss = minLoss;
    minMetricLoss = minLoss;
    minClustLoss = minLoss;
    minTrackLoss = minLoss;
    cptBlocked = 0;
    iteration = 0;
  };
  initLoop();
  int convWinSize = 5;
  int noConverged = 0, noConvergedToGet = 10;
  std::vector<float> losses, metricLosses, clusteringLosses, trackingLosses;
  float windowLoss = 0;

  double assignmentTime = 0.0, updateTime = 0.0, projectionTime = 0.0,
         lossTime = 0.0;

  int bestIteration = 0;
  std::vector<torch::Tensor> bestVSTensor, bestVSPrimeTensor, bestVS2Tensor,
    bestVS2PrimeTensor;
  std::vector<TorchUtils::TorchMergeTree<float>> bestOrigins, bestOriginsPrime,
    bestOrigins2, bestOrigins2Prime;
  std::vector<std::vector<torch::Tensor>> bestAlphasInit;
  std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> bestRecs,
    bestRecs2;
  double bestTime = 0;

  auto printLoss
    = [this](float loss, float recLoss, float metricLoss, float clustLoss,
             float trackLoss, int iterationT, int iterationTT, double time,
             const debug::Priority &priority = debug::Priority::INFO) {
        std::stringstream prefix;
        prefix << (priority == debug::Priority::VERBOSE ? "Iter " : "Best ");
        std::stringstream ssBestLoss;
        ssBestLoss << prefix.str() << "loss is " << loss << " (iteration "
                   << iterationT << " / " << iterationTT << ") at time "
                   << time;
        printMsg(ssBestLoss.str(), priority);
        if(priority != debug::Priority::VERBOSE)
          prefix.str("");
        if(metricLossWeight_ != 0 or clusteringLossWeight_ != 0
           or trackingLossWeight_ != 0) {
          ssBestLoss.str("");
          ssBestLoss << "- Rec. " << prefix.str() << "loss   = " << recLoss;
          printMsg(ssBestLoss.str(), priority);
        }
        if(metricLossWeight_ != 0) {
          ssBestLoss.str("");
          ssBestLoss << "- Metric " << prefix.str() << "loss = " << metricLoss;
          printMsg(ssBestLoss.str(), priority);
        }
        if(clusteringLossWeight_ != 0) {
          ssBestLoss.str("");
          ssBestLoss << "- Clust. " << prefix.str() << "loss = " << clustLoss;
          printMsg(ssBestLoss.str(), priority);
        }
        if(trackingLossWeight_ != 0) {
          ssBestLoss.str("");
          ssBestLoss << "- Track. " << prefix.str() << "loss = " << trackLoss;
          printMsg(ssBestLoss.str(), priority);
        }
      };

  // ----- Algorithm
  Timer t_alg;
  bool converged = false;
  while(not converged) {
    if(iteration % iterationGap_ == 0) {
      std::stringstream ss;
      ss << "Iteration " << iteration;
      printMsg(debug::Separator::L2);
      printMsg(ss.str());
    }

    bool forwardReset = false;
    std::vector<float> iterationLosses, iterationMetricLosses,
      iterationClusteringLosses, iterationTrackingLosses;
    if(noBatch != 1) {
      std::vector<unsigned int> indexes(trees.size());
      std::iota(indexes.begin(), indexes.end(), 0);
      std::shuffle(std::begin(indexes), std::end(indexes), rng);
      for(unsigned int i = 0; i < allIndexes.size(); ++i) {
        unsigned int noProcessed = batchSize * i;
        unsigned int remaining = trees.size() - noProcessed;
        unsigned int size = std::min(batchSize, remaining);
        allIndexes[i].resize(size);
        for(unsigned int j = 0; j < size; ++j)
          allIndexes[i][j] = indexes[noProcessed + j];
      }
    }
    for(unsigned batchNum = 0; batchNum < allIndexes.size(); ++batchNum) {
      auto &indexes = allIndexes[batchNum];

      // --- Assignment
      Timer t_assignment;
      std::vector<TorchUtils::TorchMergeTree<float>> outs, outs2;
      std::vector<std::vector<torch::Tensor>> bestAlphas;
      std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> layersOuts,
        layersOuts2;
      std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
        matchings, matchings2;
      float loss;
      bool computeReconstructionError = true;
      forwardReset
        = forwardStep(torchTrees, torchTrees2, indexes, k, allAlphas_,
                      computeReconstructionError, outs, outs2, bestAlphas,
                      layersOuts, layersOuts2, matchings, matchings2, loss);
      if(forwardReset)
        break;
      for(unsigned int ind = 0; ind < indexes.size(); ++ind) {
        unsigned int i = indexes[ind];
        for(unsigned int j = 0; j < bestAlphas[i].size(); ++j)
          TorchUtils::copyTensor(bestAlphas[i][j], allAlphas_[i][j]);
      }
      assignmentTime += t_assignment.getElapsedTime();

      // --- Loss
      Timer t_loss;
      /*std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
        matchings, matchings2;
      float loss = computeLoss(
        torchTrees, outs, torchTrees2, outs2, indexes, matchings, matchings2);*/
      losses.emplace_back(loss);
      iterationLosses.emplace_back(loss);
      // - Metric Loss
      torch::Tensor metricLoss;
      if(metricLossWeight_ != 0) {
        computeMetricLoss(layersOuts, layersOuts2, bestAlphas, distanceMatrix_,
                          indexes, metricLoss);
        float metricLossF = metricLoss.item<float>();
        metricLosses.emplace_back(metricLossF);
        iterationMetricLosses.emplace_back(metricLossF);
      }
      // - Clustering Loss
      torch::Tensor clusteringLoss;
      if(clusteringLossWeight_ != 0) {
        torch::Tensor asgn;
        computeClusteringLoss(bestAlphas, indexes, clusteringLoss, asgn);
        if(iteration % 100 == 0) {
          std::cout << asgn << std::endl;
        }
        float clusteringLossF = clusteringLoss.item<float>();
        clusteringLosses.emplace_back(clusteringLossF);
        iterationClusteringLosses.emplace_back(clusteringLossF);
      }
      // - Tracking Loss
      torch::Tensor trackingLoss;
      if(trackingLossWeight_ != 0) {
        computeTrackingLoss(trackingLoss);
        float trackingLossF = trackingLoss.item<float>();
        trackingLosses.emplace_back(trackingLossF);
        iterationTrackingLosses.emplace_back(trackingLossF);
      }
      lossTime += t_loss.getElapsedTime();

      // --- Update
      Timer t_update;
      backwardStep(torchTrees, outs, matchings, torchTrees2, outs2, matchings2,
                   *optimizer, indexes, metricLoss, clusteringLoss,
                   trackingLoss);
      updateTime += t_update.getElapsedTime();

      // --- Projection
      Timer t_projection;
      projectionStep();
      projectionTime += t_projection.getElapsedTime();
    }

    if(forwardReset) {
      // TODO is there a batter way to manage this? init new parameters and
      // start again?
      printWrn("Forward reset!");
      break;
    }

    // --- Get iteration loss
    // TODO an approximation is made here if batch size != 1 because the
    // iteration loss will not be exact, we need to do a forward step and
    // compute loss with the whole dataset
    /*if(batchSize_ != 1)
      printWrn("iteration loss approximation (batchSize_ != 1)");*/
    float iterationRecLoss
      = torch::tensor(iterationLosses).mean().item<float>();
    float iterationLoss = reconstructionLossWeight_ * iterationRecLoss;
    float iterationMetricLoss = 0;
    if(metricLossWeight_ != 0) {
      iterationMetricLoss
        = torch::tensor(iterationMetricLosses).mean().item<float>();
      iterationLoss
        += metricLossWeight_
           * getCustomLossDynamicWeight(iterationRecLoss, baseRecLoss_)
           * iterationMetricLoss;
    }
    float iterationClusteringLoss = 0;
    if(clusteringLossWeight_ != 0) {
      iterationClusteringLoss
        = torch::tensor(iterationClusteringLosses).mean().item<float>();
      iterationLoss
        += clusteringLossWeight_
           * getCustomLossDynamicWeight(iterationRecLoss, baseRecLoss_)
           * iterationClusteringLoss;
    }
    float iterationTrackingLoss = 0;
    if(trackingLossWeight_ != 0) {
      iterationTrackingLoss
        = torch::tensor(iterationTrackingLosses).mean().item<float>();
      iterationLoss += trackingLossWeight_ * iterationTrackingLoss;
    }
    printLoss(iterationLoss, iterationRecLoss, iterationMetricLoss,
              iterationClusteringLoss, iterationTrackingLoss, iteration,
              iteration, t_alg.getElapsedTime() - t_allVectorCopy_time_,
              debug::Priority::VERBOSE);

    // --- Update best parameters
    bool isBest = isBestLoss(iterationLoss, minLoss, cptBlocked);
    if(isBest) {
      Timer t_copy;
      bestIteration = iteration;
      copyParams(origins_, originsPrime_, vSTensor_, vSPrimeTensor_, origins2_,
                 origins2Prime_, vS2Tensor_, vS2PrimeTensor_, allAlphas_,
                 bestOrigins, bestOriginsPrime, bestVSTensor, bestVSPrimeTensor,
                 bestOrigins2, bestOrigins2Prime, bestVS2Tensor,
                 bestVS2PrimeTensor, bestAlphasInit);
      copyParams(recs_, bestRecs);
      copyParams(recs2_, bestRecs2);
      t_allVectorCopy_time_ += t_copy.getElapsedTime();
      bestTime = t_alg.getElapsedTime() - t_allVectorCopy_time_;
      minRecLoss = iterationRecLoss;
      minMetricLoss = iterationMetricLoss;
      minClustLoss = iterationClusteringLoss;
      minTrackLoss = iterationTrackingLoss;
      printLoss(minLoss, minRecLoss, minMetricLoss, minClustLoss, minTrackLoss,
                bestIteration, iteration, bestTime, debug::Priority::DETAIL);
    }

    // --- Convergence
    windowLoss += iterationLoss;
    if((iteration + 1) % convWinSize == 0) {
      windowLoss /= convWinSize;
      converged = convergenceStep(windowLoss, oldLoss, minLoss, cptBlocked);
      windowLoss = 0;
      if(converged) {
        ++noConverged;
      } else
        noConverged = 0;
      converged = noConverged >= noConvergedToGet;
      if(converged and iteration < minIteration_)
        printMsg("convergence is detected but iteration < minIteration_",
                 debug::Priority::DETAIL);
      if(iteration < minIteration_)
        converged = false;
      if(converged)
        break;
    }

    // --- Print
    if(iteration % iterationGap_ == 0) {
      printMsg("Assignment", 1, assignmentTime, threadNumber_);
      printMsg("Loss", 1, lossTime, threadNumber_);
      printMsg("Update", 1, updateTime, threadNumber_);
      printMsg("Projection", 1, projectionTime, threadNumber_);
      assignmentTime = 0.0;
      lossTime = 0.0;
      updateTime = 0.0;
      projectionTime = 0.0;
      std::stringstream ss;
      float loss = torch::tensor(losses).mean().item<float>();
      losses.clear();
      ss << "Rec. loss   = " << loss;
      printMsg(ss.str());
      if(metricLossWeight_ != 0) {
        float metricLoss = torch::tensor(metricLosses).mean().item<float>();
        metricLosses.clear();
        ss.str("");
        ss << "Metric loss = " << metricLoss;
        printMsg(ss.str());
      }
      if(clusteringLossWeight_ != 0) {
        float clusteringLoss
          = torch::tensor(clusteringLosses).mean().item<float>();
        clusteringLosses.clear();
        ss.str("");
        ss << "Clust. loss = " << clusteringLoss;
        printMsg(ss.str());
      }
      if(trackingLossWeight_ != 0) {
        float trackingLoss = torch::tensor(trackingLosses).mean().item<float>();
        trackingLosses.clear();
        ss.str("");
        ss << "Track. loss = " << trackingLoss;
        printMsg(ss.str());
      }

      // Verify grad and big values (testing)
      for(unsigned int l = 0; l < noLayers_; ++l) {
        if(originsNoZeroGrad_[l] != 0)
          std::cout << originsNoZeroGrad_[l] << " originsNoZeroGrad_[" << l
                    << "]" << std::endl;
        if(originsPrimeNoZeroGrad_[l] != 0)
          std::cout << originsPrimeNoZeroGrad_[l] << " originsPrimeNoZeroGrad_["
                    << l << "]" << std::endl;
        if(vSNoZeroGrad_[l] != 0)
          std::cout << vSNoZeroGrad_[l] << " vSNoZeroGrad_[" << l << "]"
                    << std::endl;
        if(vSPrimeNoZeroGrad_[l] != 0)
          std::cout << vSPrimeNoZeroGrad_[l] << " vSPrimeNoZeroGrad_[" << l
                    << "]" << std::endl;
        /*if(originsNoZeroGrad_[l] != 0)
          for(unsigned int i = 0; i < trees.size(); ++i)
            std::cout << allAlphas_[i][l] << std::endl;*/
        originsNoZeroGrad_[l] = 0;
        originsPrimeNoZeroGrad_[l] = 0;
        vSNoZeroGrad_[l] = 0;
        vSPrimeNoZeroGrad_[l] = 0;
        if(useDoubleInput_) {
          if(origins2NoZeroGrad_[l] != 0)
            std::cout << origins2NoZeroGrad_[l] << " origins2NoZeroGrad_[" << l
                      << "]" << std::endl;
          if(origins2PrimeNoZeroGrad_[l] != 0)
            std::cout << origins2PrimeNoZeroGrad_[l]
                      << " origins2PrimeNoZeroGrad_[" << l << "]" << std::endl;
          if(vS2NoZeroGrad_[l] != 0)
            std::cout << vS2NoZeroGrad_[l] << " vS2NoZeroGrad_[" << l << "]"
                      << std::endl;
          if(vS2PrimeNoZeroGrad_[l] != 0)
            std::cout << vS2PrimeNoZeroGrad_[l] << " vS2PrimeNoZeroGrad_[" << l
                      << "]" << std::endl;
          origins2NoZeroGrad_[l] = 0;
          origins2PrimeNoZeroGrad_[l] = 0;
          vS2NoZeroGrad_[l] = 0;
          vS2PrimeNoZeroGrad_[l] = 0;
        }
        if(isTreeHasBigValues(origins_[l].mTree, bigValuesThreshold_))
          std::cout << "origins_[" << l << "] has big values!" << std::endl;
        if(isTreeHasBigValues(originsPrime_[l].mTree, bigValuesThreshold_))
          std::cout << "originsPrime_[" << l << "] has big values!"
                    << std::endl;
      }
    }

    ++iteration;
    if(maxIteration_ != 0 and iteration >= maxIteration_) {
      printMsg("iteration >= maxIteration_", debug::Priority::DETAIL);
      break;
    }
  }
  printMsg(debug::Separator::L2);
  printLoss(minLoss, minRecLoss, minMetricLoss, minClustLoss, minTrackLoss,
            bestIteration, iteration, bestTime);
  printMsg(debug::Separator::L2);

  Timer t_copy;
  copyParams(bestOrigins, bestOriginsPrime, bestVSTensor, bestVSPrimeTensor,
             bestOrigins2, bestOrigins2Prime, bestVS2Tensor, bestVS2PrimeTensor,
             bestAlphasInit, origins_, originsPrime_, vSTensor_, vSPrimeTensor_,
             origins2_, origins2Prime_, vS2Tensor_, vS2PrimeTensor_,
             allAlphas_);
  copyParams(bestRecs, recs_);
  copyParams(bestRecs2, recs2_);
  t_allVectorCopy_time_ += t_copy.getElapsedTime();
  printMsg("Copy time", 1, t_allVectorCopy_time_, threadNumber_);
}

void ttk::MergeTreeAutoencoder::execute(
  std::vector<ftm::MergeTree<float>> &trees,
  std::vector<ftm::MergeTree<float>> &trees2) {
  // makeExponentialExample(trees, trees2);

  // --- Preprocessing
  Timer t_preprocess;
  preprocessingTrees<float>(trees, treesNodeCorr_);
  if(trees2.size() != 0)
    preprocessingTrees<float>(trees2, trees2NodeCorr_);
  printMsg("Preprocessing", 1, t_preprocess.getElapsedTime(), threadNumber_);
  useDoubleInput_ = (trees2.size() != 0);

  // --- Fit autoencoder
  Timer t_total;
  fit(trees, trees2);
  auto totalTime = t_total.getElapsedTime() - t_allVectorCopy_time_;
  printMsg(debug::Separator::L1);
  printMsg("Total time", 1, totalTime, threadNumber_);
  hasComputedOnce_ = true;

  // --- End functions
  createScaledAlphas();
  createActivatedAlphas();
  computeTrackingInformation();
  // Correlation
  auto latLayer = getLatentLayerIndex();
  std::vector<std::vector<double>> allTs;
  auto noGeod = allAlphas_[0][latLayer].sizes()[0];
  allTs.resize(noGeod);
  for(unsigned int i = 0; i < noGeod; ++i) {
    allTs[i].resize(allAlphas_.size());
    for(unsigned int j = 0; j < allAlphas_.size(); ++j)
      allTs[i][j] = allAlphas_[j][latLayer][i].item<double>();
  }
  computeBranchesCorrelationMatrix(origins_[0].mTree, trees, dataMatchings_[0],
                                   allTs, branchesCorrelationMatrix_,
                                   persCorrelationMatrix_);
  // Custom recs
  originsCopy_.resize(origins_.size());
  originsPrimeCopy_.resize(originsPrime_.size());
  for(unsigned int l = 0; l < origins_.size(); ++l) {
    TorchUtils::copyTorchMergeTree<float>(origins_[l], originsCopy_[l]);
    TorchUtils::copyTorchMergeTree<float>(
      originsPrime_[l], originsPrimeCopy_[l]);
  }
  createCustomRecs();

  // --- Postprocessing
  if(createOutput_) {
    for(unsigned int i = 0; i < trees.size(); ++i)
      postprocessingPipeline<float>(&(trees[i].tree));
    for(unsigned int i = 0; i < trees2.size(); ++i)
      postprocessingPipeline<float>(&(trees2[i].tree));
    for(unsigned int l = 0; l < origins_.size(); ++l) {
      postprocessingPipeline<float>(&(origins_[l].mTree.tree));
      postprocessingPipeline<float>(&(originsPrime_[l].mTree.tree));
    }
    for(unsigned int j = 0; j < recs_[0].size(); ++j) {
      /*std::cout << "======================================== " << j
                << std::endl;*/
      for(unsigned int i = 0; i < recs_.size(); ++i) {
        // std::cout << "==================== " << i << std::endl;
        // ttk::printPairs(recs_[i][j].mTree);
        postprocessingPipeline<float>(&(recs_[i][j].mTree.tree));
        fixTreePrecisionScalars(recs_[i][j].mTree);
        /*std::cout << recs_[i][j]
                       .mTree.tree.template printTreeScalars<float>(false)
                       .str()
                  << std::endl;
        std::cout << recs_[i][j].mTree.tree.printTree().str() << std::endl;*/
      }
    }
  }

  if(not isPersistenceDiagram_) {
    for(unsigned int l = 0; l < originsMatchings_.size(); ++l) {
      auto &tree1 = (l == 0 ? origins_[0] : originsPrime_[l - 1]);
      auto &tree2 = (l == 0 ? originsPrime_[0] : originsPrime_[l]);
      convertBranchDecompositionMatching<float>(
        &(tree1.mTree.tree), &(tree2.mTree.tree), originsMatchings_[l]);
    }
    for(unsigned int l = 0; l < dataMatchings_.size(); ++l) {
      for(unsigned int i = 0; i < recs_.size(); ++i) {
        auto &origin = (l == 0 ? origins_[0] : originsPrime_[l - 1]);
        convertBranchDecompositionMatching<float>(&(origin.mTree.tree),
                                                  &(recs_[i][l].mTree.tree),
                                                  dataMatchings_[l][i]);
      }
    }
    for(unsigned int i = 0; i < reconstMatchings_.size(); ++i) {
      auto l = recs_[i].size() - 1;
      convertBranchDecompositionMatching<float>(&(recs_[i][0].mTree.tree),
                                                &(recs_[i][l].mTree.tree),
                                                reconstMatchings_[i]);
    }
  }
}

//  ---------------------------------------------------------------------------
//  --- Custom Losses
//  ---------------------------------------------------------------------------
double ttk::MergeTreeAutoencoder::getCustomLossDynamicWeight(double recLoss,
                                                             double &baseLoss) {
  /*int noDigit = 0;
  float temp = recLoss * 1e6;
  while(temp >= 10) {
    temp /= 10.0;
    ++noDigit;
  }
  noDigit -= 6;
  return std::pow(10, noDigit);*/
  baseLoss = std::min(recLoss, baseLoss);
  // return baseLoss;
  // return recLoss;
  // return 1.0;
  if(customLossDynamicWeight_)
    return baseLoss;
  else
    return 1.0;
}

void ttk::MergeTreeAutoencoder::getDistanceMatrix(
  std::vector<TorchUtils::TorchMergeTree<float>> &tmts,
  std::vector<std::vector<float>> &distanceMatrix,
  bool useDoubleInput,
  bool isFirstInput) {
  distanceMatrix.clear();
  distanceMatrix.resize(tmts.size(), std::vector<float>(tmts.size(), 0));
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel num_threads(this->threadNumber_) if(parallelize_) \
  shared(distanceMatrix, tmts)
  {
#pragma omp single nowait
    {
#endif
      for(unsigned int i = 0; i < tmts.size(); ++i) {
        for(unsigned int j = i + 1; j < tmts.size(); ++j) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp task UNTIED() shared(distanceMatrix, tmts) firstprivate(i, j)
          {
#endif
            std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> matching;
            float distance;
            bool isCalled = true;
            computeOneDistance(tmts[i].mTree, tmts[j].mTree, matching, distance,
                               isCalled, useDoubleInput, isFirstInput);
            distance = distance * distance;
            distanceMatrix[i][j] = distance;
            distanceMatrix[j][i] = distance;
#ifdef TTK_ENABLE_OPENMP
          } // pragma omp task
#endif
        }
      }
#ifdef TTK_ENABLE_OPENMP
#pragma omp taskwait
    } // pragma omp single nowait
  } // pragma omp parallel
#endif
}

void ttk::MergeTreeAutoencoder::getDistanceMatrix(
  std::vector<TorchUtils::TorchMergeTree<float>> &tmts,
  std::vector<TorchUtils::TorchMergeTree<float>> &tmts2,
  std::vector<std::vector<float>> &distanceMatrix) {
  getDistanceMatrix(tmts, distanceMatrix, useDoubleInput_);
  if(useDoubleInput_) {
    std::vector<std::vector<float>> distanceMatrix2;
    getDistanceMatrix(tmts2, distanceMatrix2, useDoubleInput_, false);
    mixDistancesMatrix<float>(distanceMatrix, distanceMatrix2);
  }
}

void ttk::MergeTreeAutoencoder::getDifferentiableDistanceFromMatchings(
  TorchUtils::TorchMergeTree<float> &tree1,
  TorchUtils::TorchMergeTree<float> &tree2,
  TorchUtils::TorchMergeTree<float> &tree1_2,
  TorchUtils::TorchMergeTree<float> &tree2_2,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matchings,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matchings2,
  torch::Tensor &tensorDist,
  bool doSqrt) {
  // TODO ensure that there is no additionnal pairs in tensors
  torch::Tensor reorderedITensor, reorderedJTensor;
  dataReorderingGivenMatching(
    tree1, tree2, matchings, reorderedITensor, reorderedJTensor);
  if(useDoubleInput_) {
    // TODO The min max pair is counted twice,  and the loss is not
    // exactly like mixDistances
    torch::Tensor reorderedI2Tensor, reorderedJ2Tensor;
    dataReorderingGivenMatching(
      tree1_2, tree2_2, matchings2, reorderedI2Tensor, reorderedJ2Tensor);
    reorderedITensor = torch::cat({reorderedITensor, reorderedI2Tensor});
    reorderedJTensor = torch::cat({reorderedJTensor, reorderedJ2Tensor});
  }
  tensorDist = (reorderedITensor - reorderedJTensor).pow(2).sum();
  if(doSqrt)
    tensorDist = tensorDist.sqrt();
}

void ttk::MergeTreeAutoencoder::getDifferentiableDistance(
  TorchUtils::TorchMergeTree<float> &tree1,
  TorchUtils::TorchMergeTree<float> &tree2,
  TorchUtils::TorchMergeTree<float> &tree1_2,
  TorchUtils::TorchMergeTree<float> &tree2_2,
  torch::Tensor &tensorDist,
  bool isCalled,
  bool doSqrt) {
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> matchings,
    matchings2;
  float distance;
  computeOneDistance<float>(
    tree1.mTree, tree2.mTree, matchings, distance, isCalled, useDoubleInput_);
  if(useDoubleInput_) {
    float distance2;
    computeOneDistance<float>(tree1_2.mTree, tree2_2.mTree, matchings2,
                              distance2, isCalled, useDoubleInput_, false);
  }
  getDifferentiableDistanceFromMatchings(
    tree1, tree2, tree1_2, tree2_2, matchings, matchings2, tensorDist, doSqrt);
}

void ttk::MergeTreeAutoencoder::getDifferentiableDistance(
  TorchUtils::TorchMergeTree<float> &tree1,
  TorchUtils::TorchMergeTree<float> &tree2,
  torch::Tensor &tensorDist,
  bool isCalled,
  bool doSqrt) {
  TorchUtils::TorchMergeTree<float> tree1_2, tree2_2;
  getDifferentiableDistance(
    tree1, tree2, tree1_2, tree2_2, tensorDist, isCalled, doSqrt);
}

void ttk::MergeTreeAutoencoder::getDifferentiableDistanceMatrix(
  std::vector<TorchUtils::TorchMergeTree<float> *> &trees,
  std::vector<TorchUtils::TorchMergeTree<float> *> &trees2,
  std::vector<std::vector<torch::Tensor>> &outDistMat) {
  outDistMat.resize(trees.size(), std::vector<torch::Tensor>(trees.size()));
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel num_threads(this->threadNumber_) if(parallelize_) \
  shared(trees, trees2, outDistMat)
  {
#pragma omp single nowait
    {
#endif
      for(unsigned int i = 0; i < trees.size(); ++i) {
        outDistMat[i][i] = torch::tensor(0);
        for(unsigned int j = i + 1; j < trees.size(); ++j) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp task UNTIED() shared(trees, trees2, outDistMat) firstprivate(i, j)
          {
#endif
            bool isCalled = true;
            bool doSqrt = false;
            torch::Tensor tensorDist;
            getDifferentiableDistance(*(trees[i]), *(trees[j]), *(trees2[i]),
                                      *(trees2[j]), tensorDist, isCalled,
                                      doSqrt);
            outDistMat[i][j] = tensorDist;
            outDistMat[j][i] = tensorDist;
#ifdef TTK_ENABLE_OPENMP
          } // pragma omp task
#endif
        }
      }
#ifdef TTK_ENABLE_OPENMP
#pragma omp taskwait
    } // pragma omp single nowait
  } // pragma omp parallel
#endif
}

void ttk::MergeTreeAutoencoder::getAlphasTensor(
  std::vector<std::vector<torch::Tensor>> &alphas,
  std::vector<unsigned int> &indexes,
  unsigned int layerIndex,
  torch::Tensor &alphasOut) {
  alphasOut = alphas[indexes[0]][layerIndex].transpose(0, 1);
  for(unsigned int ind = 1; ind < indexes.size(); ++ind)
    alphasOut = torch::cat(
      {alphasOut, alphas[indexes[ind]][layerIndex].transpose(0, 1)});
}

void ttk::MergeTreeAutoencoder::computeMetricLoss(
  std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &layersOuts,
  std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &layersOuts2,
  std::vector<std::vector<torch::Tensor>> alphas,
  std::vector<std::vector<float>> &baseDistanceMatrix,
  std::vector<unsigned int> &indexes,
  torch::Tensor &metricLoss) {
  auto layerIndex = getLatentLayerIndex();
  std::vector<std::vector<torch::Tensor>> losses(
    layersOuts.size(), std::vector<torch::Tensor>(layersOuts.size()));

  std::vector<TorchUtils::TorchMergeTree<float> *> trees, trees2;
  for(unsigned int ind = 0; ind < indexes.size(); ++ind) {
    unsigned int i = indexes[ind];
    trees.emplace_back(&(layersOuts[i][layerIndex]));
    if(useDoubleInput_)
      trees2.emplace_back(&(layersOuts2[i][layerIndex]));
  }

  std::vector<std::vector<torch::Tensor>> outDistMat;
  torch::Tensor coefDistMat;
  if(customLossSpace_) {
    getDifferentiableDistanceMatrix(trees, trees2, outDistMat);
  } else {
    std::vector<std::vector<torch::Tensor>> scaledAlphas;
    createScaledAlphas(alphas, vSTensor_, scaledAlphas);
    torch::Tensor latentAlphas;
    getAlphasTensor(scaledAlphas, indexes, layerIndex, latentAlphas);
    if(customLossActivate_)
      latentAlphas = activation(latentAlphas);
    coefDistMat = torch::cdist(latentAlphas, latentAlphas).pow(2);
  }

  torch::Tensor maxLoss = torch::tensor(0);
  metricLoss = torch::tensor(0);
  float div = 0;
  for(unsigned int ind = 0; ind < indexes.size(); ++ind) {
    unsigned int i = indexes[ind];
    for(unsigned int ind2 = ind + 1; ind2 < indexes.size(); ++ind2) {
      unsigned int j = indexes[ind2];
      torch::Tensor loss;
      torch::Tensor toCompare
        = (customLossSpace_ ? outDistMat[i][j] : coefDistMat[ind][ind2]);
      loss = torch::nn::MSELoss()(
        torch::tensor(baseDistanceMatrix[i][j]), toCompare);
      metricLoss = metricLoss + loss;
      maxLoss = torch::max(loss, maxLoss);
      ++div;
    }
  }
  metricLoss = metricLoss / torch::tensor(div) / maxLoss;
}

void ttk::MergeTreeAutoencoder::computeClusteringLoss(
  std::vector<std::vector<torch::Tensor>> &alphas,
  std::vector<unsigned int> &indexes,
  torch::Tensor &clusteringLoss,
  torch::Tensor &asgn) {
  // Compute distance matrix
  unsigned int layerIndex = getLatentLayerIndex();
  torch::Tensor latentAlphas;
  getAlphasTensor(alphas, indexes, layerIndex, latentAlphas);
  if(customLossActivate_)
    latentAlphas = activation(latentAlphas);
  torch::Tensor centroids = latentCentroids_[0].transpose(0, 1);
  for(unsigned int i = 1; i < latentCentroids_.size(); ++i)
    centroids = torch::cat({centroids, latentCentroids_[i].transpose(0, 1)});
  torch::Tensor dist = torch::cdist(latentAlphas, centroids);

  // Compute softmax and one hot real asgn
  dist = dist * -clusteringLossTemp_;
  asgn = torch::nn::Softmax(1)(dist);
  std::vector<float> clusterAsgn;
  for(unsigned int ind = 0; ind < indexes.size(); ++ind) {
    clusterAsgn.emplace_back(clusterAsgn_[indexes[ind]]);
  }
  torch::Tensor realAsgn = torch::tensor(clusterAsgn).to(torch::kInt64);
  realAsgn
    = torch::nn::functional::one_hot(realAsgn, asgn.sizes()[1]).to(torch::kF32);

  // Compute KL div.
  clusteringLoss = torch::nn::KLDivLoss(
    torch::nn::KLDivLossOptions().reduction(torch::kBatchMean))(asgn, realAsgn);
  // clusteringLoss = torch::nn::MSELoss()(asgn, realAsgn);
}

void ttk::MergeTreeAutoencoder::computeTrackingLoss(
  torch::Tensor &trackingLoss) {
  unsigned int latentLayerIndex = getLatentLayerIndex() + 1;
  auto endLayer = (trackingLossDecoding_ ? noLayers_ : latentLayerIndex);
  std::vector<torch::Tensor> losses(endLayer);
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) \
  num_threads(this->threadNumber_) if(parallelize_)
#endif
  for(unsigned int l = 0; l < endLayer; ++l) {
    auto &tree1 = (l == 0 ? origins_[0] : originsPrime_[l - 1]);
    auto &tree2 = (l == 0 ? originsPrime_[0] : originsPrime_[l]);
    torch::Tensor tensorDist;
    bool isCalled = true, doSqrt = false;
    getDifferentiableDistance(tree1, tree2, tensorDist, isCalled, doSqrt);
    losses[l] = tensorDist;
  }
  trackingLoss = torch::tensor(0, torch::kFloat32);
  for(unsigned int i = 0; i < losses.size(); ++i)
    trackingLoss += losses[i];
}

//  ---------------------------------------------------------------------------
//  --- End Functions
//  ---------------------------------------------------------------------------
void ttk::MergeTreeAutoencoder::createCustomRecs() {
  if(customAlphas_.empty())
    return;

  std::vector<torch::Tensor> allTreesAlphas(noLayers_);
  for(unsigned int l = 0; l < noLayers_; ++l) {
    allTreesAlphas[l] = allAlphas_[0][l].reshape({-1, 1});
    for(unsigned int i = 1; i < allAlphas_.size(); ++i)
      allTreesAlphas[l] = torch::cat({allTreesAlphas[l], allAlphas_[i][l]}, 1);
    allTreesAlphas[l] = allTreesAlphas[l].transpose(0, 1);
  }

  unsigned int latLayer = getLatentLayerIndex();
  customRecs_.resize(customAlphas_.size());
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) \
  num_threads(this->threadNumber_) if(parallelize_)
#endif
  for(unsigned int i = 0; i < customAlphas_.size(); ++i) {
    torch::Tensor alphas = torch::tensor(customAlphas_[i]).reshape({-1, 1});

    auto driver = "gelsd";
    torch::Tensor alphasWeight
      = std::get<0>(
          torch::linalg::lstsq(allTreesAlphas[latLayer].transpose(0, 1), alphas,
                               c10::nullopt, driver))
          .transpose(0, 1);

    // Reconst latent
    std::vector<TorchUtils::TorchMergeTree<float>> outs, outs2;
    auto noOuts = noLayers_ - latLayer;
    outs.resize(noOuts);
    outs2.resize(noOuts);
    TorchUtils::TorchMergeTree<float> out, out2;
    outputBasisReconstruction(
      originsPrimeCopy_[latLayer], vSPrimeTensor_[latLayer],
      origins2Prime_[latLayer], vS2PrimeTensor_[latLayer], alphas, outs[0],
      outs2[0]);
    // Decoding
    unsigned int k = 16;
    for(unsigned int l = latLayer + 1; l < noLayers_; ++l) {
      torch::Tensor alphasInit
        = torch::matmul(alphasWeight, allTreesAlphas[l]).transpose(0, 1),
        dataAlphas;
      auto outIndex = l - latLayer;
      auto &outToUse = (l != noLayers_ - 1 ? outs[outIndex] : customRecs_[i]);
      forwardOneLayer(outs[outIndex - 1], originsCopy_[l], vSTensor_[l],
                      originsPrimeCopy_[l], vSPrimeTensor_[l],
                      outs2[outIndex - 1], origins2_[l], vS2Tensor_[l],
                      origins2Prime_[l], vS2PrimeTensor_[l], k, alphasInit,
                      outToUse, outs2[outIndex], dataAlphas);
    }
  }

  customMatchings_.resize(customRecs_.size());
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) \
  num_threads(this->threadNumber_) if(parallelize_)
#endif
  for(unsigned int i = 0; i < customRecs_.size(); ++i) {
    bool isCalled = true;
    float distance;
    computeOneDistance<float>(originsCopy_[0].mTree, customRecs_[i].mTree,
                              customMatchings_[i], distance, isCalled,
                              useDoubleInput_);
  }

  for(unsigned int i = 0; i < customRecs_.size(); ++i) {
    postprocessingPipeline<float>(&(customRecs_[i].mTree.tree));
    if(not isPersistenceDiagram_) {
      TorchUtils::TorchMergeTree<float> originCopy;
      TorchUtils::copyTorchMergeTree<float>(originsCopy_[0], originCopy);
      postprocessingPipeline<float>(&(originCopy.mTree.tree));
      convertBranchDecompositionMatching<float>(&(originCopy.mTree.tree),
                                                &(customRecs_[i].mTree.tree),
                                                customMatchings_[i]);
    }
  }
}

void ttk::MergeTreeAutoencoder::computeTrackingInformation() {
  unsigned int latentLayerIndex = getLatentLayerIndex() + 1;
  originsMatchings_.resize(latentLayerIndex);
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) \
  num_threads(this->threadNumber_) if(parallelize_)
#endif
  for(unsigned int l = 0; l < latentLayerIndex; ++l) {
    auto &tree1 = (l == 0 ? origins_[0] : originsPrime_[l - 1]);
    auto &tree2 = (l == 0 ? originsPrime_[0] : originsPrime_[l]);
    bool isCalled = true;
    float distance;
    computeOneDistance<float>(tree1.mTree, tree2.mTree, originsMatchings_[l],
                              distance, isCalled, useDoubleInput_);
  }

  // Data matchings
  ++latentLayerIndex;
  dataMatchings_.resize(latentLayerIndex);
  for(unsigned int l = 0; l < latentLayerIndex; ++l) {
    dataMatchings_[l].resize(recs_.size());
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) \
  num_threads(this->threadNumber_) if(parallelize_)
#endif
    for(unsigned int i = 0; i < recs_.size(); ++i) {
      bool isCalled = true;
      float distance;
      auto &origin = (l == 0 ? origins_[0] : originsPrime_[l - 1]);
      computeOneDistance<float>(origin.mTree, recs_[i][l].mTree,
                                dataMatchings_[l][i], distance, isCalled,
                                useDoubleInput_);
    }
  }

  // Reconst matchings
  reconstMatchings_.resize(recs_.size());
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) \
  num_threads(this->threadNumber_) if(parallelize_)
#endif
  for(unsigned int i = 0; i < recs_.size(); ++i) {
    bool isCalled = true;
    float distance;
    auto l = recs_[i].size() - 1;
    computeOneDistance<float>(recs_[i][0].mTree, recs_[i][l].mTree,
                              reconstMatchings_[i], distance, isCalled,
                              useDoubleInput_);
  }
}

void ttk::MergeTreeAutoencoder::createScaledAlphas(
  std::vector<std::vector<torch::Tensor>> &alphas,
  std::vector<torch::Tensor> &vSTensor,
  std::vector<std::vector<torch::Tensor>> &scaledAlphas) {
  scaledAlphas.clear();
  scaledAlphas.resize(
    alphas.size(), std::vector<torch::Tensor>(alphas[0].size()));
  for(unsigned int l = 0; l < alphas[0].size(); ++l) {
    torch::Tensor scale = vSTensor[l].pow(2).sum(0).sqrt();
    for(unsigned int i = 0; i < alphas.size(); ++i) {
      scaledAlphas[i][l] = alphas[i][l] * scale.reshape({-1, 1});
    }
  }
}

void ttk::MergeTreeAutoencoder::createScaledAlphas() {
  createScaledAlphas(allAlphas_, vSTensor_, allScaledAlphas_);
}

void ttk::MergeTreeAutoencoder::createActivatedAlphas() {
  allActAlphas_ = allAlphas_;
  for(unsigned int i = 0; i < allActAlphas_.size(); ++i)
    for(unsigned int j = 0; j < allActAlphas_[i].size(); ++j)
      allActAlphas_[i][j] = activation(allActAlphas_[i][j]);
  createScaledAlphas(allActAlphas_, vSTensor_, allActScaledAlphas_);
}

void ttk::MergeTreeAutoencoder::fixTreePrecisionScalars(
  ftm::MergeTree<float> &mTree) {
  bool isJT = mTree.tree.isJoinTree<float>();
  auto getSign = [](float v) { return (v > 0 ? 1.0 : -1.0); };
  auto shiftSubtree
    = [&mTree, &isJT, &getSign](ftm::idNode node, std::vector<float> &scalars) {
        std::queue<ftm::idNode> queue;
        queue.emplace(node);
        while(!queue.empty()) {
          ftm::idNode nodeT = queue.front();
          queue.pop();
          ftm::idNode nodeParent = mTree.tree.getParentSafe(nodeT);
          float s = getSign(scalars[nodeParent]);
          if((isJT and scalars[nodeT] >= scalars[nodeParent] * (1 - s * 1e-6))
             or (not isJT
                 and scalars[nodeT] <= scalars[nodeParent] * (1 + s * 1e-6))) {
            float newValue
              = scalars[nodeParent] * (1 + (isJT ? -1.0 : 1.0) * s * 1e-6);
            scalars[nodeT] = newValue;
            std::vector<ftm::idNode> children;
            mTree.tree.getChildren(nodeT, children);
            for(auto &child : children)
              queue.emplace(child);
          }
        }
      };
  std::vector<float> scalars;
  getTreeScalars(mTree, scalars);
  std::queue<ftm::idNode> queue;
  auto root = mTree.tree.getRoot();
  queue.emplace(root);
  while(!queue.empty()) {
    ftm::idNode node = queue.front();
    queue.pop();
    ftm::idNode nodeParent = mTree.tree.getParentSafe(node);
    float s = getSign(scalars[nodeParent]);
    if(!mTree.tree.isRoot(node)
       and ((isJT and scalars[node] >= scalars[nodeParent] * (1 - s * 1e-6))
            or (not isJT
                and scalars[node] <= scalars[nodeParent] * (1 + s * 1e-6)))) {
      shiftSubtree(node, scalars);
    }
    std::vector<ftm::idNode> children;
    mTree.tree.getChildren(node, children);
    for(auto &child : children)
      queue.emplace(child);
  }
  ftm::setTreeScalars<float>(mTree, scalars);
}

//  ---------------------------------------------------------------------------
//  --- Utils
//  ---------------------------------------------------------------------------
void ttk::MergeTreeAutoencoder::copyParams(
  std::vector<TorchUtils::TorchMergeTree<float>> &srcOrigins,
  std::vector<TorchUtils::TorchMergeTree<float>> &srcOriginsPrime,
  std::vector<torch::Tensor> &srcVS,
  std::vector<torch::Tensor> &srcVSPrime,
  std::vector<TorchUtils::TorchMergeTree<float>> &srcOrigins2,
  std::vector<TorchUtils::TorchMergeTree<float>> &srcOrigins2Prime,
  std::vector<torch::Tensor> &srcVS2,
  std::vector<torch::Tensor> &srcVS2Prime,
  std::vector<std::vector<torch::Tensor>> &srcAlphas,
  std::vector<TorchUtils::TorchMergeTree<float>> &dstOrigins,
  std::vector<TorchUtils::TorchMergeTree<float>> &dstOriginsPrime,
  std::vector<torch::Tensor> &dstVS,
  std::vector<torch::Tensor> &dstVSPrime,
  std::vector<TorchUtils::TorchMergeTree<float>> &dstOrigins2,
  std::vector<TorchUtils::TorchMergeTree<float>> &dstOrigins2Prime,
  std::vector<torch::Tensor> &dstVS2,
  std::vector<torch::Tensor> &dstVS2Prime,
  std::vector<std::vector<torch::Tensor>> &dstAlphas) {
  dstOrigins.resize(noLayers_);
  dstOriginsPrime.resize(noLayers_);
  dstVS.resize(noLayers_);
  dstVSPrime.resize(noLayers_);
  dstAlphas.resize(srcAlphas.size(), std::vector<torch::Tensor>(noLayers_));
  if(useDoubleInput_) {
    dstOrigins2.resize(noLayers_);
    dstOrigins2Prime.resize(noLayers_);
    dstVS2.resize(noLayers_);
    dstVS2Prime.resize(noLayers_);
  }
  for(unsigned int l = 0; l < noLayers_; ++l) {
    TorchUtils::copyTorchMergeTree(srcOrigins[l], dstOrigins[l]);
    TorchUtils::copyTorchMergeTree(srcOriginsPrime[l], dstOriginsPrime[l]);
    TorchUtils::copyTensor(srcVS[l], dstVS[l]);
    TorchUtils::copyTensor(srcVSPrime[l], dstVSPrime[l]);
    if(useDoubleInput_) {
      TorchUtils::copyTorchMergeTree(srcOrigins2[l], dstOrigins2[l]);
      TorchUtils::copyTorchMergeTree(srcOrigins2Prime[l], dstOrigins2Prime[l]);
      TorchUtils::copyTensor(srcVS2[l], dstVS2[l]);
      TorchUtils::copyTensor(srcVS2Prime[l], dstVS2Prime[l]);
    }
    for(unsigned int i = 0; i < srcAlphas.size(); ++i)
      TorchUtils::copyTensor(srcAlphas[i][l], dstAlphas[i][l]);
  }
}

void ttk::MergeTreeAutoencoder::copyParams(
  std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &src,
  std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &dst) {
  dst.resize(src.size());
  for(unsigned int i = 0; i < src.size(); ++i) {
    dst[i].resize(src[i].size());
    for(unsigned int j = 0; j < src[i].size(); ++j)
      TorchUtils::copyTorchMergeTree(src[i][j], dst[i][j]);
  }
}

void ttk::MergeTreeAutoencoder::getDeltaProjTensor(
  torch::Tensor &diagTensor, torch::Tensor &deltaProjTensor) {
  deltaProjTensor
    = (diagTensor.index({Slice(), 0}) + diagTensor.index({Slice(), 1})) / 2.0;
  deltaProjTensor = deltaProjTensor.reshape({-1, 1});
  deltaProjTensor = torch::cat({deltaProjTensor, deltaProjTensor}, 1);
}

void ttk::MergeTreeAutoencoder::dataReorderingGivenMatching(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &tree2,
  torch::Tensor &tree1ProjIndexer,
  torch::Tensor &tree2ReorderingIndexes,
  torch::Tensor &tree2ReorderedTensor,
  torch::Tensor &tree2DeltaProjTensor,
  torch::Tensor &tree1ReorderedTensor,
  torch::Tensor &tree2ProjIndexer,
  bool doubleReordering) {
  // Reorder tree2 tensor
  torch::Tensor tree2DiagTensor = tree2.tensor.reshape({-1, 2});
  tree2ReorderedTensor = torch::cat({tree2DiagTensor, torch::zeros({1, 2})});
  tree2ReorderedTensor = tree2ReorderedTensor.index({tree2ReorderingIndexes});

  // Create tree projection given matching
  torch::Tensor treeDiagTensor = tree.tensor.reshape({-1, 2});
  getDeltaProjTensor(treeDiagTensor, tree2DeltaProjTensor);
  tree2DeltaProjTensor = tree2DeltaProjTensor * tree1ProjIndexer;

  // Double reordering
  if(doubleReordering) {
    torch::Tensor tree1DeltaProjTensor;
    getDeltaProjTensor(tree2DiagTensor, tree1DeltaProjTensor);
    torch::Tensor tree2ProjIndexerR = tree2ProjIndexer.reshape({-1});
    tree1DeltaProjTensor = tree1DeltaProjTensor.index({tree2ProjIndexerR});
    tree1ReorderedTensor = torch::cat({treeDiagTensor, tree1DeltaProjTensor});
    tree1ReorderedTensor = tree1ReorderedTensor.reshape({-1, 1});
    torch::Tensor tree2UnmatchedTensor
      = tree2DiagTensor.index({tree2ProjIndexerR});
    tree2ReorderedTensor
      = torch::cat({tree2ReorderedTensor, tree2UnmatchedTensor});
    tree2DeltaProjTensor = torch::cat(
      {tree2DeltaProjTensor, torch::zeros_like(tree2UnmatchedTensor)});
  }

  // Reshape
  tree2ReorderedTensor = tree2ReorderedTensor.reshape({-1, 1});
  tree2DeltaProjTensor = tree2DeltaProjTensor.reshape({-1, 1});
}

void ttk::MergeTreeAutoencoder::dataReorderingGivenMatching(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &tree2,
  torch::Tensor &tree1ProjIndexer,
  torch::Tensor &tree2ReorderingIndexes,
  torch::Tensor &tree2ReorderedTensor,
  torch::Tensor &tree2DeltaProjTensor) {
  torch::Tensor tree1ReorderedTensor;
  torch::Tensor tree2ProjIndexer;
  bool doubleReordering = false;
  dataReorderingGivenMatching(tree, tree2, tree1ProjIndexer,
                              tree2ReorderingIndexes, tree2ReorderedTensor,
                              tree2DeltaProjTensor, tree1ReorderedTensor,
                              tree2ProjIndexer, doubleReordering);
}

void ttk::MergeTreeAutoencoder::dataReorderingGivenMatching(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &tree2,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
  torch::Tensor &tree1ReorderedTensor,
  torch::Tensor &tree2ReorderedTensor,
  bool doubleReordering) {
  // Get tensor matching
  std::vector<int> tensorMatching;
  TorchUtils::getTensorMatching(tree, tree2, matching, tensorMatching);
  torch::Tensor tree2ReorderingIndexes = torch::tensor(tensorMatching);
  torch::Tensor tree1ProjIndexer
    = (tree2ReorderingIndexes == -1).reshape({-1, 1});
  // Reorder tensor
  torch::Tensor tree2DeltaProjTensor;
  if(not doubleReordering) {
    dataReorderingGivenMatching(tree, tree2, tree1ProjIndexer,
                                tree2ReorderingIndexes, tree2ReorderedTensor,
                                tree2DeltaProjTensor);
  } else {
    std::vector<int> tensorMatching2;
    TorchUtils::getInverseTensorMatching(tree, tree2, matching, tensorMatching);
    torch::Tensor tree1ReorderingIndexes = torch::tensor(tensorMatching);
    torch::Tensor tree2ProjIndexer
      = (tree1ReorderingIndexes == -1).reshape({-1, 1});
    dataReorderingGivenMatching(tree, tree2, tree1ProjIndexer,
                                tree2ReorderingIndexes, tree2ReorderedTensor,
                                tree2DeltaProjTensor, tree1ReorderedTensor,
                                tree2ProjIndexer, doubleReordering);
  }
  tree2ReorderedTensor = tree2ReorderedTensor + tree2DeltaProjTensor;
}

void ttk::MergeTreeAutoencoder::dataReorderingGivenMatching(
  TorchUtils::TorchMergeTree<float> &tree,
  TorchUtils::TorchMergeTree<float> &tree2,
  std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
  torch::Tensor &tree2ReorderedTensor) {
  torch::Tensor tree1ReorderedTensor;
  bool doubleReordering = false;
  dataReorderingGivenMatching(tree, tree2, matching, tree1ReorderedTensor,
                              tree2ReorderedTensor, doubleReordering);
}

void ttk::MergeTreeAutoencoder::meanBirthShift(torch::Tensor &diagTensor,
                                               torch::Tensor &diagBaseTensor) {
  torch::Tensor birthShiftValue = diagBaseTensor.index({Slice(), 0}).mean()
                                  - diagTensor.index({Slice(), 0}).mean();
  torch::Tensor shiftTensor
    = torch::full({diagTensor.sizes()[0], 2}, birthShiftValue.item<float>());
  diagTensor.index_put_({None}, diagTensor + shiftTensor);
}

void ttk::MergeTreeAutoencoder::meanBirthMaxPersShift(
  torch::Tensor &tensor, torch::Tensor &baseTensor) {
  torch::Tensor diagTensor = tensor.reshape({-1, 2});
  torch::Tensor diagBaseTensor = baseTensor.reshape({-1, 2});
  // Shift to have same max pers
  torch::Tensor baseMaxPers
    = (diagBaseTensor.index({Slice(), 1}) - diagBaseTensor.index({Slice(), 0}))
        .max();
  torch::Tensor maxPers
    = (diagTensor.index({Slice(), 1}) - diagTensor.index({Slice(), 0})).max();
  torch::Tensor shiftTensor = (baseMaxPers - maxPers) / 2.0;
  shiftTensor = torch::stack({-shiftTensor, shiftTensor});
  diagTensor.index_put_({None}, diagTensor + shiftTensor);
  // Shift to have same birth mean
  meanBirthShift(diagTensor, diagBaseTensor);
}

void ttk::MergeTreeAutoencoder::belowDiagonalPointsShift(
  torch::Tensor &tensor, torch::Tensor &backupTensor) {
  torch::Tensor oPDiag = tensor.reshape({-1, 2});
  torch::Tensor badPointsIndexer
    = (oPDiag.index({Slice(), 0}) > oPDiag.index({Slice(), 1}));
  torch::Tensor goodPoints = oPDiag.index({~badPointsIndexer});
  if(goodPoints.sizes()[0] == 0)
    goodPoints = backupTensor.reshape({-1, 2});
  torch::Tensor badPoints = oPDiag.index({badPointsIndexer});
  // Shift to be above diagonal with median pers
  torch::Tensor pers
    = (goodPoints.index({Slice(), 1}) - goodPoints.index({Slice(), 0}))
        .median();
  torch::Tensor shiftTensor
    = (torch::full({badPoints.sizes()[0], 1}, pers.item<float>())
       - badPoints.index({Slice(), 1}).reshape({-1, 1})
       + badPoints.index({Slice(), 0}).reshape({-1, 1}))
      / 2.0;
  shiftTensor = torch::cat({-shiftTensor, shiftTensor}, 1);
  badPoints = badPoints + shiftTensor;
  // Shift to have same birth mean
  /*if(goodPoints.sizes()[0] != 1 or badPoints.sizes()[0] != 1)
    meanBirthShift(badPoints, goodPoints);*/
  // Update tensor
  oPDiag.index_put_({badPointsIndexer}, badPoints);
  tensor = oPDiag.reshape({-1, 1}).detach();
}

void ttk::MergeTreeAutoencoder::normalizeVectors(torch::Tensor &originTensor,
                                                 torch::Tensor &vectorsTensor) {
  torch::Tensor vSliced = vectorsTensor.index({Slice(2, None)});
  vSliced.index_put_({None}, vSliced / (originTensor[1] - originTensor[0]));
}

void ttk::MergeTreeAutoencoder::normalizeVectors(
  TorchUtils::TorchMergeTree<float> &origin,
  std::vector<std::vector<double>> &vectors) {
  std::queue<ftm::idNode> queue;
  queue.emplace(origin.mTree.tree.getRoot());
  while(!queue.empty()) {
    ftm::idNode node = queue.front();
    queue.pop();
    if(not origin.mTree.tree.isRoot(node))
      for(unsigned int i = 0; i < 2; ++i)
        vectors[node][i] /= (origin.tensor[1] - origin.tensor[0]).item<float>();
    std::vector<ftm::idNode> children;
    origin.mTree.tree.getChildren(node, children);
    for(auto &child : children)
      queue.emplace(child);
  }
}

unsigned int ttk::MergeTreeAutoencoder::getLatentLayerIndex() {
  unsigned int idx = noLayers_ / 2 - 1;
  if(idx > noLayers_) // unsigned negativeness
    idx = 0;
  return idx;
}

// TODO make it work for merge trees
bool ttk::MergeTreeAutoencoder::isThereMissingPairs(
  TorchUtils::TorchMergeTree<float> &interpolation) {
  float maxPers
    = interpolation.mTree.tree.template getMaximumPersistence<float>();
  torch::Tensor interTensor = interpolation.tensor;
  torch::Tensor indexer
    = torch::abs(interTensor.reshape({-1, 2}).index({Slice(), 0})
                 - interTensor.reshape({-1, 2}).index({Slice(), 1}))
      > (maxPers * 0.001 / 100.0);
  torch::Tensor indexed = interTensor.reshape({-1, 2}).index({indexer});
  return indexed.sizes()[0] > interpolation.mTree.tree.getRealNumberOfNodes();
}

//  ---------------------------------------------------------------------------
//  --- Testing
//  ---------------------------------------------------------------------------
void ttk::MergeTreeAutoencoder::makeExponentialExample(
  std::vector<ftm::MergeTree<float>> &trees,
  std::vector<ftm::MergeTree<float>> &trees2) {
  trees.clear();
  trees2.clear();
  std::vector<std::vector<float>> allScalarsVector;
  std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode>>> allArcs;

  int noTrees = 5;
  allScalarsVector.resize(noTrees);
  allArcs
    = std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode>>>(noTrees);
  for(int i = 0; i < noTrees; ++i)
    allArcs[i] = std::vector<std::tuple<ftm::idNode, ftm::idNode>>{
      std::make_tuple(0, 3), std::make_tuple(2, 3), std::make_tuple(3, 1)};
  allScalarsVector[0] = std::vector<float>{-10, 70, -8, 60};
  allScalarsVector[1] = std::vector<float>{-9.5820, 70.0053, -7.5820, 59.9947};
  allScalarsVector[2] = std::vector<float>{-9.2340, 70.2020, -7.2340, 59.7980};
  allScalarsVector[3] = std::vector<float>{-9.0840, 70.5907, -7.0840, 59.4093};
  allScalarsVector[4] = std::vector<float>{-9., 71., -7., 59.};

  // Create trees
  trees = std::vector<ftm::MergeTree<float>>(allScalarsVector.size());
  for(unsigned int i = 0; i < allScalarsVector.size(); ++i) {
    std::vector<float> scalarsVector(allScalarsVector[i].size());
    for(unsigned int j = 0; j < allScalarsVector[i].size(); ++j)
      scalarsVector[j] = allScalarsVector[i][j];
    trees[i] = makeFakeMergeTree(scalarsVector, allArcs[i]);
    if(not normalizedWasserstein_) {
      ftm::computePersistencePairs<float>(&(trees[i].tree));
      std::vector<std::vector<ftm::idNode>> treeNodeMerged(
        trees[i].tree.getNumberOfNodes());
      computeBranchDecomposition<float>(&(trees[i].tree), treeNodeMerged);
    }
  }
}

bool ttk::MergeTreeAutoencoder::isTreeHasBigValues(ftm::MergeTree<float> &mTree,
                                                   float threshold) {
  bool found = false;
  for(unsigned int n = 0; n < mTree.tree.getNumberOfNodes(); ++n) {
    if(mTree.tree.isNodeAlone(n))
      continue;
    auto birthDeath = mTree.tree.template getBirthDeath<float>(n);
    if(std::abs(std::get<0>(birthDeath)) > threshold
       or std::abs(std::get<1>(birthDeath)) > threshold) {
      found = true;
      break;
    }
  }
  return found;
}
#endif
