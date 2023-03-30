#include <ttkFTMTreeUtils.h>
#include <ttkMergeTreeAutoencoder.h>
#include <ttkMergeTreeVisualization.h>

#include <vtkInformation.h>

#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkFloatArray.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkTable.h>

#include <ttkMacros.h>
#include <ttkUtils.h>

// A VTK macro that enables the instantiation of this class via ::New()
// You do not have to modify this
vtkStandardNewMacro(ttkMergeTreeAutoencoder);

/**
 * Implement the filter constructor and destructor in the cpp file.
 *
 * The constructor has to specify the number of input and output ports
 * with the functions SetNumberOfInputPorts and SetNumberOfOutputPorts,
 * respectively. It should also set default values for all filter
 * parameters.
 *
 * The destructor is usually empty unless you want to manage memory
 * explicitly, by for example allocating memory on the heap that needs
 * to be freed when the filter is destroyed.
 */
ttkMergeTreeAutoencoder::ttkMergeTreeAutoencoder() {
  this->SetNumberOfInputPorts(4);
  this->SetNumberOfOutputPorts(4);
}

/**
 * Specify the required input data type of each input port
 *
 * This method specifies the required input object data types of the
 * filter by adding the vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE() key to
 * the port information.
 */
int ttkMergeTreeAutoencoder::FillInputPortInformation(int port,
                                                      vtkInformation *info) {
  if(port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkMultiBlockDataSet");
  } else if(port == 1) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkMultiBlockDataSet");
    info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  } else if(port == 2 or port == 3) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkTable");
    info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  } else
    return 0;
  return 1;
}

/**
 * Specify the data object type of each output port
 *
 * This method specifies in the port information object the data type of the
 * corresponding output objects. It is possible to either explicitly
 * specify a type by adding a vtkDataObject::DATA_TYPE_NAME() key:
 *
 *      info->Set( vtkDataObject::DATA_TYPE_NAME(), "vtkUnstructuredGrid" );
 *
 * or to pass a type of an input port to an output port by adding the
 * ttkAlgorithm::SAME_DATA_TYPE_AS_INPUT_PORT() key (see below).
 *
 * Note: prior to the execution of the RequestData method the pipeline will
 * initialize empty output data objects based on this information.
 */
int ttkMergeTreeAutoencoder::FillOutputPortInformation(int port,
                                                       vtkInformation *info) {
  if(port == 0 or port == 1 or port == 2 or port == 3) {
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkMultiBlockDataSet");
  } else {
    return 0;
  }
  return 1;
}

/**
 * Pass VTK data to the base code and convert base code output to VTK
 *
 * This method is called during the pipeline execution to update the
 * already initialized output data objects based on the given input
 * data objects and filter parameters.
 *
 * Note:
 *     1) The passed input data objects are validated based on the information
 *        provided by the FillInputPortInformation method.
 *     2) The output objects are already initialized based on the information
 *        provided by the FillOutputPortInformation method.
 */
int ttkMergeTreeAutoencoder::RequestData(vtkInformation *ttkNotUsed(request),
                                         vtkInformationVector **inputVector,
                                         vtkInformationVector *outputVector) {
#ifndef TTK_ENABLE_TORCH
  printErr("This filter requires Torch.");
  return 0;
#endif

  // ------------------------------------------------------------------------------------
  // --- Get input object from input vector
  // ------------------------------------------------------------------------------------
  auto blocks = vtkMultiBlockDataSet::GetData(inputVector[0], 0);
  auto blocks2 = vtkMultiBlockDataSet::GetData(inputVector[1], 0);
  auto table = vtkTable::GetData(inputVector[2], 0);
  auto table2 = vtkTable::GetData(inputVector[3], 0);

  // ------------------------------------------------------------------------------------
  // --- Load blocks
  // ------------------------------------------------------------------------------------
  std::vector<vtkSmartPointer<vtkMultiBlockDataSet>> inputTrees, inputTrees2;
  ttk::ftm::loadBlocks(inputTrees, blocks);
  ttk::ftm::loadBlocks(inputTrees2, blocks2);

  // Load table
  clusterAsgn_.clear();
  vtkAbstractArray *clusterAsgn;
  if(table) {
    clusterAsgn = table->GetColumnByName("ClusterAssignment");
    if(clusterAsgn) {
      clusterAsgn_.resize(clusterAsgn->GetNumberOfValues());
      for(unsigned int i = 0; i < clusterAsgn_.size(); ++i)
        clusterAsgn_[i] = clusterAsgn->GetVariantValue(i).ToInt();
    }
  }
  if((not table or not clusterAsgn) and clusteringLossWeight_ != 0) {
    printErr("You must provide ClusterAssignment table column in info input to "
             "use clustering loss");
    return 0;
  }
  if(clusteringLossWeight_ != 0) {
    for(auto &e : clusterAsgn_)
      std::cout << e << " ";
    std::cout << std::endl;
  }

  if(table2) {
    std::vector<std::string> names{
      getTableCoefficientName(2, 0), getTableCoefficientName(2, 1)};
    std::vector<vtkAbstractArray *> columns{
      table2->GetColumnByName(names[0].c_str()),
      table2->GetColumnByName(names[1].c_str())};
    if(columns[0] and columns[1]) {
      auto numberOfInputs = table2->GetNumberOfRows();
      unsigned int noCoefs = 2;
      customAlphas_.resize(numberOfInputs, std::vector<float>(noCoefs, 0));
      for(unsigned int i = 0; i < numberOfInputs; ++i)
        for(unsigned int j = 0; j < noCoefs; ++j)
          customAlphas_[i][j] = columns[j]->GetVariantValue(i).ToFloat();
    }
  }

  // ------------------------------------------------------------------------------------
  // If we have already computed once but the input has changed
  if((treesNodes.size() != 0 and inputTrees[0]->GetBlock(0) != treesNodes[0])
     or (treesNodes2.size() != inputTrees2.size()))
    resetDataVisualization();

  // Parameters
  branchDecomposition_ = true;
  if(not normalizedWasserstein_) {
    oldEpsilonTree1 = epsilonTree1_;
    epsilonTree1_ = 100;
  } else
    epsilonTree1_ = oldEpsilonTree1;
  if(normalizedWasserstein_)
    printMsg("Computation with normalized Wasserstein.");
  else
    printMsg("Computation without normalized Wasserstein.");

  return run(outputVector, inputTrees, inputTrees2);
}

int ttkMergeTreeAutoencoder::run(
  vtkInformationVector *outputVector,
  std::vector<vtkSmartPointer<vtkMultiBlockDataSet>> &inputTrees,
  std::vector<vtkSmartPointer<vtkMultiBlockDataSet>> &inputTrees2) {
  if(not isDataVisualizationFilled())
    runCompute(outputVector, inputTrees, inputTrees2);
  runOutput(outputVector, inputTrees, inputTrees2);
  return 1;
}

int ttkMergeTreeAutoencoder::runCompute(
  vtkInformationVector *ttkNotUsed(outputVector),
  std::vector<vtkSmartPointer<vtkMultiBlockDataSet>> &inputTrees,
  std::vector<vtkSmartPointer<vtkMultiBlockDataSet>> &inputTrees2) {
  // ------------------------------------------------------------------------------------
  // --- Construct trees
  // ------------------------------------------------------------------------------------
  std::vector<ttk::ftm::MergeTree<float>> intermediateMTrees,
    intermediateMTrees2;

  bool useSadMaxPairs = (mixtureCoefficient_ == 0);
  isPersistenceDiagram_ = ttk::ftm::constructTrees<float>(
    inputTrees, intermediateMTrees, treesNodes, treesArcs, treesSegmentation,
    useSadMaxPairs);
  // If merge trees are provided in input and normalization is not asked
  convertToDiagram_
    = (not isPersistenceDiagram_ and not normalizedWasserstein_);
  if(not isPersistenceDiagram_
     or (mixtureCoefficient_ != 0 and mixtureCoefficient_ != 1)) {
    auto &inputTrees2ToUse
      = (not isPersistenceDiagram_ ? inputTrees2 : inputTrees);
    ttk::ftm::constructTrees<float>(inputTrees2ToUse, intermediateMTrees2,
                                    treesNodes2, treesArcs2, treesSegmentation2,
                                    !useSadMaxPairs);
  }
  isPersistenceDiagram_ |= (not normalizedWasserstein_);

  const int numInputs = intermediateMTrees.size();
  const int numInputs2 = intermediateMTrees2.size();
  setDataVisualization(numInputs, numInputs2);

  // ------------------------------------------------------------------------------------
  // --- Call base
  // ------------------------------------------------------------------------------------
#ifdef TTK_ENABLE_TORCH
  if(doCompute_ or not hasComputedOnce_)
    execute(intermediateMTrees, intermediateMTrees2);
  else
    createCustomRecs();
#endif

  ttk::ftm::mergeTreesTemplateToDouble<float>(
    intermediateMTrees, intermediateDTrees);

  return 1;
}

void ttkMergeTreeAutoencoder::makeOneOutput(
  ttk::ftm::MergeTree<float> &tree,
  vtkUnstructuredGrid *treeNodes,
  std::vector<int> &treeNodeCorr,
  vtkDataSet *treeSegmentation,
  vtkSmartPointer<vtkUnstructuredGrid> &vtkOutputNode,
  vtkSmartPointer<vtkUnstructuredGrid> &vtkOutputArc,
  vtkSmartPointer<vtkDataSet> &vtkOutputSegmentation,
  unsigned int treeID,
  std::vector<std::tuple<std::string, std::vector<int>>> &customIntArrays,
  std::vector<std::tuple<std::string, std::vector<double>>> &customDoubleArrays,
  bool outputSegmentation) {
  vtkOutputNode = vtkSmartPointer<vtkUnstructuredGrid>::New();
  vtkOutputArc = vtkSmartPointer<vtkUnstructuredGrid>::New();

  ttkMergeTreeVisualization visuMakerBary;
  visuMakerBary.setShiftMode(-1); // Line
  visuMakerBary.setVtkOutputNode(vtkOutputNode);
  if(not isPersistenceDiagram_)
    visuMakerBary.setVtkOutputArc(vtkOutputArc);
  else {
    visuMakerBary.setVtkOutputArc(vtkOutputNode);
    /*if(mixtureCoefficient_ != 0 and mixtureCoefficient_ != 1)
      visuMakerBary.setIsPDSadMax(blockId);
    else*/
    visuMakerBary.setIsPDSadMax(mixtureCoefficient_ == 0);
  }
  for(auto &tup : customIntArrays)
    visuMakerBary.addCustomIntArray(std::get<0>(tup), std::get<1>(tup));
  for(auto &tup : customDoubleArrays)
    visuMakerBary.addCustomArray(std::get<0>(tup), std::get<1>(tup));
  visuMakerBary.setDebugLevel(this->debugLevel_);
  visuMakerBary.setIsPersistenceDiagram(isPersistenceDiagram_);
  visuMakerBary.setConvertedToDiagram(convertToDiagram_);
  if(outputSegmentation) {
    visuMakerBary.setTreesNodes(treeNodes);
    visuMakerBary.setTreesNodeCorrMesh(treeNodeCorr);
    vtkOutputSegmentation = vtkSmartPointer<vtkUnstructuredGrid>::New();
    visuMakerBary.setTreesSegmentation(treeSegmentation);
    visuMakerBary.setPlanarLayout(false);
    visuMakerBary.setOutputSegmentation(true);
    visuMakerBary.setVtkOutputSegmentation(vtkOutputSegmentation);
  } else {
    visuMakerBary.setPlanarLayout(true);
  }
  visuMakerBary.setISampleOffset(treeID);
  visuMakerBary.makeTreesOutput<float>(&(tree.tree));
}

void ttkMergeTreeAutoencoder::makeManyOutput(
  std::vector<ttk::ftm::MergeTree<float> *> &trees,
  std::vector<vtkUnstructuredGrid *> &treesNodesT,
  std::vector<std::vector<int>> &treesNodeCorr,
  std::vector<vtkDataSet *> &treesSegmentationT,
  vtkSmartPointer<vtkMultiBlockDataSet> &output,
  std::vector<std::vector<std::tuple<std::string, std::vector<int>>>>
    &customIntArrays,
  std::vector<std::vector<std::tuple<std::string, std::vector<double>>>>
    &customDoubleArrays) {
  vtkSmartPointer<vtkMultiBlockDataSet> allNodes
    = vtkSmartPointer<vtkMultiBlockDataSet>::New();
  // allNodes->SetNumberOfBlocks(trees.size());
  vtkSmartPointer<vtkMultiBlockDataSet> allArcs;
  if(not isPersistenceDiagram_) {
    allArcs = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    // allArcs->SetNumberOfBlocks(trees.size());
  }
  bool outputSegmentation
    = !treesSegmentationT.empty() and treesSegmentationT[0];
  vtkSmartPointer<vtkMultiBlockDataSet> allSegs;
  if(outputSegmentation) {
    allSegs = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    // allSegs->SetNumberOfBlocks(trees.size());
  }
  int shift = 0;
  for(unsigned int i = 0; i < trees.size(); ++i) {
    if(trees[i]->tree.template getMaximumPersistence<float>() == 0) {
      ++shift;
      continue;
    }
    vtkUnstructuredGrid *treeNodes = nullptr;
    vtkDataSet *treeSegmentation = nullptr;
    std::vector<int> treeNodeCorr;
    if(outputSegmentation) {
      treeNodes = treesNodesT[i];
      treeSegmentation = treesSegmentationT[i];
      treeNodeCorr = treesNodeCorr[i];
    }
    vtkSmartPointer<vtkUnstructuredGrid> vtkOutputNode, vtkOutputArc;
    vtkSmartPointer<vtkDataSet> vtkOutputSegmentation;
    makeOneOutput(*(trees[i]), treeNodes, treeNodeCorr, treeSegmentation,
                  vtkOutputNode, vtkOutputArc, vtkOutputSegmentation, i,
                  customIntArrays[i], customDoubleArrays[i],
                  outputSegmentation);
    allNodes->SetBlock(i - shift, vtkOutputNode);
    if(not isPersistenceDiagram_)
      allArcs->SetBlock(i - shift, vtkOutputArc);
    if(outputSegmentation)
      allSegs->SetBlock(i - shift, vtkOutputSegmentation);
  }
  if(not isPersistenceDiagram_) {
    output->SetNumberOfBlocks(2);
    output->SetBlock(0, allNodes);
    output->SetBlock(1, allArcs);
    if(outputSegmentation)
      output->SetBlock(2, allSegs);
  } else {
    if(not outputSegmentation) {
      output->ShallowCopy(allNodes);
    } else {
      output->SetNumberOfBlocks(2);
      output->SetBlock(0, allNodes);
      output->SetBlock(1, allSegs);
    }
  }
}

void ttkMergeTreeAutoencoder::makeManyOutput(
  std::vector<ttk::ftm::MergeTree<float> *> &trees,
  vtkSmartPointer<vtkMultiBlockDataSet> &output,
  std::vector<std::vector<std::tuple<std::string, std::vector<int>>>>
    &customIntArrays,
  std::vector<std::vector<std::tuple<std::string, std::vector<double>>>>
    &customDoubleArrays) {
  std::vector<vtkUnstructuredGrid *> treesNodesT;
  std::vector<vtkDataSet *> treesSegmentationT;
  std::vector<std::vector<int>> treesNodeCorr;
  makeManyOutput(trees, treesNodesT, treesNodeCorr, treesSegmentationT, output,
                 customIntArrays, customDoubleArrays);
}

void ttkMergeTreeAutoencoder::makeManyOutput(
  std::vector<ttk::ftm::MergeTree<float> *> &trees,
  vtkSmartPointer<vtkMultiBlockDataSet> &output) {
  std::vector<std::vector<std::tuple<std::string, std::vector<int>>>>
    customIntArrays(trees.size());
  std::vector<std::vector<std::tuple<std::string, std::vector<double>>>>
    customDoubleArrays(trees.size());
  makeManyOutput(trees, output, customIntArrays, customDoubleArrays);
}

// TODO manage double input
int ttkMergeTreeAutoencoder::runOutput(
  vtkInformationVector *outputVector,
  std::vector<vtkSmartPointer<vtkMultiBlockDataSet>> &inputTrees,
  std::vector<vtkSmartPointer<vtkMultiBlockDataSet>> &ttkNotUsed(inputTrees2)) {
  if(not createOutput_)
    return 1;
  // ------------------------------------------------------------------------------------
  // --- Create output
  // ------------------------------------------------------------------------------------
  auto output_data = vtkMultiBlockDataSet::GetData(outputVector, 0);
  auto output_origins = vtkMultiBlockDataSet::GetData(outputVector, 1);
  auto output_coef = vtkMultiBlockDataSet::GetData(outputVector, 2);
  auto output_vectors = vtkMultiBlockDataSet::GetData(outputVector, 3);

#ifdef TTK_ENABLE_TORCH
  // ------------------------------------------
  // --- Tracking information
  // ------------------------------------------
  std::vector<std::vector<ttk::ftm::idNode>> originsMatchingVector;
  std::vector<std::vector<double>> originsPersPercent, originsPersDiff;
  std::vector<double> originPersPercent, originPersDiff;
  std::vector<int> originPersistenceOrder;
  {
    originsMatchingVector.resize(originsMatchings_.size());
    originsPersPercent.resize(originsMatchings_.size());
    originsPersDiff.resize(originsMatchings_.size());
    for(unsigned int l = 0; l < originsMatchings_.size(); ++l) {
      auto &tree1 = (l == 0 ? origins_[0] : originsPrime_[l - 1]);
      auto &tree2 = (l == 0 ? originsPrime_[0] : originsPrime_[l]);
      getInverseMatchingVector(tree1.mTree, tree2.mTree, originsMatchings_[l],
                               originsMatchingVector[l]);
      if(l != 0) {
        for(unsigned int i = 0; i < originsMatchingVector[l].size(); ++i)
          if(originsMatchingVector[l][i] < originsMatchingVector[l - 1].size())
            originsMatchingVector[l][i]
              = originsMatchingVector[l - 1][originsMatchingVector[l][i]];
      }
      originsPersPercent[l].resize(tree2.mTree.tree.getNumberOfNodes());
      originsPersDiff[l].resize(tree2.mTree.tree.getNumberOfNodes());
      for(unsigned int i = 0; i < originsMatchingVector[l].size(); ++i) {
        if(originsMatchingVector[l][i]
           >= origins_[0].mTree.tree.getNumberOfNodes())
          continue;
        auto pers = origins_[0].mTree.tree.template getNodePersistence<float>(
          originsMatchingVector[l][i]);
        auto treePers = tree2.mTree.tree.template getNodePersistence<float>(i);
        originsPersPercent[l][i] = treePers * 100 / pers;
        originsPersDiff[l][i] = treePers - pers;
      }
    }

    originPersPercent.resize(origins_[0].mTree.tree.getNumberOfNodes());
    originPersDiff.resize(origins_[0].mTree.tree.getNumberOfNodes());
    std::vector<ttk::ftm::idNode> originMatchingVector;
    for(unsigned int l = 0; l < originsMatchings_.size(); ++l) {
      auto &tree1 = (l == 0 ? origins_[0] : originsPrime_[l - 1]);
      auto &tree2 = (l == 0 ? originsPrime_[0] : originsPrime_[l]);
      std::vector<ttk::ftm::idNode> originMatchingVectorT;
      getMatchingVector(
        tree1.mTree, tree2.mTree, originsMatchings_[l], originMatchingVectorT);
      if(l == 0) {
        originMatchingVector = originMatchingVectorT;
      } else {
        for(unsigned int i = 0; i < originMatchingVector.size(); ++i)
          if(originMatchingVector[i] < originMatchingVectorT.size())
            originMatchingVector[i]
              = originMatchingVectorT[originMatchingVector[i]];
      }
    }
    unsigned int l2 = originsMatchings_.size() - 1;
    for(unsigned int i = 0; i < originMatchingVector.size(); ++i) {
      if(originMatchingVector[i] < originsPersDiff[l2].size()) {
        originPersPercent[i] = originsPersPercent[l2][originMatchingVector[i]];
        originPersDiff[i] = originsPersDiff[l2][originMatchingVector[i]];
      }
    }

    originPersistenceOrder.resize(
      origins_[0].mTree.tree.getNumberOfNodes(), -1);
    std::vector<std::tuple<ttk::ftm::idNode, ttk::ftm::idNode, float>>
      pairsBary;
    bool useBD = isPersistenceDiagram_;
    origins_[0].mTree.tree.template getPersistencePairsFromTree<float>(
      pairsBary, useBD);
    for(unsigned int j = 0; j < pairsBary.size(); ++j) {
      int index = pairsBary.size() - 1 - j;
      originPersistenceOrder[std::get<0>(pairsBary[j])] = index;
      originPersistenceOrder[std::get<1>(pairsBary[j])] = index;
    }
  }

  // ------------------------------------------
  // --- Data
  // ------------------------------------------
  output_data->SetNumberOfBlocks(1);
  vtkSmartPointer<vtkMultiBlockDataSet> data
    = vtkSmartPointer<vtkMultiBlockDataSet>::New();
  data->SetNumberOfBlocks(recs_[0].size());
  vtkSmartPointer<vtkMultiBlockDataSet> dataSeg
    = vtkSmartPointer<vtkMultiBlockDataSet>::New();
  dataSeg->SetNumberOfBlocks(recs_.size());
  bool outputSegmentation = !treesSegmentation.empty() and treesSegmentation[0];
  for(unsigned int l = 0; l < recs_[0].size(); ++l) {
    vtkSmartPointer<vtkMultiBlockDataSet> out_layer_i
      = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    out_layer_i->SetNumberOfBlocks(recs_.size());
    std::vector<ttk::ftm::MergeTree<float> *> trees(recs_.size());
    for(unsigned int i = 0; i < recs_.size(); ++i)
      trees[i] = &(recs_[i][l].mTree);

    // Custom arrays
    std::vector<std::vector<std::tuple<std::string, std::vector<int>>>>
      customIntArrays(recs_.size());
    std::vector<std::vector<std::tuple<std::string, std::vector<double>>>>
      customDoubleArrays(recs_.size());
    std::vector<std::vector<ttk::ftm::idNode>> matchingVectors(recs_.size());
    std::vector<std::vector<double>> dataPersPercent, dataPersDiff;
    std::vector<std::vector<int>> dataOriginPersOrder;
    std::vector<std::vector<std::vector<double>>> dataCorrelation;
    if(l < dataMatchings_.size()) {
      for(unsigned int i = 0; i < recs_.size(); ++i) {
        auto &origin = (l == 0 ? origins_[0] : originsPrime_[l - 1]);
        getInverseMatchingVector(origin.mTree, recs_[i][l].mTree,
                                 dataMatchings_[l][i], matchingVectors[i]);
        if(l != 0) {
          for(unsigned int j = 0; j < matchingVectors[i].size(); ++j)
            if(matchingVectors[i][j] < originsMatchingVector[l - 1].size())
              matchingVectors[i][j]
                = originsMatchingVector[l - 1][matchingVectors[i][j]];
        }
      }
    }
    if(l == 0 or l == dataMatchings_.size() - 1) {
      dataPersPercent.resize(recs_.size());
      dataPersDiff.resize(recs_.size());
      for(unsigned int i = 0; i < recs_.size(); ++i) {
        dataPersPercent[i].resize(recs_[i][l].mTree.tree.getNumberOfNodes());
        dataPersDiff[i].resize(recs_[i][l].mTree.tree.getNumberOfNodes());
        std::vector<ttk::ftm::idNode> originMatchingVector;
        std::vector<ttk::ftm::idNode> matchingVector;
        if(l == 0) {
          matchingVector = matchingVectors[i];
          for(unsigned int l2 = 0; l2 < originsMatchings_.size(); ++l2) {
            auto &tree1 = (l2 == 0 ? origins_[0] : originsPrime_[l2 - 1]);
            auto &tree2 = (l2 == 0 ? originsPrime_[0] : originsPrime_[l2]);
            getMatchingVector(tree1.mTree, tree2.mTree, originsMatchings_[l2],
                              originMatchingVector);
            for(unsigned int j = 0; j < matchingVector.size(); ++j)
              if(matchingVector[j] < originMatchingVector.size())
                matchingVector[j] = originMatchingVector[matchingVector[j]];
          }
        } else {
          getInverseMatchingVector(originsPrime_[l - 1].mTree,
                                   recs_[i][l].mTree, dataMatchings_[l][i],
                                   matchingVector);
        }
        unsigned int l2 = originsMatchings_.size() - 1;
        for(unsigned int j = 0; j < matchingVector.size(); ++j) {
          if(matchingVector[j] < originsPersDiff[l2].size()) {
            dataPersDiff[i][j] = originsPersDiff[l2][matchingVector[j]];
            dataPersPercent[i][j] = originsPersPercent[l2][matchingVector[j]];
          }
        }
      }

      if(l == 0) {
        dataCorrelation.resize(recs_.size());
        for(unsigned int i = 0; i < recs_.size(); ++i) {
          dataCorrelation[i].resize(persCorrelationMatrix_[0].size());
          for(unsigned int j = 0; j < persCorrelationMatrix_[0].size(); ++j) {
            dataCorrelation[i][j].resize(
              recs_[i][l].mTree.tree.getNumberOfNodes());
            for(unsigned int k = 0; k < matchingVectors[i].size(); ++k) {
              if(matchingVectors[i][k] < persCorrelationMatrix_.size())
                dataCorrelation[i][j][k]
                  = persCorrelationMatrix_[matchingVectors[i][k]][j];
            }
          }
        }
      }
    }

    if(l == 0 or l == dataMatchings_.size() - 1 or l == recs_[0].size() - 1) {
      dataOriginPersOrder.resize(recs_.size());
      for(unsigned int i = 0; i < recs_.size(); ++i) {
        std::vector<ttk::ftm::idNode> matchingVector = matchingVectors[i];
        if(l == recs_[0].size() - 1) {
          getInverseMatchingVector(recs_[i][0].mTree, recs_[i][l].mTree,
                                   reconstMatchings_[i], matchingVector);
          std::vector<ttk::ftm::idNode> matchingVectorT;
          getInverseMatchingVector(origins_[0].mTree, recs_[i][0].mTree,
                                   dataMatchings_[0][i], matchingVectorT);
          for(unsigned int j = 0; j < matchingVector.size(); ++j)
            if(matchingVector[j] < matchingVectorT.size())
              matchingVector[j] = matchingVectorT[matchingVector[j]];
        }
        dataOriginPersOrder[i].resize(
          recs_[i][l].mTree.tree.getNumberOfNodes());
        for(unsigned int j = 0; j < matchingVector.size(); ++j) {
          if(matchingVector[j] < originPersistenceOrder.size())
            dataOriginPersOrder[i][j]
              = originPersistenceOrder[matchingVector[j]];
          else
            dataOriginPersOrder[i][j] = -1;
        }
      }
    }

    for(unsigned int i = 0; i < recs_.size(); ++i) {
      if(l < dataMatchings_.size()) {
        std::vector<int> customArrayMatching;
        for(auto &e : matchingVectors[i])
          customArrayMatching.emplace_back(e);
        std::string name{"OriginTrueNodeId"};
        customIntArrays[i].emplace_back(
          std::make_tuple(name, customArrayMatching));
        if(l == 0 or l == dataMatchings_.size() - 1) {
          std::string name2{"OriginPersPercent"};
          customDoubleArrays[i].emplace_back(
            std::make_tuple(name2, dataPersPercent[i]));
          std::string name3{"OriginPersDiff"};
          customDoubleArrays[i].emplace_back(
            std::make_tuple(name3, dataPersDiff[i]));
        }
        if(l == 0) {
          for(unsigned int j = 0; j < dataCorrelation[i].size(); ++j) {
            std::string name2
              = getTableCorrelationPersName(dataCorrelation[i].size(), j);
            customDoubleArrays[i].emplace_back(
              std::make_tuple(name2, dataCorrelation[i][j]));
          }
        }
      }
      if(l == 0 or l == dataMatchings_.size() - 1 or l == recs_[0].size() - 1) {
        std::string name4{"OriginPersOrder"};
        customIntArrays[i].emplace_back(
          std::make_tuple(name4, dataOriginPersOrder[i]));
      }
    }

    // Create output
    makeManyOutput(trees, out_layer_i, customIntArrays, customDoubleArrays);
    if(outputSegmentation and l == 0) {
      makeManyOutput(trees, treesNodes, treesNodeCorr_, treesSegmentation,
                     dataSeg, customIntArrays, customDoubleArrays);
    }
    data->SetBlock(l, out_layer_i);
    std::stringstream ss;
    ss << (l == 0 ? "Input" : "Layer") << l;
    data->GetMetaData(l)->Set(vtkCompositeDataSet::NAME(), ss.str());
  }
  output_data->SetBlock(0, data);
  unsigned int num = 0;
  output_data->GetMetaData(num)->Set(
    vtkCompositeDataSet::NAME(), "layersTrees");
  if(outputSegmentation)
    output_data->SetBlock(1, dataSeg);

  if(!customRecs_.empty()) {
    std::vector<std::vector<std::tuple<std::string, std::vector<int>>>>
      customRecsIntArrays(customRecs_.size());
    std::vector<std::vector<std::tuple<std::string, std::vector<double>>>>
      customRecsDoubleArrays(customRecs_.size());
    std::vector<std::vector<int>> customOriginPersOrder(customRecs_.size());
    vtkSmartPointer<vtkMultiBlockDataSet> dataCustom
      = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    // dataCustom->SetNumberOfBlocks(customRecs_.size());
    std::vector<ttk::ftm::MergeTree<float> *> trees(customRecs_.size());
    for(unsigned int i = 0; i < customRecs_.size(); ++i) {
      trees[i] = &(customRecs_[i].mTree);
      std::vector<ttk::ftm::idNode> matchingVector;
      getInverseMatchingVector(origins_[0].mTree, customRecs_[i].mTree,
                               customMatchings_[i], matchingVector);
      customOriginPersOrder[i].resize(
        customRecs_[i].mTree.tree.getNumberOfNodes());
      for(unsigned int j = 0; j < matchingVector.size(); ++j) {
        if(matchingVector[j] < originPersistenceOrder.size())
          customOriginPersOrder[i][j]
            = originPersistenceOrder[matchingVector[j]];
        else
          customOriginPersOrder[i][j] = -1;
      }
      std::string name4{"OriginPersOrder"};
      customRecsIntArrays[i].emplace_back(
        std::make_tuple(name4, customOriginPersOrder[i]));
    }
    makeManyOutput(
      trees, dataCustom, customRecsIntArrays, customRecsDoubleArrays);
    output_data->SetBlock(2, dataCustom);
  }

  // ------------------------------------------
  // --- Origins
  // ------------------------------------------
  output_origins->SetNumberOfBlocks(2);
  // Origins
  vtkSmartPointer<vtkMultiBlockDataSet> origins
    = vtkSmartPointer<vtkMultiBlockDataSet>::New();
  vtkSmartPointer<vtkMultiBlockDataSet> originsP
    = vtkSmartPointer<vtkMultiBlockDataSet>::New();
  origins->SetNumberOfBlocks(noLayers_);
  originsP->SetNumberOfBlocks(noLayers_);
  std::vector<ttk::ftm::MergeTree<float> *> trees(noLayers_);
  std::vector<std::vector<std::tuple<std::string, std::vector<int>>>>
    customIntArrays(noLayers_);
  std::vector<std::vector<std::tuple<std::string, std::vector<double>>>>
    customDoubleArrays(noLayers_);
  for(unsigned int l = 0; l < noLayers_; ++l) {
    trees[l] = &(origins_[l].mTree);
    if(l == 0) {
      std::string name2{"OriginPersPercent"};
      customDoubleArrays[l].emplace_back(
        std::make_tuple(name2, originPersPercent));
      std::string name3{"OriginPersDiff"};
      customDoubleArrays[l].emplace_back(
        std::make_tuple(name3, originPersDiff));
      std::string nameOrder{"OriginPersOrder"};
      customIntArrays[l].emplace_back(
        std::make_tuple(nameOrder, originPersistenceOrder));
    }
  }
  makeManyOutput(trees, origins, customIntArrays, customDoubleArrays);

  customIntArrays.clear();
  customIntArrays.resize(noLayers_);
  customDoubleArrays.clear();
  customDoubleArrays.resize(noLayers_);
  for(unsigned int l = 0; l < noLayers_; ++l) {
    trees[l] = &(originsPrime_[l].mTree);
    if(l < originsMatchingVector.size()) {
      std::vector<int> customArrayMatching,
        originPersOrder(trees[l]->tree.getNumberOfNodes(), -1);
      for(unsigned int i = 0; i < originsMatchingVector[l].size(); ++i) {
        customArrayMatching.emplace_back(originsMatchingVector[l][i]);
        if(originsMatchingVector[l][i] < originPersistenceOrder.size())
          originPersOrder[i]
            = originPersistenceOrder[originsMatchingVector[l][i]];
      }
      std::string name{"OriginTrueNodeId"};
      customIntArrays[l].emplace_back(
        std::make_tuple(name, customArrayMatching));
      std::string nameOrder{"OriginPersOrder"};
      customIntArrays[l].emplace_back(
        std::make_tuple(nameOrder, originPersOrder));
      std::string name2{"OriginPersPercent"};
      customDoubleArrays[l].emplace_back(
        std::make_tuple(name2, originsPersPercent[l]));
      std::string name3{"OriginPersDiff"};
      customDoubleArrays[l].emplace_back(
        std::make_tuple(name3, originsPersDiff[l]));
    }
  }
  makeManyOutput(trees, originsP, customIntArrays, customDoubleArrays);
  output_origins->SetBlock(0, origins);
  output_origins->SetBlock(1, originsP);
  // TODO all origins nodes are together and arcs together. They should probably
  // be splitted in different blocks.
  // for(unsigned int l = 0; l < 2; ++l) {
  for(unsigned int l = 0; l < noLayers_; ++l) {
    if(l >= 2)
      break;
    std::stringstream ss;
    ss << (l == 0 ? "InputOrigin" : "LayerOrigin") << l;
    origins->GetMetaData(l)->Set(vtkCompositeDataSet::NAME(), ss.str());
    ss.str("");
    ss << (l == 0 ? "InputOriginPrime" : "LayerOriginPrime") << l;
    originsP->GetMetaData(l)->Set(vtkCompositeDataSet::NAME(), ss.str());
  }
  num = 0;
  output_origins->GetMetaData(num)->Set(
    vtkCompositeDataSet::NAME(), "layersOrigins");
  num = 1;
  output_origins->GetMetaData(num)->Set(
    vtkCompositeDataSet::NAME(), "layersOriginsPrime");

  // ------------------------------------------
  // --- Coefficients
  // ------------------------------------------
  output_coef->SetNumberOfBlocks(allAlphas_[0].size());
  for(unsigned int l = 0; l < allAlphas_[0].size(); ++l) {
    vtkSmartPointer<vtkTable> coef_table = vtkSmartPointer<vtkTable>::New();
    vtkNew<vtkIntArray> treeIDArray{};
    treeIDArray->SetName("TreeID");
    treeIDArray->SetNumberOfTuples(inputTrees.size());
    for(unsigned int i = 0; i < inputTrees.size(); ++i)
      treeIDArray->SetTuple1(i, i);
    coef_table->AddColumn(treeIDArray);
    auto noVec = allAlphas_[0][l].sizes()[0];
    for(unsigned int v = 0; v < noVec; ++v) {
      // Alphas
      vtkNew<vtkFloatArray> tArray{};
      std::string name = getTableCoefficientName(noVec, v);
      tArray->SetName(name.c_str());
      tArray->SetNumberOfTuples(allAlphas_.size());
      // Act Alphas
      vtkNew<vtkFloatArray> actArray{};
      std::string actName = "Act" + name;
      actArray->SetName(actName.c_str());
      actArray->SetNumberOfTuples(allAlphas_.size());
      // Scaled Alphas
      vtkNew<vtkFloatArray> tArrayNorm{};
      std::string nameNorm = getTableCoefficientNormName(noVec, v);
      tArrayNorm->SetName(nameNorm.c_str());
      tArrayNorm->SetNumberOfTuples(allAlphas_.size());
      // Act Scaled Alphas
      vtkNew<vtkFloatArray> actArrayNorm{};
      std::string actNameNorm = "Act" + nameNorm;
      actArrayNorm->SetName(actNameNorm.c_str());
      actArrayNorm->SetNumberOfTuples(allAlphas_.size());
      // Fill Arrays
      for(unsigned int i = 0; i < allAlphas_.size(); ++i) {
        tArray->SetTuple1(i, allAlphas_[i][l][v].item<float>());
        actArray->SetTuple1(i, allActAlphas_[i][l][v].item<float>());
        tArrayNorm->SetTuple1(i, allScaledAlphas_[i][l][v].item<float>());
        actArrayNorm->SetTuple1(i, allActScaledAlphas_[i][l][v].item<float>());
      }
      coef_table->AddColumn(tArray);
      coef_table->AddColumn(actArray);
      coef_table->AddColumn(tArrayNorm);
      coef_table->AddColumn(actArrayNorm);
    }
    if(!clusterAsgn_.empty()) {
      vtkNew<vtkIntArray> clusterArray{};
      clusterArray->SetName("ClusterAssignment");
      clusterArray->SetNumberOfTuples(inputTrees.size());
      for(unsigned int i = 0; i < clusterAsgn_.size(); ++i)
        clusterArray->SetTuple1(i, clusterAsgn_[i]);
      coef_table->AddColumn(clusterArray);
    }
    output_coef->SetBlock(l, coef_table);
    std::stringstream ss;
    ss << "Coef" << l;
    output_coef->GetMetaData(l)->Set(vtkCompositeDataSet::NAME(), ss.str());
  }

  /*// Copy Field Data
  // - aggregate input field data
  for(unsigned int b = 0; b < inputTrees[0]->GetNumberOfBlocks(); ++b) {
    vtkNew<vtkFieldData> fd{};
    fd->CopyStructure(inputTrees[0]->GetBlock(b)->GetFieldData());
    fd->SetNumberOfTuples(inputTrees.size());
    for(size_t i = 0; i < inputTrees.size(); ++i) {
      fd->SetTuple(i, 0, inputTrees[i]->GetBlock(b)->GetFieldData());
    }

    // - copy input field data to output row data
    for(int i = 0; i < fd->GetNumberOfArrays(); ++i) {
      auto array = fd->GetAbstractArray(i);
      array->SetName(array->GetName());
      output_coef->AddColumn(array);
    }
  }

  // Field Data Input Parameters
  std::vector<std::string> paramNames;
  getParamNames(paramNames);
  for(auto paramName : paramNames) {
    vtkNew<vtkDoubleArray> array{};
    array->SetName(paramName.c_str());
    array->InsertNextTuple1(getParamValueFromName(paramName));
    output_coef->GetFieldData()->AddArray(array);
  }*/

  // ------------------------------------------
  // --- Geodesics Vectors
  // ------------------------------------------
  output_vectors->SetNumberOfBlocks(2);
  vtkSmartPointer<vtkMultiBlockDataSet> vectors
    = vtkSmartPointer<vtkMultiBlockDataSet>::New();
  vectors->SetNumberOfBlocks(vSTensor_.size());
  vtkSmartPointer<vtkMultiBlockDataSet> vectorsPrime
    = vtkSmartPointer<vtkMultiBlockDataSet>::New();
  vectorsPrime->SetNumberOfBlocks(vSTensor_.size());
  for(unsigned int l = 0; l < vSTensor_.size(); ++l) {
    vtkSmartPointer<vtkTable> vectorsTable = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkTable> vectorsPrimeTable
      = vtkSmartPointer<vtkTable>::New();
    for(unsigned int v = 0; v < vSTensor_[l].sizes()[1]; ++v) {
      // Vs
      vtkNew<vtkFloatArray> vectorArray{};
      std::string name
        = getTableVectorName(vSTensor_[l].sizes()[1], v, 0, 0, false);
      vectorArray->SetName(name.c_str());
      vectorArray->SetNumberOfTuples(vSTensor_[l].sizes()[0]);
      for(unsigned int i = 0; i < vSTensor_[l].sizes()[0]; ++i)
        vectorArray->SetTuple1(i, vSTensor_[l][i][v].item<float>());
      vectorsTable->AddColumn(vectorArray);
      // Vs Prime
      vtkNew<vtkFloatArray> vectorPrimeArray{};
      std::string name2
        = getTableVectorName(vSTensor_[l].sizes()[1], v, 0, 0, false);
      vectorPrimeArray->SetName(name2.c_str());
      vectorPrimeArray->SetNumberOfTuples(vSPrimeTensor_[l].sizes()[0]);
      for(unsigned int i = 0; i < vSPrimeTensor_[l].sizes()[0]; ++i)
        vectorPrimeArray->SetTuple1(i, vSPrimeTensor_[l][i][v].item<float>());
      vectorsPrimeTable->AddColumn(vectorPrimeArray);
    }
    vectors->SetBlock(l, vectorsTable);
    std::stringstream ss;
    ss << "Vectors" << l;
    vectors->GetMetaData(l)->Set(vtkCompositeDataSet::NAME(), ss.str());
    vectorsPrime->SetBlock(l, vectorsPrimeTable);
    ss.str("");
    ss << "VectorsPrime" << l;
    vectorsPrime->GetMetaData(l)->Set(vtkCompositeDataSet::NAME(), ss.str());
  }
  output_vectors->SetBlock(0, vectors);
  output_vectors->SetBlock(1, vectorsPrime);
  num = 0;
  output_vectors->GetMetaData(num)->Set(vtkCompositeDataSet::NAME(), "Vectors");
  num = 1;
  output_vectors->GetMetaData(num)->Set(
    vtkCompositeDataSet::NAME(), "VectorsPrime");
#endif

  return 1;
}
