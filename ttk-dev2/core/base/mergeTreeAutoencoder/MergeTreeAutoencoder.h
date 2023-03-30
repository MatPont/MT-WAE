/// \ingroup base
/// \class ttk::MergeTreeAutoencoder
/// \author XXX <XXX@XXX>
/// \date 2023.
///
/// This module defines the %MergeTreeAutoencoder class that TODO
///
/// \b Related \b publication: \n
/// TODO
///

#pragma once

// ttk common includes
#include <Debug.h>
#include <Geometry.h>
#include <MergeTreeAxesAlgorithmBase.h>
#include <TorchUtils.h>

#ifdef TTK_ENABLE_TORCH
#include <torch/torch.h>
#endif

namespace ttk {

  /**
   * The MergeTreeAutoencoder class provides methods to compute TODO
   */
  class MergeTreeAutoencoder : virtual public Debug,
                               public MergeTreeAxesAlgorithmBase {

  protected:
    bool doCompute_;
    bool hasComputedOnce_ = false;

    // Model hyper-parameters;
    int encoderNoLayers_, optimizer_;
    unsigned int inputNumberOfGeodesics_, noInit_;
    double inputOriginPrimeSizePercent_, latentSpaceOriginPrimeSizePercent_;
    double gradientStepSize_, beta1_, beta2_, batchSize_,
      reconstructionLossWeight_, trackingLossWeight_, metricLossWeight_,
      clusteringLossWeight_, baseRecLoss_, baseRecLoss2_;
    bool fullSymmetricAE_, activate_, euclideanVectorsInit_,
      activateOutputInit_, customLossSpace_, customLossActivate_,
      customLossDynamicWeight_, scaleLayerAfterLatent_,
      initOriginPrimeStructByCopy_;
    unsigned int minIteration_, maxIteration_, activationFunction_,
      iterationGap_;
    std::vector<unsigned int> clusterAsgn_;
    float clusteringLossTemp_ = 10;
    bool createOutput_;
    std::vector<std::vector<float>> distanceMatrix_, customAlphas_;

    double trackingLossInitRandomness_;
    bool trackingLossDecoding_;

#ifdef TTK_ENABLE_TORCH
    // Model optimized parameters
    std::vector<torch::Tensor> vSTensor_, vSPrimeTensor_, vS2Tensor_,
      vS2PrimeTensor_, latentCentroids_;
    std::vector<TorchUtils::TorchMergeTree<float>> origins_, originsPrime_,
      origins2_, origins2Prime_;

    std::vector<TorchUtils::TorchMergeTree<float>> originsCopy_,
      originsPrimeCopy_;

    // Filled by the algorithm
    std::vector<std::vector<torch::Tensor>> allAlphas_, allScaledAlphas_,
      allActAlphas_, allActScaledAlphas_;
    std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> recs_, recs2_;
    std::vector<TorchUtils::TorchMergeTree<float>> customRecs_;
#endif

    // Filled by the algorithm
    unsigned noLayers_;
    std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
      baryMatchings_L0_, baryMatchings2_L0_;
    std::vector<double> inputToBaryDistances_L0_;

    // Tracking matchings
    std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
      originsMatchings_, reconstMatchings_, customMatchings_;
    std::vector<
      std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>>
      dataMatchings_;
    std::vector<std::vector<double>> branchesCorrelationMatrix_,
      persCorrelationMatrix_;

    // Testing
    double t_allVectorCopy_time_ = 0.0;
    std::vector<unsigned int> originsNoZeroGrad_, originsPrimeNoZeroGrad_,
      vSNoZeroGrad_, vSPrimeNoZeroGrad_, origins2NoZeroGrad_,
      origins2PrimeNoZeroGrad_, vS2NoZeroGrad_, vS2PrimeNoZeroGrad_;
    bool outputInit_ = true;
#ifdef TTK_ENABLE_TORCH
    std::vector<TorchUtils::TorchMergeTree<float>> initOrigins_,
      initOriginsPrime_, initRecs_;
#endif
    float bigValuesThreshold_ = 0;

  public:
    MergeTreeAutoencoder();

#ifdef TTK_ENABLE_TORCH
    //  -----------------------------------------------------------------------
    //  --- Init
    //  -----------------------------------------------------------------------
    void initOutputBasisTreeStructure(
      TorchUtils::TorchMergeTree<float> &originPrime,
      bool isJT,
      TorchUtils::TorchMergeTree<float> &baseOrigin);

    void initOutputBasis(unsigned int l, unsigned int dim, unsigned int dim2);

    void initOutputBasisVectors(unsigned int l,
                                torch::Tensor &w,
                                torch::Tensor &w2);

    void initOutputBasisVectors(unsigned int l,
                                unsigned int dim,
                                unsigned int dim2);

    void initInputBasisOrigin(
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
        &baryMatchings2);

    void initInputBasisVectors(
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
      torch::Tensor &vS2Tensor);

    void initClusteringLossParameters();

    float initParameters(std::vector<TorchUtils::TorchMergeTree<float>> &trees,
                         std::vector<TorchUtils::TorchMergeTree<float>> &trees2,
                         bool computeReconstructionError = false);

    void initStep(std::vector<TorchUtils::TorchMergeTree<float>> &trees,
                  std::vector<TorchUtils::TorchMergeTree<float>> &trees2);

    //  -----------------------------------------------------------------------
    //  --- Interpolation
    //  -----------------------------------------------------------------------
    void interpolationDiagonalProjection(
      TorchUtils::TorchMergeTree<float> &interpolationTensor);

    void interpolationNestingProjection(
      TorchUtils::TorchMergeTree<float> &interpolation);

    void
      interpolationProjection(TorchUtils::TorchMergeTree<float> &interpolation);

    void
      getMultiInterpolation(TorchUtils::TorchMergeTree<float> &origin,
                            torch::Tensor &vS,
                            torch::Tensor &alphas,
                            TorchUtils::TorchMergeTree<float> &interpolation);

    //  -----------------------------------------------------------------------
    //  --- Forward
    //  -----------------------------------------------------------------------
    void getAlphasOptimizationTensors(
      TorchUtils::TorchMergeTree<float> &tree,
      TorchUtils::TorchMergeTree<float> &origin,
      torch::Tensor &vSTensor,
      TorchUtils::TorchMergeTree<float> &interpolated,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
      torch::Tensor &reorderedTreeTensor,
      torch::Tensor &deltaOrigin,
      torch::Tensor &deltaA,
      torch::Tensor &originTensor_f,
      torch::Tensor &vSTensor_f);

    void computeAlphas(
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
      torch::Tensor &alphasOut);

    float assignmentOneData(
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
      bool isCalled = false);

    float assignmentOneData(TorchUtils::TorchMergeTree<float> &tree,
                            TorchUtils::TorchMergeTree<float> &origin,
                            torch::Tensor &vSTensor,
                            TorchUtils::TorchMergeTree<float> &tree2,
                            TorchUtils::TorchMergeTree<float> &origin2,
                            torch::Tensor &vS2Tensor,
                            unsigned int k,
                            torch::Tensor &alphasInit,
                            torch::Tensor &bestAlphas,
                            bool isCalled = false);

    torch::Tensor activation(torch::Tensor &in);

    void
      outputBasisReconstruction(TorchUtils::TorchMergeTree<float> &originPrime,
                                torch::Tensor &vSPrimeTensor,
                                TorchUtils::TorchMergeTree<float> &origin2Prime,
                                torch::Tensor &vS2PrimeTensor,
                                torch::Tensor &alphas,
                                TorchUtils::TorchMergeTree<float> &out,
                                TorchUtils::TorchMergeTree<float> &out2,
                                bool activate = true);

    bool forwardOneLayer(TorchUtils::TorchMergeTree<float> &tree,
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
                         torch::Tensor &bestAlphas);

    bool forwardOneData(TorchUtils::TorchMergeTree<float> &tree,
                        TorchUtils::TorchMergeTree<float> &tree2,
                        unsigned int treeIndex,
                        unsigned int k,
                        std::vector<torch::Tensor> &alphasInit,
                        TorchUtils::TorchMergeTree<float> &out,
                        TorchUtils::TorchMergeTree<float> &out2,
                        std::vector<torch::Tensor> &dataAlphas,
                        std::vector<TorchUtils::TorchMergeTree<float>> &outs,
                        std::vector<TorchUtils::TorchMergeTree<float>> &outs2);

    bool forwardStep(
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
      float &loss);

    bool forwardStep(std::vector<TorchUtils::TorchMergeTree<float>> &trees,
                     std::vector<TorchUtils::TorchMergeTree<float>> &trees2,
                     std::vector<unsigned int> &indexes,
                     unsigned int k,
                     std::vector<std::vector<torch::Tensor>> &allAlphasInit,
                     std::vector<TorchUtils::TorchMergeTree<float>> &outs,
                     std::vector<TorchUtils::TorchMergeTree<float>> &outs2,
                     std::vector<std::vector<torch::Tensor>> &bestAlphas);

    //  -----------------------------------------------------------------------
    //  --- Backward
    //  -----------------------------------------------------------------------
    bool backwardStep(
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
      torch::Tensor &trackingLoss);

    //  -----------------------------------------------------------------------
    //  --- Projection
    //  -----------------------------------------------------------------------
    void projectionStep();

    //  -----------------------------------------------------------------------
    //  --- Convergence
    //  -----------------------------------------------------------------------
    float computeOneLoss(
      TorchUtils::TorchMergeTree<float> &tree,
      TorchUtils::TorchMergeTree<float> &out,
      TorchUtils::TorchMergeTree<float> &tree2,
      TorchUtils::TorchMergeTree<float> &out2,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching2);

    float computeLoss(
      std::vector<TorchUtils::TorchMergeTree<float>> &trees,
      std::vector<TorchUtils::TorchMergeTree<float>> &outs,
      std::vector<TorchUtils::TorchMergeTree<float>> &trees2,
      std::vector<TorchUtils::TorchMergeTree<float>> &outs2,
      std::vector<unsigned int> &indexes,
      std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
        &matchings,
      std::vector<std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>>
        &matchings2);

    bool isBestLoss(float loss, float &minLoss, unsigned int &cptBlocked);

    bool convergenceStep(float loss,
                         float &oldLoss,
                         float &minLoss,
                         unsigned int &cptBlocked);

    //  -----------------------------------------------------------------------
    //  --- Main Functions
    //  -----------------------------------------------------------------------
    void fit(std::vector<ftm::MergeTree<float>> &trees,
             std::vector<ftm::MergeTree<float>> &trees2);

    void execute(std::vector<ftm::MergeTree<float>> &trees,
                 std::vector<ftm::MergeTree<float>> &trees2);

    //  -----------------------------------------------------------------------
    //  --- Custom Losses
    //  -----------------------------------------------------------------------
    double getCustomLossDynamicWeight(double recLoss, double &baseLoss);

    void getDistanceMatrix(std::vector<TorchUtils::TorchMergeTree<float>> &tmts,
                           std::vector<std::vector<float>> &distanceMatrix,
                           bool useDoubleInput = false,
                           bool isFirstInput = true);

    void
      getDistanceMatrix(std::vector<TorchUtils::TorchMergeTree<float>> &tmts,
                        std::vector<TorchUtils::TorchMergeTree<float>> &tmts2,
                        std::vector<std::vector<float>> &distanceMatrix);

    void getDifferentiableDistanceFromMatchings(
      TorchUtils::TorchMergeTree<float> &tree1,
      TorchUtils::TorchMergeTree<float> &tree2,
      TorchUtils::TorchMergeTree<float> &tree1_2,
      TorchUtils::TorchMergeTree<float> &tree2_2,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matchings,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matchings2,
      torch::Tensor &tensorDist,
      bool doSqrt);

    void getDifferentiableDistance(TorchUtils::TorchMergeTree<float> &tree1,
                                   TorchUtils::TorchMergeTree<float> &tree2,
                                   TorchUtils::TorchMergeTree<float> &tree1_2,
                                   TorchUtils::TorchMergeTree<float> &tree2_2,
                                   torch::Tensor &tensorDist,
                                   bool isCalled,
                                   bool doSqrt);

    void getDifferentiableDistance(TorchUtils::TorchMergeTree<float> &tree1,
                                   TorchUtils::TorchMergeTree<float> &tree2,
                                   torch::Tensor &tensorDist,
                                   bool isCalled,
                                   bool doSqrt);

    void getDifferentiableDistanceMatrix(
      std::vector<TorchUtils::TorchMergeTree<float> *> &trees,
      std::vector<TorchUtils::TorchMergeTree<float> *> &trees2,
      std::vector<std::vector<torch::Tensor>> &outDistMat);

    void getAlphasTensor(std::vector<std::vector<torch::Tensor>> &alphas,
                         std::vector<unsigned int> &indexes,
                         unsigned int layerIndex,
                         torch::Tensor &alphasOut);

    void computeMetricLoss(
      std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &layersOuts,
      std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &layersOuts2,
      std::vector<std::vector<torch::Tensor>> alphas,
      std::vector<std::vector<float>> &baseDistanceMatrix,
      std::vector<unsigned int> &indexes,
      torch::Tensor &metricLoss);

    void computeClusteringLoss(std::vector<std::vector<torch::Tensor>> &alphas,
                               std::vector<unsigned int> &indexes,
                               torch::Tensor &clusteringLoss,
                               torch::Tensor &asgn);

    void computeTrackingLoss(torch::Tensor &trackingLoss);

    //  ---------------------------------------------------------------------------
    //  --- End Functions
    //  ---------------------------------------------------------------------------
    void createCustomRecs();

    void computeTrackingInformation();

    void
      createScaledAlphas(std::vector<std::vector<torch::Tensor>> &alphas,
                         std::vector<torch::Tensor> &vSTensor,
                         std::vector<std::vector<torch::Tensor>> &scaledAlphas);

    void createScaledAlphas();

    void createActivatedAlphas();

    void fixTreePrecisionScalars(ftm::MergeTree<float> &mTree);

    //  -----------------------------------------------------------------------
    //  --- Utils
    //  -----------------------------------------------------------------------
    void copyParams(
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
      std::vector<std::vector<torch::Tensor>> &dstAlphas);

    void copyParams(
      std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &src,
      std::vector<std::vector<TorchUtils::TorchMergeTree<float>>> &dst);

    void getDeltaProjTensor(torch::Tensor &diagTensor,
                            torch::Tensor &deltaProjTensor);

    void dataReorderingGivenMatching(TorchUtils::TorchMergeTree<float> &tree,
                                     TorchUtils::TorchMergeTree<float> &tree2,
                                     torch::Tensor &tree1ProjIndexer,
                                     torch::Tensor &tree2ReorderingIndexes,
                                     torch::Tensor &tree2ReorderedTensor,
                                     torch::Tensor &tree2DeltaProjTensor,
                                     torch::Tensor &tree1ReorderedTensor,
                                     torch::Tensor &tree2ProjIndexer,
                                     bool doubleReordering = true);

    void dataReorderingGivenMatching(TorchUtils::TorchMergeTree<float> &tree,
                                     TorchUtils::TorchMergeTree<float> &tree2,
                                     torch::Tensor &tree1ProjIndexer,
                                     torch::Tensor &tree2ReorderingIndexes,
                                     torch::Tensor &tree2ReorderedTensor,
                                     torch::Tensor &tree2DeltaProjTensor);

    void dataReorderingGivenMatching(
      TorchUtils::TorchMergeTree<float> &tree,
      TorchUtils::TorchMergeTree<float> &tree2,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
      torch::Tensor &tree1ReorderedTensor,
      torch::Tensor &tree2ReorderedTensor,
      bool doubleReordering = true);

    void dataReorderingGivenMatching(
      TorchUtils::TorchMergeTree<float> &tree,
      TorchUtils::TorchMergeTree<float> &tree2,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
      torch::Tensor &tree2ReorderedTensor);

    void meanBirthShift(torch::Tensor &diagTensor,
                        torch::Tensor &diagBaseTensor);

    void meanBirthMaxPersShift(torch::Tensor &tensor,
                               torch::Tensor &baseTensor);

    void belowDiagonalPointsShift(torch::Tensor &tensor,
                                  torch::Tensor &backupTensor);

    void normalizeVectors(torch::Tensor &originTensor,
                          torch::Tensor &vectorsTensor);

    void normalizeVectors(TorchUtils::TorchMergeTree<float> &origin,
                          std::vector<std::vector<double>> &vectors);

    unsigned int getLatentLayerIndex();

    bool isThereMissingPairs(TorchUtils::TorchMergeTree<float> &interpolation);

    //  -----------------------------------------------------------------------
    //  --- Testing
    //  -----------------------------------------------------------------------
    void makeExponentialExample(std::vector<ftm::MergeTree<float>> &trees,
                                std::vector<ftm::MergeTree<float>> &trees2);

    bool isTreeHasBigValues(ftm::MergeTree<float> &mTree,
                            float threshold = 10000);
#endif
  }; // MergeTreeAutoencoder class

} // namespace ttk
