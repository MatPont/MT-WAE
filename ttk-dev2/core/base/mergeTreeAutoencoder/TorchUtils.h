/// \ingroup base
/// \author TODO
/// \date TODO.
///
/// \brief TODO

#pragma once

#include <Debug.h>
#include <FTMTreeUtils.h>
#include <FTMTree_MT.h>
#include <Geometry.h>
#include <MergeTreeUtils.h>

#ifdef TTK_ENABLE_TORCH
#include <torch/torch.h>
#endif

namespace ttk {

  namespace TorchUtils {

#ifdef TTK_ENABLE_TORCH
    void copyTensor(torch::Tensor &a, torch::Tensor &b);

    template <typename dataType>
    struct TorchMergeTree {
      ftm::MergeTree<dataType> mTree;
      torch::Tensor tensor;
      std::vector<unsigned int> nodeCorr;
      std::vector<ftm::idNode> parentsOri;
    };

    template <class dataType>
    void copyTorchMergeTree(TorchMergeTree<dataType> &tmTree,
                            TorchMergeTree<dataType> &out) {
      out.mTree = ftm::copyMergeTree<dataType>(tmTree.mTree);
      copyTensor(tmTree.tensor, out.tensor);
      out.nodeCorr = tmTree.nodeCorr;
      out.parentsOri = tmTree.parentsOri;
    }

    template <class dataType>
    void mergeTreeToTorchTensor(ftm::MergeTree<dataType> &mTree,
                                torch::Tensor &tensor,
                                std::vector<unsigned int> &nodeCorr,
                                bool normalize) {
      nodeCorr.clear();
      nodeCorr.assign(mTree.tree.getNumberOfNodes(),
                      std::numeric_limits<unsigned int>::max());
      std::vector<dataType> scalars;
      std::queue<ftm::idNode> queue;
      queue.emplace(mTree.tree.getRoot());
      unsigned int cpt = 0;
      while(!queue.empty()) {
        ftm::idNode node = queue.front();
        queue.pop();

        auto birthDeath = ttk::getParametrizedBirthDeath<dataType>(
          &(mTree.tree), node, normalize);
        auto birth = std::get<0>(birthDeath);
        auto death = std::get<1>(birthDeath);
        scalars.emplace_back(birth);
        scalars.emplace_back(death);

        nodeCorr[node] = cpt;
        ++cpt;

        std::vector<ftm::idNode> children;
        mTree.tree.getChildren(node, children);
        for(auto child : children)
          queue.emplace(child);
      }
      tensor = torch::tensor(scalars).reshape({-1, 1});
    }

    template <class dataType>
    void mergeTreeToTorchTensor(ftm::MergeTree<dataType> &mTree,
                                torch::Tensor &tensor,
                                bool normalize) {
      std::vector<unsigned int> nodeCorr;
      mergeTreeToTorchTensor<dataType>(mTree, tensor, nodeCorr, normalize);
    }

    template <class dataType>
    void getParentsVector(ftm::MergeTree<dataType> &mTree,
                          std::vector<ftm::idNode> &parents) {
      parents.resize(mTree.tree.getNumberOfNodes());
      std::fill(parents.begin(), parents.end(),
                std::numeric_limits<ftm::idNode>::max());
      std::queue<ftm::idNode> queue;
      queue.emplace(mTree.tree.getRoot());
      while(!queue.empty()) {
        ftm::idNode node = queue.front();
        queue.pop();
        if(!mTree.tree.isRoot(node))
          parents[node] = mTree.tree.getParentSafe(node);
        std::vector<ftm::idNode> children;
        mTree.tree.getChildren(node, children);
        for(auto child : children)
          queue.emplace(child);
      }
    }

    template <class dataType>
    void mergeTreeToTorchTree(ftm::MergeTree<dataType> &mTree,
                              TorchMergeTree<dataType> &out,
                              bool normalize) {
      out.mTree = copyMergeTree(mTree);
      getParentsVector(out.mTree, out.parentsOri);
      mergeTreeToTorchTensor<dataType>(
        out.mTree, out.tensor, out.nodeCorr, normalize);
    }

    template <class dataType>
    void mergeTreesToTorchTrees(std::vector<ftm::MergeTree<dataType>> &mTrees,
                                std::vector<TorchMergeTree<dataType>> &out,
                                bool normalize) {
      out.resize(mTrees.size());
      for(unsigned int i = 0; i < mTrees.size(); ++i)
        mergeTreeToTorchTree<dataType>(mTrees[i], out[i], normalize);
    }

    template <class dataType>
    bool torchTensorToMergeTree(TorchMergeTree<dataType> &tmt,
                                bool normalized,
                                ftm::MergeTree<dataType> &mTreeOut) {
      std::vector<unsigned int> &nodeCorr = tmt.nodeCorr;
      torch::Tensor &tensor = tmt.tensor;
      std::vector<ftm::idNode> &parentsOri = tmt.parentsOri;

      mTreeOut = ttk::ftm::copyMergeTree<dataType>(tmt.mTree);
      if(parentsOri.empty())
        std::cout << "[torchTensorToMergeTree] parentsOri.empty()" << std::endl;
      for(unsigned int i = 0; i < parentsOri.size(); ++i)
        if(parentsOri[i] < mTreeOut.tree.getNumberOfNodes()
           and mTreeOut.tree.isRoot(i))
          mTreeOut.tree.setParent(i, parentsOri[i]);
      ftm::idNode root = mTreeOut.tree.getRoot();
      if(root >= mTreeOut.tree.getNumberOfNodes())
        return true;

      bool isJT = tmt.mTree.tree.template isJoinTree<dataType>();
      std::vector<dataType> tensorVec(
        tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
      std::vector<dataType> scalarsVector;
      ttk::ftm::getTreeScalars<dataType>(mTreeOut, scalarsVector);
      std::queue<ftm::idNode> queue;
      queue.emplace(root);
      while(!queue.empty()) {
        ftm::idNode node = queue.front();
        queue.pop();

        auto birthValue = tensorVec[nodeCorr[node] * 2];
        auto deathValue = tensorVec[nodeCorr[node] * 2 + 1];
        if(normalized and !mTreeOut.tree.isRoot(node)) {
          ftm::idNode nodeParent = mTreeOut.tree.getParentSafe(node);
          /*auto birthDeathParentNode
            = mTreeOut.tree.template getBirthDeathNode<dataType>(nodeParent);
          auto birthParentNode = std::get<0>(birthDeathParentNode);
          auto deathParentNode = std::get<1>(birthDeathParentNode);*/
          ftm::idNode nodeParentOrigin
            = mTreeOut.tree.getNode(nodeParent)->getOrigin();
          ftm::idNode birthParentNode = (isJT ? nodeParentOrigin : nodeParent);
          ftm::idNode deathParentNode = (isJT ? nodeParent : nodeParentOrigin);
          auto birthParent = scalarsVector[birthParentNode];
          auto deathParent = scalarsVector[deathParentNode];
          birthValue = birthValue * (deathParent - birthParent) + birthParent;
          deathValue = deathValue * (deathParent - birthParent) + birthParent;
        }
        ftm::idNode nodeOrigin = mTreeOut.tree.getNode(node)->getOrigin();
        scalarsVector[node] = (isJT ? deathValue : birthValue);
        scalarsVector[nodeOrigin] = (isJT ? birthValue : deathValue);

        std::vector<ftm::idNode> children;
        mTreeOut.tree.getChildren(node, children);
        for(auto &child : children)
          queue.emplace(child);
      }
      ftm::setTreeScalars<dataType>(mTreeOut, scalarsVector);
      return false;
    }

    template <class dataType>
    void geodesicVectorToTorchTensor(ftm::MergeTree<dataType> &mTree,
                                     std::vector<std::vector<double>> &v,
                                     torch::Tensor &tensor) {
      std::vector<double> v_flatten;
      std::queue<ftm::idNode> queue;
      queue.emplace(mTree.tree.getRoot());
      while(!queue.empty()) {
        ftm::idNode node = queue.front();
        queue.pop();

        v_flatten.emplace_back(v[node][0]);
        v_flatten.emplace_back(v[node][1]);

        std::vector<ftm::idNode> children;
        mTree.tree.getChildren(node, children);
        for(auto child : children)
          queue.emplace(child);
      }
      tensor = torch::tensor(v_flatten);
      tensor = tensor.reshape({-1, 1});
    }

    template <class dataType>
    void geodesicVectorsToTorchTensor(
      ftm::MergeTree<dataType> &mTree,
      std::vector<std::vector<std::vector<double>>> &vS,
      torch::Tensor &tensor) {
      std::vector<torch::Tensor> allTensors;
      for(auto &v : vS) {
        torch::Tensor t;
        geodesicVectorToTorchTensor(mTree, v, t);
        allTensors.emplace_back(t);
      }
      tensor = torch::cat(allTensors, 1);
    }

    template <class dataType>
    void getTensorMatching(
      TorchMergeTree<dataType> &a,
      TorchMergeTree<dataType> &b,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
      std::vector<int> &tensorMatching) {
      tensorMatching.clear();
      tensorMatching.resize((int)(a.tensor.sizes()[0] / 2), -1);
      for(auto &match : matching) {
        auto match1 = std::get<0>(match);
        auto match2 = std::get<1>(match);
        if(a.nodeCorr[match1] < tensorMatching.size())
          tensorMatching[a.nodeCorr[match1]] = b.nodeCorr[match2];
      }
    }

    template <class dataType>
    void getInverseTensorMatching(
      TorchMergeTree<dataType> &a,
      TorchMergeTree<dataType> &b,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matching,
      std::vector<int> &tensorMatching) {
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> invMatching(
        matching.size());
      for(unsigned int i = 0; i < matching.size(); ++i)
        invMatching[i]
          = std::make_tuple(std::get<1>(matching[i]), std::get<0>(matching[i]),
                            std::get<2>(matching[i]));
      getTensorMatching(b, a, invMatching, tensorMatching);
    }
#endif

  } // namespace TorchUtils

} // namespace ttk
