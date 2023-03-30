/// \ingroup base
/// \class FTMTreePPUtils
/// \author Mathieu Pont (mathieu.pont@lip6.fr)
/// \date 2021.
///
/// Utils function for manipulating FTMTree class

#pragma once

#include <FTMTree.h>
#include <FTMTreePP.h>

namespace ttk {
  namespace ftm {

    template <class dataType>
    void getPersistencePairs(
      FTMTree_MT *tree,
      std::vector<std::tuple<SimplexId, SimplexId, dataType>> &pairs) {
      FTMTreePP pairsCompute;
      pairsCompute.setCustomTree(tree);
      pairsCompute.computePersistencePairs<dataType>(
        pairs, tree->isJoinTree<dataType>());
    }

    template <class dataType>
    std::vector<std::tuple<SimplexId, SimplexId, dataType>>
      computePersistencePairs(FTMTree_MT *tree) {
      std::vector<std::tuple<SimplexId, SimplexId, dataType>> pairs;
      getPersistencePairs<dataType>(tree, pairs);
      for(auto pair : pairs) {
        if(tree->getNode(std::get<0>(pair))->getOrigin() < std::get<0>(pair)
           and tree->getNode(std::get<0>(pair))->getOrigin() >= 0) {

          if(tree->getNode(std::get<0>(pair))->getOrigin()
             != std::get<1>(pair)) {
            std::stringstream ss;
            std::cout << "tree->getValue<dataType>(std::get<0>(pair)) = "
                      << tree->getValue<dataType>(std::get<0>(pair))
                      << std::endl;
            tree->printNodeSS(std::get<0>(pair), ss);
            std::cout << ss.str();
            std::cout << "tree->getValue<dataType>(std::get<1>(pair)) = "
                      << tree->getValue<dataType>(std::get<1>(pair))
                      << std::endl;
            ss.str("");
            tree->printNodeSS(std::get<1>(pair), ss);
            std::cout << ss.str();
            std::cout << "tree->getValue<dataType>(tree->getNode(std::get<0>("
                         "pair))->getOrigin()) = "
                      << tree->getValue<dataType>(
                           tree->getNode(std::get<0>(pair))->getOrigin())
                      << std::endl;
            ss.str("");
            tree->printNodeSS(
              tree->getNode(std::get<0>(pair))->getOrigin(), ss);
            std::cout << ss.str();
            std::cout << "computePersistencePairs weird if" << std::endl
                      << std::endl;
          }

          tree->getNode(tree->getNode(std::get<0>(pair))->getOrigin())
            ->setOrigin(std::get<1>(pair));
        }

        tree->getNode(std::get<0>(pair))->setOrigin(std::get<1>(pair));
        tree->getNode(std::get<1>(pair))->setOrigin(std::get<0>(pair));
      }
      return pairs;
    }

  } // namespace ftm
} // namespace ttk
