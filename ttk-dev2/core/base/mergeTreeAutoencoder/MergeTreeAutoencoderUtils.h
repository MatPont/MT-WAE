#pragma once

#include <FTMTree.h>
#include <FTMTree_MT.h>

namespace ttk {

  void adjustNestingScalars(std::vector<float> &scalarsVector,
                            ftm::idNode node,
                            ftm::idNode refNode);

  void createBalancedBDT(std::vector<std::vector<ftm::idNode>> &parents,
                         std::vector<std::vector<ftm::idNode>> &children,
                         std::vector<float> &scalarsVector,
                         std::vector<std::vector<ftm::idNode>> &childrenFinal);

  void printPairs(ftm::MergeTree<float> &mTree, bool useBD = true);

} // namespace ttk
