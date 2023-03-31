#include <MergeTreeAutoencoderUtils.h>

void ttk::adjustNestingScalars(std::vector<float> &scalarsVector,
                               ftm::idNode node,
                               ftm::idNode refNode) {
  float birth = scalarsVector[refNode * 2];
  float death = scalarsVector[refNode * 2 + 1];
  auto getSign = [](float v) { return (v > 0 ? 1 : -1); };
  auto getPrecValue = [&getSign](float v, bool opp = false) {
    return v * (1 + (opp ? -1 : 1) * getSign(v) * 1e-6);
  };
  // Shift scalars
  if(scalarsVector[node * 2 + 1] > getPrecValue(death, true)) {
    float diff = scalarsVector[node * 2 + 1] - getPrecValue(death, true);
    scalarsVector[node * 2] -= diff;
    scalarsVector[node * 2 + 1] -= diff;
  } else if(scalarsVector[node * 2] < getPrecValue(birth)) {
    float diff = getPrecValue(birth) - scalarsVector[node * 2];
    scalarsVector[node * 2] += getPrecValue(diff);
    scalarsVector[node * 2 + 1] += getPrecValue(diff);
  }
  // Cut scalars
  if(scalarsVector[node * 2] < getPrecValue(birth))
    scalarsVector[node * 2] = getPrecValue(birth);
  if(scalarsVector[node * 2 + 1] > getPrecValue(death, true))
    scalarsVector[node * 2 + 1] = getPrecValue(death, true);
}

void ttk::createBalancedBDT(
  std::vector<std::vector<ftm::idNode>> &parents,
  std::vector<std::vector<ftm::idNode>> &children,
  std::vector<float> &scalarsVector,
  std::vector<std::vector<ftm::idNode>> &childrenFinal) {
  // BEGIN TESTING
  std::stringstream ssBug;
  for(unsigned int i = 0; i < children.size(); ++i) {
    ssBug << i << " : ";
    for(auto &e : children[i])
      ssBug << e << " ";
    ssBug << std::endl;
  }
  // END TESTING

  // ----- Some variables
  unsigned int noNodes = scalarsVector.size() / 2;
  childrenFinal.resize(noNodes);
  int mtLevel = ceil(log(noNodes * 2) / log(2)) + 1;
  int bdtLevel = mtLevel - 1;
  int noDim = bdtLevel;

  // ----- Get node levels
  // std::cout << "// ----- Get node levels" << std::endl;
  std::vector<int> nodeLevels(noNodes, -1);
  std::queue<ftm::idNode> queueLevels;
  std::vector<int> noChildDone(noNodes, 0);
  for(unsigned int i = 0; i < children.size(); ++i) {
    if(children[i].size() == 0) {
      queueLevels.emplace(i);
      nodeLevels[i] = 1;
    }
  }
  while(!queueLevels.empty()) {
    ftm::idNode node = queueLevels.front();
    queueLevels.pop();
    for(auto &parent : parents[node]) {
      ++noChildDone[parent];
      nodeLevels[parent] = std::max(nodeLevels[parent], nodeLevels[node] + 1);
      if(noChildDone[parent] >= (int)children[parent].size())
        queueLevels.emplace(parent);
    }
  }

  // ----- Filter lambda
  /*auto filterIndexes
    = [&nodeLevels](std::vector<unsigned int> &indexes, int dim,
                    std::vector<unsigned int> &indexesOut) {
        indexesOut.clear();
        std::copy_if(
          indexes.begin(), indexes.end(), std::back_inserter(indexesOut),
          [&nodeLevels, &dim](unsigned int i) { return nodeLevels[i] > dim; });
      };*/

  // ----- Sort heuristic lambda
  // TODO should sortIndexes be called each time a struct is found/created?
  auto sortChildren = [&parents, &scalarsVector, &noNodes](
                        ftm::idNode nodeOrigin, std::vector<bool> &nodeDone,
                        std::vector<std::vector<ftm::idNode>> &childrenT) {
    double refPers = scalarsVector[1] - scalarsVector[0];
    auto getRemaining = [&nodeDone](std::vector<ftm::idNode> &vec) {
      unsigned int remaining = 0;
      for(auto &e : vec)
        remaining += (not nodeDone[e]);
      return remaining;
    };
    std::vector<unsigned int> parentsRemaining(noNodes, 0),
      childrenRemaining(noNodes, 0);
    for(auto &child : childrenT[nodeOrigin]) {
      parentsRemaining[child] = getRemaining(parents[child]);
      childrenRemaining[child] = getRemaining(childrenT[child]);
    }
    std::sort(
      childrenT[nodeOrigin].begin(), childrenT[nodeOrigin].end(),
      [&](ftm::idNode nodeI, ftm::idNode nodeJ) {
        /*if(nodeI * 2 + 1 >= scalarsVector.size())
          std::cout << "bug" << std::endl;*/
        double persI = scalarsVector[nodeI * 2 + 1] - scalarsVector[nodeI * 2];
        double persJ = scalarsVector[nodeJ * 2 + 1] - scalarsVector[nodeJ * 2];
        return parentsRemaining[nodeI] + childrenRemaining[nodeI]
                 - persI / refPers * noNodes
               < parentsRemaining[nodeJ] + childrenRemaining[nodeJ]
                   - persJ / refPers * noNodes;
      });
  };

  // ----- Greedy approach to find balanced BDT structures
  const auto findStructGivenDim =
    [&children, &noNodes, &nodeLevels](
      ftm::idNode _nodeOrigin, int _dimToFound, bool _searchMaxDim,
      std::vector<bool> &_nodeDone, std::vector<bool> &_dimFound,
      std::vector<std::vector<ftm::idNode>> &_childrenFinalOut) {
      // --- Recursive lambda
      auto findStructGivenDimImpl =
        [&children, &noNodes, &nodeLevels](
          ftm::idNode nodeOrigin, int dimToFound, bool searchMaxDim,
          std::vector<bool> &nodeDone, std::vector<bool> &dimFound,
          std::vector<std::vector<ftm::idNode>> &childrenFinalOut,
          auto &findStructGivenDimRef) mutable {
          childrenFinalOut.resize(noNodes);
          // - Find structures
          int dim = (searchMaxDim ? dimToFound - 1 : 0);
          unsigned int i = 0;
          //
          auto searchMaxDimReset = [&i, &dim, &nodeDone]() {
            --dim;
            i = 0;
            unsigned int noDone = 0;
            for(auto done : nodeDone)
              if(done)
                ++noDone;
            return noDone == nodeDone.size() - 1; // -1 for root
          };
          while(i < children[nodeOrigin].size()) {
            auto child = children[nodeOrigin][i];
            // Skip if child was already processed
            if(nodeDone[child]) {
              // If we have processed all children while searching for max
              // dim then restart at the beginning to find a lower dim
              if(searchMaxDim and i == children[nodeOrigin].size() - 1) {
                if(searchMaxDimReset())
                  break;
              } else
                ++i;
              continue;
            }
            if(dim == 0) {
              // Base case
              childrenFinalOut[nodeOrigin].emplace_back(child);
              nodeDone[child] = true;
              dimFound[0] = true;
              // TODO return value for searchMaxDim
              if(dimToFound <= 1 or searchMaxDim)
                return true;
              ++dim;
            } else {
              // General case
              std::vector<std::vector<ftm::idNode>> childrenFinalDim;
              std::vector<bool> nodeDoneDim;
              std::vector<bool> dimFoundDim(dim);
              bool found = false;
              if(nodeLevels[child] > dim) {
                nodeDoneDim = nodeDone;
                found = findStructGivenDimRef(child, dim, false, nodeDoneDim,
                                              dimFoundDim, childrenFinalDim,
                                              findStructGivenDimRef);
              }
              if(found) {
                dimFound[dim] = true;
                childrenFinalOut[nodeOrigin].emplace_back(child);
                for(unsigned int j = 0; j < childrenFinalDim.size(); ++j)
                  for(auto &e : childrenFinalDim[j])
                    childrenFinalOut[j].emplace_back(e);
                nodeDone[child] = true;
                for(unsigned int j = 0; j < nodeDoneDim.size(); ++j)
                  nodeDone[j] = nodeDone[j] || nodeDoneDim[j];
                // Return if it is the last dim to found
                if(dim == dimToFound - 1 and not searchMaxDim)
                  return true;
                // Reset index if we search for the maximum dim
                if(searchMaxDim) {
                  if(searchMaxDimReset())
                    break;
                } else {
                  ++dim;
                }
                continue;
              } else if(searchMaxDim and i == children[nodeOrigin].size() - 1) {
                // If we have processed all children while searching for max
                // dim then restart at the beginning to find a lower dim
                if(searchMaxDimReset())
                  break;
                continue;
              }
            }
            ++i;
          }
          return false;
        };
      return findStructGivenDimImpl(_nodeOrigin, _dimToFound, _searchMaxDim,
                                    _nodeDone, _dimFound, _childrenFinalOut,
                                    findStructGivenDimImpl);
    };
  std::vector<bool> dimFound(noDim - 1, false);
  std::vector<bool> nodeDone(noNodes, false);
  for(unsigned int i = 0; i < children.size(); ++i)
    sortChildren(i, nodeDone, children);
  Timer t_find;
  ftm::idNode startNode = 0;
  findStructGivenDim(startNode, noDim, true, nodeDone, dimFound, childrenFinal);
  if(t_find.getElapsedTime() > 10)
    std::cout << "findStructGivenDim time = " << t_find.getElapsedTime()
              << std::endl;

  // BEGIN TESTING
  ssBug << std::endl;
  for(unsigned int i = 0; i < childrenFinal.size(); ++i) {
    ssBug << i << " : ";
    for(auto &e : childrenFinal[i])
      ssBug << e << " ";
    ssBug << std::endl;
  }
  // END TESTING

  // ----- Greedy approach to create non found structures
  const auto createStructGivenDim =
    [&children, &noNodes, &findStructGivenDim, &nodeLevels, &ssBug](
      int _nodeOrigin, int _dimToCreate, std::vector<bool> &_nodeDone,
      ftm::idNode &_structOrigin, std::vector<float> &_scalarsVectorOut,
      std::vector<std::vector<ftm::idNode>> &_childrenFinalOut) {
      // --- Recursive lambda
      auto createStructGivenDimImpl =
        [&children, &noNodes, &findStructGivenDim, &nodeLevels, &ssBug](
          int nodeOrigin, int dimToCreate, std::vector<bool> &nodeDoneImpl,
          ftm::idNode &structOrigin, std::vector<float> &scalarsVectorOut,
          std::vector<std::vector<ftm::idNode>> &childrenFinalOut,
          auto &createStructGivenDimRef) mutable {
          // Deduction of auto lambda type
          if(false)
            return;
          // - Find structures of lower dimension
          int dimToFound = dimToCreate - 1;
          // BEGIN TESTING
          ssBug << "search for 2 dimToFound = " << dimToFound << std::endl;
          // END TESTING
          std::vector<std::vector<std::vector<ftm::idNode>>> childrenFinalT(2);
          std::array<ftm::idNode, 2> structOrigins;
          for(unsigned int n = 0; n < 2; ++n) {
            bool found = false;
            for(unsigned int i = 0; i < children[nodeOrigin].size(); ++i) {
              auto child = children[nodeOrigin][i];
              if(nodeDoneImpl[child])
                continue;
              if(dimToFound != 0) {
                if(nodeLevels[child] > dimToFound) {
                  std::vector<bool> dimFoundT(dimToFound, false);
                  childrenFinalT[n].clear();
                  childrenFinalT[n].resize(noNodes);
                  std::vector<bool> nodeDoneImplFind = nodeDoneImpl;
                  found = findStructGivenDim(child, dimToFound, false,
                                             nodeDoneImplFind, dimFoundT,
                                             childrenFinalT[n]);
                }
              } else
                found = true;
              if(found) {
                // BEGIN TESTING
                ssBug << "- found " << std::endl;
                ssBug << "  - (f) structOrigin = " << child << std::endl;
                // END TESTING
                structOrigins[n] = child;
                nodeDoneImpl[child] = true;
                for(unsigned int j = 0; j < childrenFinalT[n].size(); ++j) {
                  // BEGIN TESTING
                  if(!childrenFinalT[n][j].empty())
                    ssBug << "  - " << j << " : ";
                  // END TESTING
                  for(auto &e : childrenFinalT[n][j]) {
                    // BEGIN TESTING
                    ssBug << e << " ";
                    // END TESTING
                    childrenFinalOut[j].emplace_back(e);
                    nodeDoneImpl[e] = true;
                  }
                  // BEGIN TESTING
                  if(!childrenFinalT[n][j].empty())
                    ssBug << std::endl;
                  // END TESTING
                }
                break;
              }
            } // end for children[nodeOrigin]
            if(not found) {
              if(dimToFound <= 0) {
                structOrigins[n] = std::numeric_limits<ftm::idNode>::max();
                continue;
              }
              // BEGIN TESTING
              ssBug << "- createStructGivenDimRef" << std::endl;
              // END TESTING
              childrenFinalT[n].clear();
              childrenFinalT[n].resize(noNodes);
              createStructGivenDimRef(
                nodeOrigin, dimToFound, nodeDoneImpl, structOrigins[n],
                scalarsVectorOut, childrenFinalT[n], createStructGivenDimRef);
              // BEGIN TESTING
              ssBug << "  - (c) structOrigin = " << structOrigins[n]
                    << std::endl;
              // END TESTING
              for(unsigned int j = 0; j < childrenFinalT[n].size(); ++j) {
                // BEGIN TESTING
                if(!childrenFinalT[n][j].empty())
                  ssBug << "  - " << j << " : ";
                // END TESTING
                for(auto &e : childrenFinalT[n][j]) {
                  // BEGIN TESTING
                  ssBug << e << " ";
                  // END TESTING
                  if(e == structOrigins[n])
                    continue;
                  childrenFinalOut[j].emplace_back(e);
                }
                // BEGIN TESTING
                if(!childrenFinalT[n][j].empty())
                  ssBug << std::endl;
                // END TESTING
              }
            }
          } // end for n
          // - Combine both structures
          if(structOrigins[0] == std::numeric_limits<ftm::idNode>::max()
             and structOrigins[1] == std::numeric_limits<ftm::idNode>::max()) {
            structOrigin = std::numeric_limits<ftm::idNode>::max();
            return;
          }
          bool firstIsParent = true;
          if(structOrigins[0] == std::numeric_limits<ftm::idNode>::max())
            firstIsParent = false;
          else if(structOrigins[1] == std::numeric_limits<ftm::idNode>::max())
            firstIsParent = true;
          else if(scalarsVectorOut[structOrigins[1] * 2 + 1]
                    - scalarsVectorOut[structOrigins[1] * 2]
                  > scalarsVectorOut[structOrigins[0] * 2 + 1]
                      - scalarsVectorOut[structOrigins[0] * 2])
            firstIsParent = false;
          structOrigin = (firstIsParent ? structOrigins[0] : structOrigins[1]);
          ftm::idNode modOrigin
            = (firstIsParent ? structOrigins[1] : structOrigins[0]);
          childrenFinalOut[nodeOrigin].emplace_back(structOrigin);
          if(modOrigin != std::numeric_limits<ftm::idNode>::max()) {
            // BEGIN TESTING
            ssBug << "combine 2 dimToFound = " << dimToFound << " structures"
                  << std::endl;
            ssBug << "Update " << structOrigin << std::endl;
            // END TESTING
            childrenFinalOut[structOrigin].emplace_back(modOrigin);
            std::queue<std::array<ftm::idNode, 2>> queue;
            queue.emplace(std::array<ftm::idNode, 2>{modOrigin, structOrigin});
            // BEGIN TESTING
            ssBug << "- scalars is " << scalarsVectorOut[structOrigin * 2]
                  << " _ " << scalarsVectorOut[structOrigin * 2 + 1] << " ("
                  << (scalarsVectorOut[structOrigin * 2 + 1]
                      - scalarsVectorOut[structOrigin * 2])
                  << ")" << std::endl;
            // END TESTING
            while(!queue.empty()) {
              auto &nodeAndParent = queue.front();
              ftm::idNode node = nodeAndParent[0];
              ftm::idNode parent = nodeAndParent[1];
              // BEGIN TESTING
              ssBug << "- process " << node << std::endl;
              ssBug << "  - scalars was " << scalarsVectorOut[node * 2] << " _ "
                    << scalarsVectorOut[node * 2 + 1] << " ("
                    << (scalarsVectorOut[node * 2 + 1]
                        - scalarsVectorOut[node * 2])
                    << ")" << std::endl;
              // END TESTING
              queue.pop();
              adjustNestingScalars(scalarsVectorOut, node, parent);
              // BEGIN TESTING
              ssBug << "  - scalars is  " << scalarsVectorOut[node * 2] << " _ "
                    << scalarsVectorOut[node * 2 + 1] << " ("
                    << (scalarsVectorOut[node * 2 + 1]
                        - scalarsVectorOut[node * 2])
                    << ")" << std::endl;
              // END TESTING
              // Push children
              for(auto &child : childrenFinalOut[node])
                queue.emplace(std::array<ftm::idNode, 2>{child, node});
            }
          }
          return;
        };
      return createStructGivenDimImpl(
        _nodeOrigin, _dimToCreate, _nodeDone, _structOrigin, _scalarsVectorOut,
        _childrenFinalOut, createStructGivenDimImpl);
    };
  for(unsigned int i = 0; i < children.size(); ++i)
    sortChildren(i, nodeDone, children);
  Timer t_create;
  for(unsigned int i = 0; i < dimFound.size(); ++i) {
    if(dimFound[i])
      continue;
    ftm::idNode structOrigin;
    createStructGivenDim(
      startNode, i, nodeDone, structOrigin, scalarsVector, childrenFinal);
    // BEGIN TESTING
    ssBug << "dim = " << i << std::endl;
    for(unsigned int j = 0; j < childrenFinal.size(); ++j) {
      ssBug << j << " : ";
      for(auto &e : childrenFinal[j])
        ssBug << e << " ";
      ssBug << std::endl;
    }
    // END TESTING
  }
  if(t_create.getElapsedTime() > 10)
    std::cout << "createStructGivenDim time = " << t_create.getElapsedTime()
              << std::endl;

  // ----- Verify
  // BEGIN TESTING
  bool foundBug = false;
  // Verify that all nodes have been processed
  for(unsigned int i = 1; i < nodeDone.size(); ++i) {
    if(not nodeDone[i]) {
      ssBug << "Oops ! node " << i << " has not be processed" << std::endl;
      foundBug = true;
    }
  }
  // Verify merge tree validity
  std::vector<unsigned int> noParents(childrenFinal.size(), 0);
  for(unsigned int i = 0; i < childrenFinal.size(); ++i) {
    for(auto &e : childrenFinal[i]) {
      ++noParents[e];
      if(scalarsVector[e * 2] < scalarsVector[i * 2]
         or scalarsVector[e * 2 + 1] > scalarsVector[i * 2 + 1]) {
        ssBug << "Oops ! " << e << " does not respect nesting condition with "
              << i;
        ssBug << " (" << scalarsVector[e * 2] << ", "
              << scalarsVector[e * 2 + 1] << ") _ (" << scalarsVector[i * 2]
              << ", " << scalarsVector[i * 2 + 1] << ")" << std::endl;
        if(scalarsVector[e * 2] < scalarsVector[i * 2])
          ssBug << scalarsVector[e * 2] << " < " << scalarsVector[i * 2]
                << std::endl;
        if(scalarsVector[e * 2 + 1] > scalarsVector[i * 2 + 1])
          ssBug << scalarsVector[e * 2 + 1] << " > " << scalarsVector[i * 2 + 1]
                << std::endl;
        foundBug = true;
      }
    }
  }
  // Verify number of parents
  for(unsigned int i = 0; i < noParents.size(); ++i) {
    if(noParents[i] > 1) {
      ssBug << "Oops ! " << i << " has " << noParents[i] << " parents"
            << std::endl;
      foundBug = true;
    }
  }
  // Print
  if(foundBug) {
    std::cout << ssBug.str();
    std::cin.get();
  }
  // END TESTING
}

void ttk::printPairs(ftm::MergeTree<float> &mTree, bool useBD) {
  std::stringstream ss;
  if(mTree.tree.getRealNumberOfNodes() != 0)
    ss = mTree.tree.template printPairsFromTree<float>(useBD);
  else {
    std::vector<bool> nodeDone(mTree.tree.getNumberOfNodes(), false);
    for(unsigned int i = 0; i < mTree.tree.getNumberOfNodes(); ++i) {
      if(nodeDone[i])
        continue;
      std::tuple<ftm::idNode, ftm::idNode, float> pair
        = std::make_tuple(i, mTree.tree.getNode(i)->getOrigin(),
                          mTree.tree.getNodePersistence<float>(i));
      ss << std::get<0>(pair) << " ("
         << mTree.tree.getValue<float>(std::get<0>(pair)) << ") _ ";
      ss << std::get<1>(pair) << " ("
         << mTree.tree.getValue<float>(std::get<1>(pair)) << ") _ ";
      ss << std::get<2>(pair) << std::endl;
      nodeDone[i] = true;
      nodeDone[mTree.tree.getNode(i)->getOrigin()] = true;
    }
  }
  ss << std::endl;
  std::cout << ss.str();
}
