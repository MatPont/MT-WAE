/// \ingroup base
/// \class ttk::AssignmentSinkhorn
/// \author Mathieu Pont (mathieu.pont@lip6.fr)
///
/// Sinkhorn algorithm for Balanced and Unbalanced Assignement Problem
///
/// For the unbalanced problem:
///   The cost matrix in input has a size of (n + 1) x (m + 1)
///   - n is the number of jobs, m the number of workers
///   - the nth row contains the cost of not assigning workers
///   - the mth column is the same but with jobs
///   - the last cell (costMatrix[n][m]) is not used

#ifndef _ASSIGNMENTSINKHORN_H
#define _ASSIGNMENTSINKHORN_H

#include <Debug.h>

#include <math.h> /* exp */

#include "AssignmentSolver.h"

namespace ttk {

  template <class dataType>
  class AssignmentSinkhorn : virtual public Debug,
                             public AssignmentSolver<dataType> {

  public:
    AssignmentSinkhorn() = default;

    ~AssignmentSinkhorn() = default;

    double getLambda() {
      return lambda;
    }
    void setLambda(double l) {
      lambda = l;
    }

    int run(std::vector<MatchingType> &matchings) {

      // this->printTableVector(this->costMatrix);
      // Make balanced cost matrix
      if(not this->balancedAssignment)
        this->makeBalancedMatrix(this->costMatrix);
      // this->printTableVector(this->costMatrix);

      // K, Kbar in size*size
      //

      int size = this->costMatrix.size();

      std::vector<double> r(size, 1.0 / size), c(size, 1.0 / size);

      std::vector<std::vector<double>> K(size, std::vector<double>(size)),
        Kbar(size, std::vector<double>(size));
      for(unsigned int i = 0; i < this->costMatrix.size(); ++i)
        for(unsigned int j = 0; j < this->costMatrix[0].size(); ++j) {
          K[i][j] = exp(-lambda * this->costMatrix[i][j]);
          if(K[i][j] != K[i][j])
            printErr("K[i][j] != K[i][j]");
          Kbar[i][j] = K[i][j] * 1.0 / r[i];
        }
      std::vector<std::vector<double>> Kprime;
      transpose(K, Kprime);

      std::vector<double> u(r.size(), 1.0 / r.size()), v;

      std::cout << "===============" << std::endl;

      bool converged = false;
      while(not converged) {
        std::vector<double> KprimeDotU, divisor, oldU = u;
        matrixVectorDot(Kprime, u, KprimeDotU);
        vectorDivide(c, KprimeDotU, v);
        matrixVectorDot(Kbar, v, divisor);
        std::vector<double> ones(u.size(), 1.0);
        vectorDivide(ones, divisor, u);

        // Convergence
        /*for(unsigned int i = 0; i < u.size(); ++i)
          std::cout << u[i] << " __ " << oldU[i] << std::endl;*/
        double dist = l2Distance(u, oldU);
        converged = (dist < 1e-3);
        std::cout << "dist = " << dist << std::endl;
      }
      std::vector<double> KprimeDotU;
      matrixVectorDot(Kprime, u, KprimeDotU);
      vectorDivide(c, KprimeDotU, v);

      // Get transporation matrix
      std::vector<std::vector<double>> P, diagU, diagV, temp;
      diag(u, diagU);
      diag(v, diagV);
      matrixDot(diagU, K, temp);
      matrixDot(temp, diagV, P);

      /*std::cout << "P" << std::endl;
      for(unsigned int i = 0; i < P.size(); ++i) {
        for(unsigned int j = 0; j < P[i].size(); ++j)
          std::cout << P[i][j] << " ";
        std::cout << std::endl;
      }*/

      // Create matching
      matchingsFromTransportationMatrix(P, matchings);

      return 0;
    }

    void
      matchingsFromTransportationMatrix(std::vector<std::vector<double>> &P,
                                        std::vector<MatchingType> &matchings) {
      matchings.clear();

      std::vector<double> P_flatten(P.size() * P[0].size());
      for(unsigned int i = 0; i < P.size(); ++i)
        for(unsigned int j = 0; j < P[0].size(); ++j)
          P_flatten[i * P[0].size() + j] = P[i][j];

      std::vector<int> idx(P_flatten.size());
      iota(idx.begin(), idx.end(), 0);
      std::stable_sort(
        idx.begin(), idx.end(), [&P_flatten](size_t i1, size_t i2) {
          return P_flatten[i1] > P_flatten[i2];
        });

      std::vector<bool> firstDone(P.size(), false),
        secondDone(P[0].size(), false);
      unsigned int done = 0;
      for(unsigned int i = 0; i < idx.size(); ++i) {
        int i0 = idx[i] / P[0].size();
        int i1 = idx[i] % P[0].size();

        if(not firstDone[i0] and not secondDone[i1]) {
          firstDone[i0] = true;
          secondDone[i1] = true;
          MatchingType matching{i0, i1, this->costMatrix[i0][i1]};
          if(this->balancedAssignment
             or (not this->balancedAssignment
                 and not(i0 >= this->rowSize - 1 and i1 >= this->colSize - 1)))
            matchings.push_back(matching);
          ++done;
        }

        if(done == P.size())
          break;
      }
    }

    // ------------------------------------------------------------------------
    // Matrix Utils
    // ------------------------------------------------------------------------
    double l2Distance(std::vector<double> &v, std::vector<double> &v2) {
      double distance = 0;
      for(unsigned int i = 0; i < v.size(); ++i)
        distance += std::pow(v[i] - v2[i], 2);
      return sqrt(distance);
    }

    void transpose(std::vector<std::vector<double>> &m,
                   std::vector<std::vector<double>> &newM) {
      newM = std::vector<std::vector<double>>(
        m[0].size(), std::vector<double>(m.size()));
      for(unsigned int i = 0; i < m.size(); ++i)
        for(unsigned int j = 0; j < m[0].size(); ++j)
          newM[j][i] = m[i][j];
    }

    void matrixDot(std::vector<std::vector<double>> &m1,
                   std::vector<std::vector<double>> &m2,
                   std::vector<std::vector<double>> &newM) {
      newM = std::vector<std::vector<double>>(
        m1.size(), std::vector<double>(m2[0].size(), 0.0));
      for(unsigned int i = 0; i < newM.size(); ++i)
        for(unsigned int j = 0; j < newM[i].size(); ++j)
          for(unsigned int k = 0; k < m1[i].size(); ++k)
            newM[i][j] += m1[i][k] * m2[k][j];
    }

    void matrixVectorDot(std::vector<std::vector<double>> &m,
                         std::vector<double> &v,
                         std::vector<double> &newV) {
      std::vector<std::vector<double>> newM, vMat(v.size());
      for(unsigned int i = 0; i < v.size(); ++i)
        vMat[i] = std::vector<double>(1, v[i]);

      matrixDot(m, vMat, newM);

      newV = std::vector<double>(newM.size());
      for(unsigned int i = 0; i < newM.size(); ++i)
        newV[i] = newM[i][0];
    }

    void vectorDivide(std::vector<double> &v,
                      std::vector<double> &v2,
                      std::vector<double> &newV) {
      newV = std::vector<double>(v.size());
      for(unsigned int i = 0; i < v.size(); ++i) {
        if(v2[i] == 0.0) {
          std::cout << v2[i] << std::endl;
          printErr("divide by zero");
          int temp;
          std::cin >> temp;
        } else
          newV[i] = v[i] / v2[i];
      }
    }

    void diag(std::vector<double> &v, std::vector<std::vector<double>> &diagM) {
      diagM = std::vector<std::vector<double>>(
        v.size(), std::vector<double>(v.size()));
      for(unsigned int i = 0; i < v.size(); ++i)
        diagM[i][i] = v[i];
    }

  private:
    double lambda = 100.0;

  }; // AssignmentSinkhorn Class

} // namespace ttk

#endif
