#pragma once

#include <cassert>
#include "dense_vector.hpp"

namespace fe { namespace la {
  /**
   * Solves the matrix system Ax = b inplace.
   * Matrix A must be a result of LU decomposition.
   *
   * @tparam Matrix The class of a system matrix.
   * @tparam Vector The class of a right-hand side vector.
   *
   * @param A Matrix of a system.
   * @param b Right-hand side of the equation.
   */
  template<class Matrix, class Vector>
  void solve_lu_inplace(Matrix const & A, Vector & b) {
    assert(A.dim2() == b.dim());

    // First solve Ly = b
    for (int i = 0; i < b.dim(); ++i) {
      for (int j = i + 1; j < b.dim(); ++j) {
        b(j) -= b(i) * A(j, i);
      }
    }

    // Now solve Ux = y
    for (int i = b.dim() - 1; i >= 0; --i) {
      b(i) /= A(i ,i);
      for (int j = i - 1; j >= 0; --j) {
        b(j) -= b(i) * A(j, i);
      }
    }
  }
} } // namespace fe::la
