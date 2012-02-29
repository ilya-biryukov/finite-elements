#ifndef DECOMPOSITION_HPP_
#define DECOMPOSITION_HPP_

#include <cassert>

#include "details/sparse_element_proxy.hpp"

namespace fe { namespace la {

  template<class Mat>
  void sparse_lu_decomposition(Mat & mat) {
	// benefit of using this method instead of lu_decomposition - 
	// it makes O(n^2 * m) instead of O(n^3), where m - number of nun-zero elements
    assert(mat.dim1() == mat.dim2());

    size_t n = mat.dim1();
    for (size_t k = 0; k < n; ++k) {
      auto mat_kk = mat(k, k);
      for (size_t i = k + 1; i < n; ++i) {
        mat(i, k) /= mat_kk;
      }

      for (size_t i = k + 1; i < n; ++i) {
        for (auto col_it = mat.nnrow_begin(i); col_it != mat.nnrow_end(i); ++col_it) {
		  // run on the i row  until we reach k+1 element 	
          if (col_it.index() < k + 1) {
            continue;
          }
		  // assign new value on the row using pointer of it's position	
          *col_it -= mat(i, k) * mat(k, col_it.index());
        }
      }
    }
  }


  template<class Mat>
  void lu_decomposition(Mat & mat) {
    // for k = 1 to n
    //    u_kk = a_kk
    //    for i = k+1 to n
    //      l_ik = a_ik / u_kk
    //      u_ki = a_ki
    //    for i = k + 1 to n
    //      for j = k + 1 to n
    //        a_ij = a_ij - l_ik u_ki
    assert(mat.dim1() == mat.dim2());

    size_t n = mat.dim1();
    for (size_t k = 0; k < n; ++k) {
      auto mat_kk = mat(k, k);
      for (size_t i = k + 1; i < n; ++i) {
        mat(i, k) /= mat_kk;
      }

      for (size_t i = k + 1; i < n; ++i) {
        for (size_t j = k + 1; j < n; ++j) {
          mat(i, j) -= mat(i, k) * mat(k, j);
        }
      }
    }
  }

  template<class Mat>
  void sparse_ldu_decomposition(Mat & mat) {
    // benefit of this method is the same as of sparse_lu_decomposition
	// it makes O(n^2 * m) operation instead of O(n^3) as in ldu_decomposition
    assert(mat.dim1() == mat.dim2());

    size_t n = mat.dim1();
    for (size_t k = 0; k < n; ++k) {
      auto mat_kk = mat(k, k);
      for (size_t i = k + 1; i < n; ++i) {
        mat(i, k) /= mat_kk;
        mat(k, i) /= mat_kk;
      }

      for (size_t i = k + 1; i < n; ++i) {
        for (auto col_it = mat.nnrow_begin(i); col_it != mat.nnrow_end(i); ++col_it) {
          if (col_it.index() < k + 1) {
            continue;
          }	
          *col_it -= mat(i, k) * mat_kk * mat(k, col_it.index());
        }
      }
    }
  }

  template<class Mat>
  void ldu_decomposition(Mat & mat) {
    // for k = 1 to n
    //    u_kk = 1
    //    d_kk = a_kk
    //    for i = k+1 to n
    //      l_ik = a_ik / d_kk
    //      u_ki = a_ki / d_kk
    //    for i = k + 1 to n
    //      for j = k + 1 to n
    //        a_ij = a_ij - l_ik * d_kk * u_ki
    assert(mat.dim1() == mat.dim2());

    size_t n = mat.dim1();
    for (size_t k = 0; k < n; ++k) {
      auto mat_kk = mat(k, k);
      for (size_t i = k + 1; i < n; ++i) {
        mat(i, k) /= mat_kk;
        mat(k, i) /= mat_kk;
      }

      for (size_t i = k + 1; i < n; ++i) {
        for (size_t j = k + 1; j < n; ++j) {
          mat(i, j) -= mat(i, k) * mat_kk * mat(k, j);
        }
      }
    }
  }
} } // namespace fe::la


#endif /* DECOMPOSITION_HPP_ */
