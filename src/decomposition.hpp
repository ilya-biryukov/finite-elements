#ifndef DECOMPOSITION_HPP_
#define DECOMPOSITION_HPP_

#include <cassert>

#include "details/sparse_element_proxy.hpp"

namespace fe { namespace la {
	
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
      for (size_t i = k + 1; i < n; ++i) {
        mat(i, k) /= mat(k, k);
      }

      for (size_t i = k + 1; i < n; ++i) {
        for (size_t j = k + 1; j < n; ++j) {
          mat(i, j) = mat(i, j) - mat(i, k) * mat(k, j);
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
      for (size_t i = k + 1; i < n; ++i) {
        mat(i, k) /= mat(k, k);
        mat(k, i) /= mat(k, k);
      }

      for (size_t i = k + 1; i < n; ++i) {
        for (size_t j = k + 1; j < n; ++j) {
          mat(i, j) = mat(i, j) - mat(i, k) * mat(k, k) * mat(k, j);
        }
      }
    }
  }
} } // namespace fe::la


#endif /* DECOMPOSITION_HPP_ */
