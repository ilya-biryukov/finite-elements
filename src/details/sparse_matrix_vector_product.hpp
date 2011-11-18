#ifndef SPARSE_MATRIX_VECTOR_PRODUCT_HPP_
#define SPARSE_MATRIX_VECTOR_PRODUCT_HPP_

#include <cassert>
#include "../dense_vector.hpp"

namespace fe { namespace la { namespace details {
  template<
      class Scalar,
      class Storage,
      template<class Sc, class St> class SparseMatrix>
  dense_vector<Scalar, Storage> mv_sparse_prod(SparseMatrix<Scalar, Storage> const & matrix,
        dense_vector<Scalar, Storage> const & vector) {
    assert(matrix.dim2() == vector.dim1());

    dense_vector<Scalar, Storage> res{vector.dim()};
    for (size_t i = 0; i < res.dim(); ++i) {
      {
        auto iter_end = matrix.nnrow_cend(i);
        for (auto iter = matrix.nnrow_cbegin(i); iter != iter_end; ++iter) {
          res(i) += vector(iter.index()) * (*iter);
        }
      }
    }

    return res;
  }

  template<
      class Scalar,
      class Storage,
      template<class Sc, class St> class SparseMatrix>
  dense_vector<Scalar, Storage> mv_sparse_prod(dense_vector<Scalar, Storage> const & vector,
      SparseMatrix<Scalar, Storage> const & matrix) {
    assert(vector.dim2() == matrix.dim1());

    dense_vector<Scalar, Storage> res{vector.dim(), vector_type::ROW_VECTOR};
    for (size_t i = 0; i < res.dim(); ++i) {
      {
        auto iter_end = matrix.nncol_cend(i);
        for (auto iter = matrix.nncol_cbegin(i); iter != iter_end; ++iter) {
          res(i) += vector(iter.index()) * (*iter);
        }
      }
    }

    return res;
  }
} } } // namespace fe::la::details


#endif /* SPARSE_MATRIX_VECTOR_PRODUCT_HPP_ */
