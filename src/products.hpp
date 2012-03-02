#pragma once

#include "dense_vector.hpp"

namespace fe { namespace la {
  namespace details {
    // General implementation, sparse matrices have explicit template specializations
    template<
        template<class Sc, class St> class Matrix
        ,class Scalar
        ,class Storage>
    struct mat_rowvec_prod_impl_f {
      dense_vector<Scalar, Storage> operator()(dense_vector<Scalar, Storage> lhs
          ,Matrix<Scalar, Storage> rhs) const {
        assert(lhs.dim2() == rhs.dim1());

        typedef dense_vector<Scalar, Storage> vec;
        typedef dense_matrix<Scalar, Storage> mat;

        vec res{rhs.dim2(), vector_type::ROW_VECTOR};

        details::mprod_inplace(lhs, rhs, res);

        return res;
      }
    };

    template<
        template<class Sc, class St> class Matrix
        ,class Scalar
        ,class Storage>
    struct mat_colvec_prod_impl_f;

    template<
        template<class Sc, class St> class Matrix
        ,class Scalar
        ,class Storage>
    struct mat_colvec_prod_impl_f {
      dense_vector<Scalar, Storage> operator()(Matrix<Scalar, Storage> lhs
          , dense_vector<Scalar, Storage> rhs) const {
        assert(lhs.dim2() == rhs.dim1());

        typedef dense_vector<Scalar, Storage> vec;
        typedef dense_matrix<Scalar, Storage> mat;

        vec res{lhs.dim1()};

        details::mprod_inplace(lhs, rhs, res);

        return res;
      }
    };
  } // namespace details

  template<template<class Sc, class St> class Matrix, class Scalar, class Storage>
  dense_vector<Scalar, Storage> mvprod(dense_vector<Scalar, Storage> lhs
      ,Matrix<Scalar, Storage> rhs) {
    return details::mat_rowvec_prod_impl_f<Matrix, Scalar, Storage>{}(lhs, rhs);
  }

  template<template<class Sc, class St> class Matrix, class Scalar, class Storage>
  dense_vector<Scalar, Storage> mvprod(Matrix<Scalar, Storage> lhs
      , dense_vector<Scalar, Storage> rhs) {
    return details::mat_colvec_prod_impl_f<Matrix, Scalar, Storage>{}(lhs, rhs);
  }

} } // namespace fe::la
