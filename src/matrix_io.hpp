#pragma once

#include <istream>
#include <ostream>
#include <vector>
#include <complex>

#include "dense_matrix.hpp"

namespace fe { namespace la { namespace io {
  template<class Matrix>
  void save_to_stream(Matrix const & matrix, std::ostream & out) {
    out << matrix.dim1() << " " << matrix.dim2() << " ";
    for (size_t i = 0; i < matrix.dim1(); ++i) {
      for (size_t j = 0; j < matrix.dim2(); ++j) {
        out << matrix(i, j) << " ";
      }
    }
  }

  template<class Scalar, class Storage>
  dense_matrix<Scalar, Storage> load_from_stream(std::istream & in) {
    size_t dim1;
    size_t dim2;
    in >> dim1 >> dim2;

    dense_matrix<Scalar, Storage> res{dim1, dim2};
    for (size_t i = 0; i < dim1; ++i) {
      for (size_t j = 0; j < dim2; ++j) {
        in >> res(i, j);
      }
    }
    return res;
  }

  dense_matrix_real load_real_matrix_from_stream(std::istream & in) {
    return load_from_stream<double, std::vector<double>>(in);
  }

  dense_matrix_complex load_complex_matrix_from_stream(std::istream & in) {
    typedef std::complex<double> scalar_t;
    return load_from_stream<scalar_t, std::vector<scalar_t>>(in);
  }

} } } // namespace fe::la::io
