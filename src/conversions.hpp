#ifndef CONVERSIONS_HPP_
#define CONVERSIONS_HPP_

#include <algorithm>
#include <utility>

#include "dense_matrix.hpp"
#include "band_matrix.hpp"
#include "rowprof_matrix.hpp"
#include "compressed_row_matrix.hpp"

namespace fe { namespace la {

namespace details {
template<
    template<class Sc, class St> class OutputMatrix,
    template<class Sc, class St> class InputMatrix,
    class Scalar,
    class Storage>
struct convert_matrix_f;
} //namespace details

template<
    template<class Sc, class St> class OutputMatrix,
    template<class Sc, class St> class InputMatrix,
    class Scalar,
    class Storage>
OutputMatrix<Scalar, Storage> convert_matrix(InputMatrix<Scalar, Storage> input) {
  return details::convert_matrix_f<OutputMatrix, InputMatrix, Scalar, Storage>()(input);
}

namespace details {
template<
    template<class Sc, class St> class FromMatrix,
    template<class Sc, class St> class ToMatrix,
    class Scalar,
    class Storage>
void assign_elementwise(ToMatrix<Scalar, Storage> & to, FromMatrix<Scalar, Storage> const & from);

template<class Scalar, class Storage>
std::pair<size_t, size_t> calculate_band_count(dense_matrix<Scalar, Storage> const & matrix);
} // namespace details

namespace details {

// Any matrix converted to dense_matrix by element-wise assignment
template<
    template<class Sc, class St> class InputMatrix,
    class Scalar,
    class Storage>
struct convert_matrix_f<dense_matrix, InputMatrix, Scalar, Storage> {
  dense_matrix<Scalar, Storage> operator () (InputMatrix<Scalar, Storage> const & input) {
    dense_matrix<Scalar, Storage> res{input.dim1(), input.dim2()};

    details::assign_elementwise(res, input);

    return res;
  }
};

template<class Scalar, class Storage>
struct convert_matrix_f<band_matrix, dense_matrix, Scalar, Storage> {
  band_matrix<Scalar, Storage> operator () (dense_matrix<Scalar, Storage> const & input) {
    size_t left_bands;
    size_t right_bands;
    std::tie(left_bands, right_bands) = details::calculate_band_count(input);

    std::cout << left_bands << " " << right_bands << std::endl;

    band_matrix<Scalar, Storage> res{input.dim1(), input.dim2(), left_bands, right_bands};

    details::assign_elementwise(res, input);

    return res;
  }
};


template<class Scalar, class Storage>
struct convert_matrix_f<rowprof_matrix, dense_matrix, Scalar, Storage> {
  rowprof_matrix<Scalar, Storage> operator () (dense_matrix<Scalar, Storage> const & input) {
    return rowprof_from_dense(input);
  }
};

template<class Scalar, class Storage>
struct convert_matrix_f<compressed_row_matrix, dense_matrix, Scalar, Storage> {
  compressed_row_matrix<Scalar, Storage> operator () (dense_matrix<Scalar, Storage> const & input) {
    return crmatrix_from_dense(input);
  }
};


template<
    template<class Sc, class St> class FromMatrix,
    template<class Sc, class St> class ToMatrix,
    class Scalar,
    class Storage>
void assign_elementwise(ToMatrix<Scalar, Storage> & to, FromMatrix<Scalar, Storage> const & from) {
  assert(from.dim1() == to.dim1());
  assert(from.dim2() == to.dim2());

  for (size_t i = 0; i < to.dim1(); ++i) {
    for (size_t j = 0; j < to.dim2(); ++j) {
      to(i, j) = from(i, j);
    }
  }
}

// First element of returned pair is left_band_index, second element is right_band_index
template<class Scalar, class Storage>
std::pair<size_t, size_t> calculate_band_count(dense_matrix<Scalar, Storage> const & matrix) {
  size_t left_band = 0;
  size_t right_band = 0;
  // For each row i
  for (size_t i = 0; i < matrix.dim1(); ++i) {
    // Iterate from the first row element to diagonal and set
    //   lid to the index of leftmost non-null element
    size_t lid = i;
    for (size_t j = 0; j < i && j < matrix.dim2(); ++j) {
      if (matrix(i, j) != 0) {
        lid = j;
        break;
      }
    }
    // Iterate from the last row element to diagonal and set
    //   rid to the index of rightmost non-null element
    size_t rid = i;
    for (size_t j = matrix.dim2() - 1; j > i ; --j) {
      if (matrix(i, j) != 0) {
        rid = j;
        break;
      }
    }

    left_band = std::max(left_band, i - lid);
    right_band = std::max(right_band, rid - i);
  }

  return {left_band, right_band};
}
} // namespace details

} } // namespace fe::la

#endif /* CONVERSIONS_HPP_ */
