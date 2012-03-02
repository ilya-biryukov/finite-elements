#pragma once

#include "dense_matrix.hpp"

#include <vector>

namespace fe { namespace la {

enum class vector_type {
  COLUMN_VECTOR,
  ROW_VECTOR
};

template<class Scalar, class Storage>
class dense_vector : public dense_matrix<Scalar, Storage> {
  private:
    typedef dense_matrix<Scalar, Storage> base;
  public:
    // Sadly, I have to copy this
    typedef typename base::storage_t storage_t;
    typedef typename base::scalar_t scalar_t;
    typedef typename base::reference_t reference_t;
    typedef typename base::const_reference_t const_reference_t;
    using base::operator ();

    dense_vector() = delete;
    dense_vector(dense_vector const &) = default;
    dense_vector(dense_vector &&) = default;


    explicit dense_vector(size_t dim, vector_type type = vector_type::COLUMN_VECTOR)
        : dense_matrix<Scalar, Storage>(
            type == vector_type::COLUMN_VECTOR ? dim : 1,
            type == vector_type::COLUMN_VECTOR ? 1 : dim) {
    }


    reference_t operator() (size_t i) {
      assert(i < std::max(this->dim1(), this->dim2()));

      return this->data()[i];
    }

    const_reference_t operator () (size_t i) const {
      return (*const_cast<dense_vector *>(this))(i);
    }

    size_t dim() const {
      return std::max(this->dim1(), this->dim2());
    }
};

typedef dense_vector<double, std::vector<double>> dense_vector_real;
typedef dense_vector<
    std::complex<double>, std::vector<std::complex<double>>>
  dense_vector_complex;

// Multiply matrix by column vector
template<class Scalar, class Storage>
struct mvprod_f {
  dense_vector<Scalar, Storage> mvprod(dense_matrix<Scalar, Storage> const & lhs, dense_vector<Scalar, Storage> const & rhs) {
    assert(lhs.dim2() == rhs.dim1());

    typedef dense_vector<Scalar, Storage> vec;
    typedef dense_matrix<Scalar, Storage> mat;

    vec res{lhs.dim1()};

    details::mprod_inplace(lhs, rhs, res);

    return res;
  }
};

// Multiply row vector by matrix
template<class Scalar, class Storage>
dense_vector<Scalar, Storage> mvprod(dense_vector<Scalar, Storage> const & lhs, dense_matrix<Scalar, Storage> const & rhs) {
  assert(lhs.dim2() == rhs.dim1());

  typedef dense_vector<Scalar, Storage> vec;
  typedef dense_matrix<Scalar, Storage> mat;

  vec res{rhs.dim2(), vector_type::ROW_VECTOR};

  details::mprod_inplace(lhs, rhs, res);

  return res;
}

} } // namespace fe::la
