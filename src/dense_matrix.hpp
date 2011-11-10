#ifndef DENSE_MATRIX_HPP_
#define DENSE_MATRIX_HPP_

#include <cassert>
#include <vector>
#include <utility>
#include <algorithm>
#include <complex>

namespace fe { namespace la {

template<class Scalar, class Storage>
class dense_matrix {
  public:
    typedef Storage storage_t;
    typedef Scalar scalar_t;
    typedef Scalar & reference_t;
    typedef Scalar const & const_reference_t;

    dense_matrix(size_t dim1, size_t dim2)
        : dim1_(dim1), dim2_(dim2), data_(dim1 * dim2) {
    }
    dense_matrix(dense_matrix && other) = default;
    dense_matrix(dense_matrix const & other) = default;
    virtual ~dense_matrix() {};

    storage_t & data() {
      return data_;
    }

    storage_t const & data() const {
      return data_;
    }

    size_t dim1() const {
      return dim1_;
    }

    size_t dim2() const {
      return dim2_;
    }

    reference_t operator () (size_t i, size_t j) {
      return element_at(i, j);
    }

    const_reference_t operator () (size_t i, size_t j) const {
      return const_cast<dense_matrix*>(this)->element_at(i, j);
    }

    dense_matrix & operator += (dense_matrix const & rhs) {
      assert(dim1() == rhs.dim1() && dim2() == rhs.dim2());

      // Add matrices element-wise
      auto const & rdata = rhs.data();

      std::transform(std::begin(data()), std::end(data()), std::begin(rdata), std::begin(data()),
            [](scalar_t left, scalar_t right) {return left + right;});

      return *this;
    }

    dense_matrix & operator -= (dense_matrix const & rhs) {
      assert(dim1() == rhs.dim1() && dim2() == rhs.dim2());

      // Add matrices element-wise
      auto const & rdata = rhs.data();

      std::transform(std::begin(data()), std::end(data()), std::begin(rdata), std::begin(data()),
            [](scalar_t left, scalar_t right) {return left - right;});

      return *this;
    }

    dense_matrix & operator *= (Scalar rhs) {
      std::for_each(std::begin(data()), std::end(data()), [rhs](Scalar & num) { num *= rhs; });
      return *this;
    }
  private:
    reference_t element_at(size_t i, size_t j) {
      assert(i < dim1() && j < dim2());

      return data_[i * dim2() + j];
    }

    size_t dim1_;
    size_t dim2_;
    storage_t data_;
};

namespace details {
template<class Scalar, class Storage>
void mprod_inplace(dense_matrix<Scalar, Storage> const & lhs,
    dense_matrix<Scalar, Storage> const & rhs, dense_matrix<Scalar, Storage> & res);
} // namespace::details;

template<class Scalar, class Storage>
dense_matrix<Scalar, Storage> mprod(dense_matrix<Scalar, Storage> const & lhs,
    dense_matrix<Scalar, Storage> const & rhs) {
  assert(lhs.dim2() == rhs.dim1());

  typedef dense_matrix<Scalar, Storage> mat;

  mat res{lhs.dim1(), rhs.dim2()};

  details::mprod_inplace(lhs, rhs, res);
  return res;
}

namespace details {
template<class Scalar, class Storage>
void mprod_inplace(dense_matrix<Scalar, Storage> const & lhs,
    dense_matrix<Scalar, Storage> const & rhs, dense_matrix<Scalar, Storage> & res) {
  assert(lhs.dim2() == rhs.dim1());
  assert(res.dim1() == lhs.dim1());
  assert(res.dim2() == rhs.dim2());

  size_t dim1 = lhs.dim1();
  size_t dim2 = rhs.dim2();
  size_t dim3 = lhs.dim2();

  for (size_t i = 0; i < dim1; ++i) {
    for (size_t j = 0; j < dim2; ++j) {
      for (size_t k = 0; k < dim3; ++k) {
        res(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
}
} // namespace details


typedef dense_matrix<double, std::vector<double>> real_dmatrix;
typedef dense_matrix<std::complex<double>, std::vector<std::complex<double>>> complex_dmatrix;

} } // namespace fe::la

#endif /* DENSE_MATRIX_HPP_ */