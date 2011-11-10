#ifndef BAND_MATRIX_HPP_
#define BAND_MATRIX_HPP_

#include <cmath>
#include <limits>
#include <complex>

#include "dense_matrix.hpp"

namespace fe { namespace la {

namespace details {
template<class Scalar>
class band_matrix_proxy;
} // namespace details

template<class Scalar, class Storage>
class band_matrix {
  private:
    typedef details::band_matrix_proxy<Scalar> proxy_t;
    typedef dense_matrix<Scalar, Storage> storage_matrix_t;
  public:
    typedef Scalar scalar_t;
    typedef Storage storage_t;
    typedef proxy_t reference_t;
    typedef Scalar const_reference_t;

    band_matrix(size_t dim1, size_t dim2, size_t bands_count)
        : dim1_(dim1), dim2_(dim2), m_(bands_count),
          storage_(dim1, bands_count) {
      assert(bands_count % 2 == 1);
    }

    band_matrix(band_matrix const &) = default;
    band_matrix(band_matrix &&) = default;
    ~band_matrix() = default;

    size_t dim1() const {
      return dim1_;
    }

    size_t dim2() const {
      return dim2_;
    }

    reference_t operator () (size_t i, size_t j) {
      assert(i < dim1());
      assert(j < dim2());
      // Technically, casting from int to size_t might be a narrowing cast, right???
      // Anyway, this should work as expected
      assert(i < static_cast<size_t>(std::numeric_limits<int>::max()));
      assert(j < static_cast<size_t>(std::numeric_limits<int>::max()));

      int inti = static_cast<int>(i);
      int intj = static_cast<int>(j);

      if (std::abs(inti - intj) > m_ /2 ) {
        return proxy_t();
      } else {
        auto & element = storage_(i, j - i + m_/2);
        return proxy_t(&element);
      }
    }

    const_reference_t operator () (size_t i, size_t j) const {
      assert(i < dim1());
      assert(j < dim2());

      auto p = (*const_cast<band_matrix*>(this))(i, j);
      return p; // p will be cast to scalar_t
    }

    storage_t & data() {
      return storage_.data();
    }

    storage_t const & data() const {
      return storage_.data();
    }
  private:
    size_t dim1_;
    size_t dim2_;
    size_t m_;
    storage_matrix_t storage_;
};

typedef band_matrix<double, std::vector<double>> real_bmatrix;
typedef band_matrix<std::complex<double>, std::vector<std::complex<double>>> complex_bmatrix;

namespace details {
template<class Scalar>
class band_matrix_proxy {
  public:
    typedef Scalar scalar_t;
    // Constructs a proxy to an element not stored in a matrix
    band_matrix_proxy()
        : value_ptr_(0) {
    }

    explicit band_matrix_proxy(Scalar * value_ptr)
        : value_ptr_(value_ptr) {
    }

    band_matrix_proxy(band_matrix_proxy const &) = default;
    band_matrix_proxy(band_matrix_proxy &&) = default;
    ~band_matrix_proxy() = default;

    // Conversion to scalar
    operator Scalar() const {
      return value_ptr_ ? *value_ptr_ : 0;
    }

    band_matrix_proxy & operator = (scalar_t value) {
      assert(value_ptr_);

      *value_ptr_ = value;

      return *this;
    }
  private:
    Scalar * value_ptr_;
};
} // namespace details


} } //namespace fe::la

#endif /* BAND_MATRIX_HPP_ */
