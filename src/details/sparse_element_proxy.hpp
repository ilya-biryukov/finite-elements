#ifndef SPARSE_ELEMENT_PROXY_HPP_
#define SPARSE_ELEMENT_PROXY_HPP_

#include <cassert>

namespace fe { namespace la { namespace details {
  template<class Scalar>
  class sparse_element_proxy {
    public:
      typedef Scalar scalar_t;
      // Constructs a proxy to an element not stored in a matrix
      sparse_element_proxy()
          : value_ptr_(0) {
      }

      explicit sparse_element_proxy(Scalar * value_ptr)
          : value_ptr_(value_ptr) {
      }

      sparse_element_proxy(sparse_element_proxy const &) = default;
      sparse_element_proxy(sparse_element_proxy &&) = default;
      ~sparse_element_proxy() = default;

      // Conversion to scalar
      operator Scalar() const {
        return value_ptr_ ? *value_ptr_ : 0;
      }

      sparse_element_proxy & operator = (scalar_t value) {
        if (value_ptr_) {
          *value_ptr_ = value;
        } else {
          assert(value == 0);
        }

        return *this;
      }

      sparse_element_proxy & operator -= (scalar_t value) {
        if (value_ptr_) {
          *value_ptr_ -= value;
        } else {
          assert(value == 0);
        }

        return *this;
      }

      sparse_element_proxy & operator /= (scalar_t value) {
        if (value_ptr_) {
          *value_ptr_ /= value;
        }
        return *this;
      }
    private:
      Scalar * value_ptr_;
  };
} } } // namespace fe::la::details

#endif // SPARSE_ELEMENT_PROXY_HPP_
