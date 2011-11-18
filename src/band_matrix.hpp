#ifndef BAND_MATRIX_HPP_
#define BAND_MATRIX_HPP_

#include <cmath>
#include <limits>
#include <complex>
#include <iterator>

#include <boost/iterator/iterator_facade.hpp>

#include "dense_matrix.hpp"
#include "dense_vector.hpp"
#include "products.hpp"
#include "details/sparse_element_proxy.hpp"
#include "details/sparse_matrix_vector_product.hpp"

namespace fe { namespace la {
  template<class Scalar, class Storage>
  class band_matrix {
    private:
      typedef details::sparse_element_proxy<Scalar> proxy_t;
      typedef dense_matrix<Scalar, Storage> storage_matrix_t;
      template<class RanIter>
      class non_null_row_iter
          : public boost::iterator_facade<
              non_null_row_iter<RanIter>
              ,typename std::iterator_traits<RanIter>::reference
              ,boost::random_access_traversal_tag> {
        public:
          non_null_row_iter(size_t start_index, RanIter iter)
              : index_(start_index), iter_(iter) {
          }

          size_t index() const {
            return index_;
          }
        private:
          bool equal(non_null_row_iter const & other) const {
            return iter_ == other.iter_;
          }

          typename std::iterator_traits<RanIter>::reference dereference() const {
            return *iter_;
          }

          typename std::iterator_traits<RanIter>::difference_type
              distance(non_null_row_iter const & other) const {
            auto dist = std::distance(other.iter_, iter_);

            assert(dist == (int)other.index_ - (int)index_);

            return dist;
          }

          void increment() {
            advance(1);
          }

          void decrement() {
            advance(-1);
          }

          void advance(
              typename std::iterator_traits<RanIter>::difference_type n) {
            std::advance(iter_, n);
            index_ += n;
          }
        private:
          RanIter iter_;
          size_t index_;
        private:
          friend class boost::iterator_core_access;
      };

      template<class RanIter>
      class non_null_col_iter
          : public boost::iterator_facade<
              non_null_col_iter<RanIter>,
              typename std::iterator_traits<RanIter>::reference,
              boost::random_access_traversal_tag> {
        public:
          non_null_col_iter(size_t start_index, RanIter iter, size_t pitch)
            : index_(start_index), iter_(iter), pitch_(pitch) {
          }

          size_t index() {
            return index_;
          }
        private:
          friend class boost::iterator_core_access;

          bool equal(non_null_col_iter const & other) const {
            return iter_ == other.iter_;
          }

          typename std::iterator_traits<RanIter>::reference dereference() const {
            return *iter_;
          }

          typename std::iterator_traits<RanIter>::difference_type
              distance(non_null_col_iter const & other) const {
            auto dist = std::distance(iter_, other.iter_) / pitch_;
            assert(dist == (int)other.index - (int)index_);
            return dist;
          }

          void advance(
              typename std::iterator_traits<RanIter>::difference_type n) {
            std::advance(iter_, n * pitch_);
            index_ += n;
          }

          void increment() {
            advance(1);
          }

          void decrement() {
            advance(-1);
          }
        private:
          size_t index_;
          size_t pitch_;
          RanIter iter_;
      };
    public:
      typedef Scalar scalar_t;
      typedef Storage storage_t;
      typedef proxy_t reference_t;
      typedef Scalar const_reference_t;
      typedef non_null_row_iter<typename Storage::iterator> nn_row_iterator_t;
      typedef non_null_row_iter<typename Storage::const_iterator> nn_row_const_iterator_t;
      typedef non_null_col_iter<typename Storage::iterator> nn_col_iterator_t;
      typedef non_null_col_iter<typename Storage::const_iterator> nn_col_const_iterator_t;

      band_matrix(size_t dim1, size_t dim2, size_t bands_left, size_t bands_right)
          : dim1_(dim1), dim2_(dim2), ml_(bands_left), mr_(bands_right),
            storage_(dim1, band_width()) {
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

      size_t band_width() const {
        return mr_ + ml_ + 1;
      }

      reference_t operator () (size_t i, size_t j) {
        assert(i < dim1());
        assert(j < dim2());

        // If j < i - ml || j > i + mr we must return 0
        if (j + ml_ < i || j > i + mr_) {
          return proxy_t();
        } else {
          // Otherwise we return a proxy to corresponding element
          auto & element = storage_(i, j - (i - ml_));
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

      // Non-null row iterators
      nn_row_iterator_t nnrow_begin(size_t row) {
        assert(row < dim1());

        return {get_nn_col_index_for_row(row), get_row_begin(row)};
      }

      nn_row_iterator_t nnrow_end(size_t row) {
        assert(row < dim1());

        return nnrow_begin(row) + get_nn_col_count_for_row(row);
      }

      nn_row_const_iterator_t nnrow_begin(size_t row) const {
        assert(row < dim1());

        return nnrow_cbegin(row);
      }

      nn_row_const_iterator_t nnrow_end(size_t row) const {
        assert(row < dim1());

        return nnrow_cend(row);
      }

      nn_row_const_iterator_t nnrow_cbegin(size_t row) const {
        assert(row < dim1());

        return {get_nn_col_index_for_row(row), get_row_begin(row)};
      }

      nn_row_const_iterator_t nnrow_cend(size_t row) const {
        assert(row < dim1());

        return nnrow_cbegin(row) + get_nn_col_count_for_row(row);
      }

      // Non-null column iterators
      nn_col_iterator_t nncol_begin(size_t col) {
        assert(col < dim2());

        size_t pitch = std::max(storage_.dim2() - 1, 1);
        return {get_nn_row_index_for_col(col), get_col_begin(col), pitch};
      }

      nn_col_iterator_t nncol_end(size_t col) {
        assert(col < dim2());

        return nncol_begin(col) + get_nn_row_count_for_col(col);
      }

      nn_col_const_iterator_t nncol_begin(size_t col) const {
        assert(col < dim2());

        return nncol_cbegin(col);
      }

      nn_col_const_iterator_t nncol_end(size_t col) const {
        assert(col < dim2());

        return nncol_cend(col);
      }

      nn_col_const_iterator_t nncol_cbegin(size_t col) const {
        assert(col < dim2());

        size_t pitch = std::max(storage_.dim2() - 1, (size_t)1);
        return {get_nn_row_index_for_col(col), get_col_begin(col), pitch};
      }

      nn_col_const_iterator_t nncol_cend(size_t col) const {
        assert(col < dim2());

        return nncol_cbegin(col) + get_nn_row_count_for_col(col);
      }
    private:
      size_t get_nn_col_index_for_row(size_t row) const {
        return (row < ml_) ? 0 : row - ml_;
      }

      size_t get_nn_col_count_for_row(size_t row) const {
        size_t first = get_nn_col_index_for_row(row);
        size_t last = (row + mr_ < dim2()) ? row + mr_ : dim2() - 1;

        return last - first + 1;
      }

      size_t get_nn_row_index_for_col(size_t col) const {
        return (col > mr_) ? col - mr_ : 0;
      }

      size_t get_nn_row_count_for_col(size_t col) const {
        size_t first = get_nn_row_index_for_col(col);
        size_t last = (col + ml_ < dim1()) ? col + ml_ : dim1() - 1;

        return last - first + 1;
      }


      typename storage_t::iterator get_row_begin(size_t row) const {
        auto that = const_cast<band_matrix*>(this);
        auto iter = that->storage_.data().begin() + that->storage_.dim2() * row;

        if (row < ml_) {
          iter += ml_ - row;
        }

        return iter;
      }

      typename storage_t::iterator get_col_begin(size_t col) const {
        auto that = const_cast<band_matrix*>(this);

        size_t first_row = get_nn_row_index_for_col(col);

        auto iter = that->storage_.data().begin() + that->storage_.dim2() * first_row;
        size_t diff = ml_ - first_row;
        iter += col + diff;

        return iter;
      }

      size_t dim1_;
      size_t dim2_;
      size_t ml_;
      size_t mr_;
      storage_matrix_t storage_;
  };

  typedef band_matrix<double, std::vector<double>> band_matrix_real;
  typedef band_matrix<std::complex<double>, std::vector<std::complex<double>>> band_matrix_complex;

  namespace details {
    template<class Scalar, class Storage>
    struct mat_colvec_prod_impl_f<band_matrix, Scalar, Storage> {
      dense_vector<Scalar, Storage> operator()(band_matrix<Scalar, Storage> lhs
          ,dense_vector<Scalar, Storage> rhs) const {
        return mv_sparse_prod(lhs, rhs);
      }
    };

    template<class Scalar, class Storage>
    struct mat_rowvec_prod_impl_f<band_matrix, Scalar, Storage> {
      dense_vector<Scalar, Storage> operator()(dense_vector<Scalar, Storage> lhs
          ,band_matrix<Scalar, Storage> rhs) const {
        return mv_sparse_prod(lhs, rhs);
      }
    };
  } // namespace details
} } //namespace fe::la

#endif /* BAND_MATRIX_HPP_ */
