#ifndef COMPRESSED_ROW_MATRIX_HPP_
#define COMPRESSED_ROW_MATRIX_HPP_

#include <vector>
#include <algorithm>
#include <iterator>

#include <boost/iterator/iterator_facade.hpp>

#include "dense_matrix.hpp"
#include "products.hpp"
#include "details/sparse_element_proxy.hpp"
#include "details/sparse_matrix_vector_product.hpp"

namespace fe { namespace la {
  // Forward declarations
  template<class Scalar, class Storage>
  class compressed_row_matrix;

  namespace details {
    template<class Scalar, class Storage>
    compressed_row_matrix<Scalar, Storage> crmatrix_from_dense(dense_matrix<Scalar, Storage> const &);
  } // namespace details


  template<class Scalar, class Storage>
  class compressed_row_matrix {
    private:
      template<class RanIndexIter, class RanIter> class non_null_row_iter;

      typedef details::sparse_element_proxy<Scalar> proxy_t;
      typedef std::vector<size_t> index_storage_t;
    public:
      typedef Scalar scalar_t;
      typedef Storage storage_t;
      typedef proxy_t reference_t;
      typedef Scalar const_reference_t;
      typedef non_null_row_iter<index_storage_t::const_iterator, typename Storage::iterator> nn_row_iterator_t;
      typedef non_null_row_iter<index_storage_t::const_iterator, typename Storage::const_iterator> nn_row_const_iterator_t;
      //    typedef non_null_col_iter<typename Storage::iterator> nn_col_iterator_t;
      //    typedef non_null_col_iter<typename Storage::const_iterator> nn_col_const_iterator_t;
    private:
      template<class RanIndexIter, class RanIter>
      class non_null_row_iter
          : public boost::iterator_facade<
              non_null_row_iter<RanIndexIter, RanIter>,
              typename std::iterator_traits<RanIter>::reference,
              boost::random_access_traversal_tag> {
        public:
          non_null_row_iter(RanIndexIter start_index, RanIter iter)
              : index_iter_(start_index), iter_(iter) {
          }

          size_t index() const {
            return *index_iter_;
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

            assert(dist == other.index_iter_ - index_iter_);

            return dist;
          }

          void increment() {
            advance(1);
          }

          void decrement() {
            advance(-1);
          }

          void advance(
              typename std::iterator_traits<RanIter>::difference_type n
          ) {
            std::advance(iter_, n);
            std::advance(index_iter_, n);
          }
        private:
          RanIter iter_;
          RanIndexIter index_iter_;
        private:
          friend class boost::iterator_core_access;
      };
    public:
      compressed_row_matrix() = delete;
      compressed_row_matrix(compressed_row_matrix const &) = default;
      compressed_row_matrix(compressed_row_matrix &&) = default;
      ~compressed_row_matrix() = default;

      size_t dim1() const {
        return dim1_;
      }

      size_t dim2() const {
        return dim2_;
      }

      storage_t & data() {
        return a_;
      }

      storage_t const & data() const {
        return a_;
      }

      reference_t operator () (size_t i, size_t j) {
        assert(i < dim1());
        assert(j < dim2());

        size_t first = ia_[i];
        size_t count = ia_[i + 1] - ia_[i];

        auto b = ja_.cbegin() + first;
        auto e = ja_.cbegin() + first + count;
        auto pos = std::lower_bound(ja_.cbegin() + first, ja_.cbegin() + first + count, j);

        if (pos == e || *pos != j ) {
          return proxy_t();
        } else {
          return proxy_t(&a_[first + (pos - b)]);
        }
      }

      const_reference_t operator () (size_t i, size_t j) const {
        return (*const_cast<compressed_row_matrix*>(this))(i, j);
      }

      // Non-null row iterators
      nn_row_iterator_t nnrow_begin(size_t row) {
        assert(row < dim1());

        size_t first = ia_[row];

        return {ja_.begin() + first, a_.begin() + first};
      }

      nn_row_iterator_t nnrow_end(size_t row) {
        assert(row < dim1());

        return nnrow_begin(row) + ia_[row + 1] - ia_[row];
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

        size_t first = ia_[row];


        return {ja_.cbegin() + first, a_.begin() + first};
      }

      nn_row_const_iterator_t nnrow_cend(size_t row) const {
        assert(row < dim1());

        return nnrow_begin(row) + ia_[row + 1] - ia_[row];
      }

    private:
      compressed_row_matrix(size_t dim1, size_t dim2)
          : dim1_(dim1), dim2_(dim2) {
      }
    private:
      size_t dim1_;
      size_t dim2_;

      index_storage_t ia_;
      index_storage_t ja_;

      storage_t a_;
    private:
      friend compressed_row_matrix
          details::crmatrix_from_dense<scalar_t, storage_t>(
              dense_matrix<scalar_t, storage_t> const &);
  };

  namespace details {
    template<class Scalar, class Storage>
    compressed_row_matrix<Scalar, Storage> crmatrix_from_dense(
        dense_matrix<Scalar, Storage> const & source) {
      compressed_row_matrix<Scalar, Storage> res{source.dim1(), source.dim2()};

      res.ia_.reserve(source.dim1() + 1);

      for (size_t i = 0; i < source.dim1(); ++i) {
        res.ia_.emplace_back(res.a_.size());
        for (size_t j = 0; j < source.dim2(); ++j) {
          if (source(i, j) != 0) {
            res.ja_.emplace_back(j);
            res.a_.emplace_back(source(i, j));
          }
        }
      }

      res.ia_.emplace_back(res.a_.size());

      return res;
    }

    template<class Scalar, class Storage>
    struct mat_colvec_prod_impl_f<compressed_row_matrix, Scalar, Storage> {
      dense_vector<Scalar, Storage> operator()(
          compressed_row_matrix<Scalar, Storage> lhs
          , dense_vector<Scalar, Storage> rhs) const {
        return mv_sparse_prod(lhs, rhs);
      }
    };
  } //namespace details

  typedef compressed_row_matrix<double, std::vector<double>> compressed_row_matrix_real;
  typedef compressed_row_matrix<std::complex<double>, std::vector<std::complex<double>>> compressed_row_matrix_complex;
} } // namespace fe::la

#endif /* COMPRESSED_ROW_MATRIX_HPP_ */
