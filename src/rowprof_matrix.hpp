  #ifndef ROWBLOCK_MATRIX_HPP_
#define ROWBLOCK_MATRIX_HPP_

#include <vector>

#include <boost/iterator/iterator_facade.hpp>

#include "dense_matrix.hpp"
#include "sparse_element_proxy.hpp"

namespace fe { namespace la {

// Forward declarations
template<class Scalar, class Storage>
class rowprof_matrix;

namespace details {
template<class Scalar, class Storage>
rowprof_matrix<Scalar, Storage>
    rowprof_from_dense(dense_matrix<Scalar, Storage> const & source);
} // namespace details

template<class Scalar, class Storage>
class rowprof_matrix {
  private:
    template<class RanIter> class non_null_row_iter;

    typedef details::sparse_element_proxy<Scalar> proxy_t;
    typedef std::vector<size_t> index_storage_t;
  public:
    typedef Scalar scalar_t;
    typedef Storage storage_t;
    typedef proxy_t reference_t;
    typedef Scalar const_reference_t;
    typedef non_null_row_iter<typename Storage::iterator> nn_row_iterator_t;
    typedef non_null_row_iter<typename Storage::const_iterator> nn_row_const_iterator_t;
    //    typedef non_null_col_iter<typename Storage::iterator> nn_col_iterator_t;
    //    typedef non_null_col_iter<typename Storage::const_iterator> nn_col_const_iterator_t;
  private:
    template<class RanIter>
    class non_null_row_iter
        : public boost::iterator_facade<
            non_null_row_iter<RanIter>,
            typename std::iterator_traits<RanIter>::reference,
            boost::random_access_traversal_tag> {
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
  public:
    rowprof_matrix() = delete;
    rowprof_matrix(rowprof_matrix const &) = default;
    rowprof_matrix(rowprof_matrix &&) = default;
    ~rowprof_matrix() = default;

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
      size_t count = ia_[i + 1] - first;

      size_t first_index = ja_[i];
      size_t last_index = first_index + count;
      if (j < first_index || j >= last_index) {
        return proxy_t();
      } else {
        return proxy_t(&a_[first + j - first_index]);
      }
    }

    const_reference_t operator () (size_t i, size_t j) const {
      return (*const_cast<rowprof_matrix*>(this))(i, j);
    }

    // Non-null row iterators
    nn_row_iterator_t nnrow_begin(size_t row) {
      assert(row < dim1());

      return {ja_[row], a_.begin() + ia_[row]};
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

      return {ja_[row], a_.begin() + ia_[row]};
    }

    nn_row_const_iterator_t nnrow_cend(size_t row) const {
      assert(row < dim1());

      return nnrow_begin(row) + ia_[row + 1] - ia_[row];
    }
  private:
    rowprof_matrix(size_t dim1, size_t dim2)
        : dim1_(dim1), dim2_(dim2), ia_(dim1 + 1), ja_(dim1) {
    }
  private:
    size_t dim1_;
    size_t dim2_;

    index_storage_t ia_;
    index_storage_t ja_;

    storage_t a_;
  private:
    friend rowprof_matrix details::rowprof_from_dense<scalar_t, storage_t>(dense_matrix<scalar_t, storage_t> const &);
};

namespace details {

template<class Scalar, class Storage>
rowprof_matrix<Scalar, Storage>
    rowprof_from_dense(dense_matrix<Scalar, Storage> const & source) {
  rowprof_matrix<Scalar, Storage> res{source.dim1(), source.dim2()};
  size_t next_index = 0;
  for (size_t i = 0; i < source.dim1(); ++i) {
    // Find the index of the leftmost non-null element on i-th row
    size_t nn_left = 0;
    for (size_t j = 0; j != source.dim2(); ++j) {
      if (source(i, j) != 0) {
        nn_left = j;
        break;
      }
    }

    // Find the index of the rightmost non-null element
    size_t nn_right = nn_left;
    for (size_t j = source.dim2() - 1; j > nn_left; --j) {
      if (source(i, j) != 0) {
        nn_right = j;
        break;
      }
    }

    // Set index array values for the i-th row
    res.ia_[i] = res.a_.size();
    res.ja_[i] = nn_left;

    for (size_t j = nn_left; j <= nn_right; ++j) {
      res.a_.emplace_back(source(i, j));
    }
  }

  res.ia_[res.dim1()] = res.a_.size();

  return res;
}
} // namespace details

} } // namespace fe::la

#endif /* ROWBLOCK_MATRIX_HPP_ */
