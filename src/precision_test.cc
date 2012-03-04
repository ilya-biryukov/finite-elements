#include <iostream>
#include <cstdlib>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include "dense_matrix.hpp"
#include "dense_vector.hpp"
#include "compressed_row_matrix.hpp"
#include "decomposition.hpp"
#include "solve.hpp"
#include "conversions.hpp"

using fe::la::dense_matrix_real;
using fe::la::dense_vector_real;
using fe::la::compressed_row_matrix_real;
using fe::la::compressed_row_matrix;
using fe::la::lu_decomposition;
using fe::la::solve_lu_inplace;
using fe::la::convert_matrix;
using fe::la::mvprod;

template<class Mat>
void print_matrix(Mat const & matrix) {
  for (int row = 0; row < matrix.dim1(); ++row) {
    for (int col = 0; col < matrix.dim2(); ++col) {
      std::cout << matrix(row, col) << " ";
    }
    std::cout << "\n";
  }
}
double rand_real() {
  return (double)std::rand() / (double)RAND_MAX;
}

dense_matrix_real gen_random_matrix(size_t size, double fill) {
  dense_matrix_real res(size, size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (i == j) {
        res(i, j) = 0.01 + rand_real();
        continue;
      }
      double p = rand_real();
      if (p <= fill) {
        res(i, j) = rand_real();
      }
    }
  }

  return res;
}

dense_vector_real gen_random_vector(size_t size, double fill) {
  dense_vector_real res(size);

  for (int i = 0; i < size; ++i) {
    double p = rand_real();
    if (p <= fill) {
      while (res(i) == 0) {
        res(i) = rand_real();
      }
    }
  }

  return res;
}



void do_check(size_t matrix_size, size_t tests_count) {
  namespace acc = boost::accumulators;

  double const MATRIX_FILL = 0.1;
  double const VECTOR_FILL = 1.0;

  dense_matrix_real mat_source(gen_random_matrix(matrix_size, MATRIX_FILL));
  auto mat = mat_source;
  auto mat2 = convert_matrix<compressed_row_matrix>(mat);

  std::cout << "Solving system with following matrix:\n";
  print_matrix(mat_source);

  std::cout << "\n\nLU decomposition:\n";
  lu_decomposition(mat);
  print_matrix(mat);

  std::cout << "\n\nLU decomposition of compressed matrix:\n";
  lu_decomposition(mat2);
  print_matrix(mat2);

  acc::accumulator_set<double, acc::stats<acc::tag::mean>> mean_acc;
  for (size_t test = 0; test < tests_count; ++test) {
    std::cout << "\nDoing test " << test << " of  " << tests_count << "\n";
    std::cout << "\n\nRhs of the equation:\n";
    dense_vector_real rhs_of_eq = gen_random_vector(matrix_size, VECTOR_FILL);
    print_matrix(rhs_of_eq);

    dense_vector_real rhs = rhs_of_eq;
    dense_vector_real rhs2 = rhs;

    solve_lu_inplace(mat, rhs);
    solve_lu_inplace(mat2, rhs2);

    double sum = 0.;
    for (size_t i = 0; i < rhs.dim(); ++i) {
      sum += abs(rhs(i) - rhs2(i));
    }

    std::cout << "\n\nSolution with dense_matrix:\n";
    print_matrix(rhs);
    std::cout << "\nProduct is:\n";
    print_matrix(mvprod(mat_source, rhs));

    std::cout << "\n\nSolution with sparse_matrix:\n";
    print_matrix(rhs2);
    std::cout << "\nProduct is:\n";
    print_matrix(mvprod(mat_source, rhs2));

    mean_acc(sum);
  }

  std::cout
      << "The mean difference between two solutions is: "
      << acc::mean(mean_acc) << std::endl;
}

int main() {
  srand(1234511);

  do_check(10, 100000);
}
