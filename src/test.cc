#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>
#include <functional>

#include "dense_matrix.hpp"
#include "dense_vector.hpp"
#include "band_matrix.hpp"
#include "rowprof_matrix.hpp"
#include "compressed_row_matrix.hpp"
#include "conversions.hpp"
#include "matrix_io.hpp"
#include "decomposition.hpp"


char const * const FILENAME = "test.mat";

template<class Mat>
void display_matrix(Mat const & mat1) {
  for (size_t i = 0; i < mat1.dim1(); ++i) {
    for (size_t j = 0; j < mat1.dim2(); ++j) {
      std::cout << mat1(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template<class Mat>
void fill(Mat & matrix) {
  for (size_t i = 0; i < matrix.dim1(); ++i) {
    if (i >= 1) {
      matrix(i, i - 1) = i;
    }
    if (i < matrix.dim2()) {
      matrix(i, i) = i;
    }

    if (i + 1 < matrix.dim2()) {
      matrix(i, i + 1) = i;
    }
  }

  //example from Cormen Algorithms book
  matrix(0, 0) = 2;
  matrix(0, 1) = 3;
  matrix(0, 2) = 1;
  matrix(0, 3) = 5;
  matrix(1, 0) = 6;
  matrix(1, 1) = 13;
  matrix(1, 2) = 5;
  matrix(1, 3) = 19;
  matrix(2, 0) = 2;
  matrix(2, 1) = 19;
  matrix(2, 2) = 10;
  matrix(2, 3) = 23;
  matrix(3, 0) = 4;
  matrix(3, 1) = 10;
  matrix(3, 2) = 11;
  matrix(3, 3) = 31;
}

template<class Vec>
void fill_vec(Vec & vec) {
  for (size_t i = 0; i < vec.dim(); ++i) {
    vec(i) = i;
  }
}

template<template<class,class> class Mat, class Scalar, class Storage>
void check_matrix(
    fe::la::dense_matrix<Scalar, Storage> const & m,
    fe::la::dense_vector<Scalar, Storage> const & col_v,
    fe::la::dense_vector<Scalar, Storage> const & row_v
) {
  using namespace fe::la;
  std::cout << "Converting from dense_matrix...\n";

  auto cm = convert_matrix<Mat>(m);
  std::cout << "We got:\n";
  display_matrix(cm);

  std::cout << "The actual number of matrix elements stored is:"
      << cm.data().size() << "\n";

  std::cout << "The product of matrix and column vector is:\n";
  display_matrix(mvprod(cm, col_v));
  std::cout << "The product of row vector and matrix is:\n";
  display_matrix(mvprod(row_v, cm));
}


//used to work with lambda function and fill matrix
template<class Callable, class Scalar, class Storage>
void fill_from_functor(Callable source, fe::la::dense_matrix<Scalar, Storage> & m) {
  for (size_t i = 0; i < m.dim1(); ++i) {
    for (size_t j = 0; j < m.dim2(); ++j) {
      m(i, j) = source(i, j);
    }
  }
}

//used to display matrix as LU all in one matrix
//assuming that upper triangle is U matrix and lower triangle is L matrix, with ones on its diagonale
template<template<class, class> class Mat, class Scalar, class Storage>
void check_decomposition(
    fe::la::dense_matrix<Scalar, Storage> const & matrix
) {
  using namespace fe::la;

  auto lu_matrix = convert_matrix<Mat>(matrix);
  auto ldu_matrix = convert_matrix<Mat>(matrix);

  std::cout << "Performing LU decomposition of matrix:\n";
  display_matrix(lu_matrix);
  fe::la::sparse_lu_decomposition(lu_matrix);
  {
	//lambda function to fill lower triangle matrix
    auto fill_l = [&lu_matrix](size_t i, size_t j)->Scalar {
      if (i > j) {
        return lu_matrix(i, j);
      } else if (i == j){
        return 1;
      } else {
        return 0;
      }
    };

	//lambda function to fill upper triangle matrix
    auto fill_u = [&lu_matrix](size_t i, size_t j)->Scalar {
      if (i > j) {
        return 0;
      } else {
        return lu_matrix(i, j);
      }
    };

	//create two new matricies L and U and fill them
    fe::la::dense_matrix<Scalar, Storage> l{matrix.dim1(), matrix.dim2()};
    fe::la::dense_matrix<Scalar, Storage> u{matrix.dim1(), matrix.dim2()};
    fill_from_functor(fill_l, l);
    fill_from_functor(fill_u, u);

	//display all matricies to check them visually
    std::cout << "L looks like this:\n";
    display_matrix(l);
    std::cout << "U looks like this:\n";
    display_matrix(u);
    std::cout << "Their product is:\n";
    display_matrix(mprod(l, u));
  }

  std::cin.get();

  std::cout << "Performing LDU decomposition of matrix:\n";
  display_matrix(ldu_matrix);
  fe::la::sparse_ldu_decomposition(ldu_matrix);
  {
	//lambda function to fill lower triangle matrix
    auto fill_l = [&ldu_matrix](size_t i, size_t j)->Scalar {
      if (i > j) {
        return ldu_matrix(i, j);
      } else if (i == j){
        return 1;
      } else {
        return 0;
      }
    };

	//lambda function to fill upper triangle matrix
    auto fill_d = [&ldu_matrix](size_t i, size_t j)->Scalar {
      if (i == j) {
        return ldu_matrix(i, j);
      } else {
        return 0;
      }
    };

	//lambda function to fill triangle matrix
    auto fill_u = [&ldu_matrix](size_t i, size_t j)->Scalar {
      if (i > j) {
        return 0;
      } else if (i == j) {
        return 1;
      } else  {
        return ldu_matrix(i, j);
      }
    };

	//create three diff matricies L, D and U and fill the with values
    fe::la::dense_matrix<Scalar, Storage> l{matrix.dim1(), matrix.dim2()};
    fe::la::dense_matrix<Scalar, Storage> d{matrix.dim1(), matrix.dim2()};
    fe::la::dense_matrix<Scalar, Storage> u{matrix.dim1(), matrix.dim2()};
    fill_from_functor(fill_l, l);
    fill_from_functor(fill_d, d);
    fill_from_functor(fill_u, u);


	//display all matricies to check them visually
    std::cout << "L looks like this:\n";
    display_matrix(l);
    std::cout << "D looks like this:\n";
    display_matrix(d);
    std::cout << "U looks like this:\n";
    display_matrix(u);
    std::cout << "Their product is:\n";
    display_matrix(mprod(mprod(l, d), u));
  }
}


int main() {
  using namespace fe::la;

  dense_matrix_real m{4, 4};


  std::cout << "Filling dense_matrix with some values...\n";
  fill(m);

  std::cout << "We got:\n";
  display_matrix(m);
  std::cin.get();

  // check_decomposition<dense_matrix>(m);
  check_decomposition<compressed_row_matrix>(m);
  return 0; //have to terminate here, do know that's not the best way to do so

  std::cout << "Saving dense_matrix to file...\n";
  {
    std::ofstream out_f(FILENAME);
    io::save_to_stream(m, out_f);
  }
  std::cin.get();


  std::cout << "Loading dense_matrix from the same file...\n";
  {
    std::ifstream in_f(FILENAME);
    auto m2 = io::load_real_matrix_from_stream(in_f);
    std::cout << "We got:\n";
    display_matrix(m2);
    std::cin.get();
  }

  std::cout << "Making a column vector...\n";
  dense_vector_real col_vec{10};
  fill_vec(col_vec);
  std::cout << "We got:\n";
  display_matrix(col_vec);
  std::cin.get();

  std::cout << "Making a row vector...\n";
  dense_vector_real row_vec{10, vector_type::ROW_VECTOR};
  fill_vec(row_vec);
  std::cout << "We got:\n";
  display_matrix(row_vec);
  std::cin.get();

  std::cout << "Checking band_matrix...\n";
  check_matrix<band_matrix>(m, col_vec, row_vec);
  std::cin.get();

  std::cout << "Checking compressed_row_matrix...\n";
  check_matrix<compressed_row_matrix>(m, col_vec, row_vec);
  std::cin.get();

  std::cout << "Checking rowprof_matrix...\n";
  check_matrix<rowprof_matrix>(m, col_vec, row_vec);
  std::cin.get();

  std::cout << "We are done." << std::endl;
}
