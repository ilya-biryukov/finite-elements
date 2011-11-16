#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

#include "dense_matrix.hpp"
#include "dense_vector.hpp"
#include "band_matrix.hpp"
#include "rowprof_matrix.hpp"
#include "compressed_row_matrix.hpp"
#include "conversions.hpp"

template<class Mat>
void display_mat(Mat const & mat1) {
  for (size_t i = 0; i < mat1.dim1(); ++i) {
    for (size_t j = 0; j < mat1.dim2(); ++j) {
      std::cout << mat1(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  using namespace fe::la;

  real_dmatrix mat1{5, 5};

  auto fill_rand = [](std::vector<double> & vec) {
    //[](std::vector<std::complex<double>> & vec) {
    for (auto & el : vec) {
      //el = std::complex<double>{rand()/((double)RAND_MAX), rand() / ((double)RAND_MAX)};
      el = rand() / ((double)RAND_MAX);
    }
  };

  //fill_rand(mat1.data());

  for (auto i : {0, 1, 2, 3, 4}) {
    mat1(i, i) = i;
    if (i + 1 < mat1.dim2()) {
      mat1(i, i + 1) = i;
    }
  }

  real_dvector vec{5};
  for (auto i : {0, 1, 2, 3, 4}) {
    vec(i) = i;
  }

  real_dvector vec2{5, vector_type::ROW_VECTOR};
  for (auto i : {0, 1, 2, 3, 4}) {
    vec2(i) = i;
  }

  auto prof = convert_matrix<compressed_row_matrix>(mat1);
  display_mat(mat1);
  display_mat(prof);

  display_mat(mvprod(mat1, vec));
  display_mat(details::mv_sparse_prod(prof, vec));
}
