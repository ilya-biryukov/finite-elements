set (${PPREF}_SRCS
  test.cc
)

add_executable (fin_elements ${${PPREF}_SRCS})


# Sources of precision
set (precision_SRCS
  precision_test.cc
)

add_executable (precision ${precision_SRCS})
target_link_libraries (precision ${precision_LIBS})
