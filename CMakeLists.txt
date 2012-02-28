cmake_minimum_required (VERSION 2.8)
project (finite-elements)

include (cmake/Modules/EnableCXX11.cmake)

set (PPREF FE)

enable_testing ()
# enable_cxx11 ()

add_subdirectory (src)
