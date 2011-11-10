# Adds compiler flags to support enable C++11 standard. Currently only works for g++

macro (ENABLE_CXX11)
  if (${CMAKE_COMPILER_IS_GNUCXX})
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++0x")
    message (STATUS "enable_cxx11: succeeded, complier is gnu c++.")
  else (${CMAKE_COMPLIER_IS_GNUCXX})
    message (FATAL_ERROR "enable_cxx11 FAILED: Unkown compiler. Can't enable C++11 support.")
  endif (${CMAKE_COMPILER_IS_GNUCXX})
endmacro (ENABLE_CXX11)
