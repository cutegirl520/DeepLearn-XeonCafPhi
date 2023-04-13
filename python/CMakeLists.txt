if(NOT HAVE_PYTHON)
  message(STATUS "Python interface is disabled or not all required dependecies found. Building without it...")
  return()
endif()

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
file(GLOB_RECURSE python_srcs ${PROJECT_SOURCE_DIR}/python/*.cpp)

add_library(pycaffe SHARED ${python_srcs})
target