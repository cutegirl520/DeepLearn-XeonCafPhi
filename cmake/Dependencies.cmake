# This list is required for static linking and exported to CaffeConfig.cmake
set(Caffe_LINKER_LIBS "")

# ---[ Boost
find_package(Boost 1.46 REQUIRED COMPONENTS system thread)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-glog
find_package(Glog REQUIRED)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

# ---[ Google-gflags
find_package(GFlags REQUIRED)
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GFLAGS_LIBRARIES})

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

# ---[ HDF5
find_package(HDF5 COMPONENTS HL REQUIRED)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES})

# ---[ LMDB
find_package(LMDB REQUIRED)
include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${LMDB_LIBRARIES})

# ---[ LevelDB
find_package(LevelDB REQUIRED)
include_directories(SYSTEM ${LevelDB_INCLUDE})
list(APPEND Caffe_LINKER_LIBS ${LevelDB_LIBRARIES})

# ---[ Snappy
find_package(Snappy REQUIRED)
include_directories(SYSTEM ${Snappy_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Snappy_LIBRARIES})

# ---[ CUDA
include(cmake/Cuda.cmake)
if(NOT HAVE_CUDA)
  if(CPU_ONLY)
    message("-- CUDA is disabled. Building without it...")
  else()
    message("-- CUDA is not detected by cmake. Building without