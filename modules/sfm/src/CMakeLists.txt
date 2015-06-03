# ==============================================================================
# Include directories
# ==============================================================================
SET(LIBMV_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  #third_party/daisy/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gtest
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gtest/include
  #third_party/eigen
  #third_party/eigen/unsupported
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ssba
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glog/src
  #third_party/OpenExif/src
  #third_party/OpenExif/src/OpenExifJpeg
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ceres/include
  #CACHE PATH "include directories for libmv and its dependencies"
)

include_directories(${LIBMV_INCLUDES})

# ==============================================================================
# libmv modules
# ==============================================================================
add_subdirectory(libmv)

# ==============================================================================
# Third-party libraries:
# ==============================================================================
add_subdirectory(third_party) 
