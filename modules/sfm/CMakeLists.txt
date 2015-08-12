find_package(Eigen QUIET)
find_package(glog QUIET)
find_package(gflags QUIET)

## CANNOT DETECT GLOG AND GFLAGS!
#if(NOT HAVE_EIGEN OR NOT GLOG_FOUND OR NOT GFLAGS_FOUND)
if(NOT HAVE_EIGEN)
  set(DISABLE_MSG "Module opencv_sfm disabled because the following dependencies are not found: ")
  if(NOT HAVE_EIGEN)
    set(DISABLE_MSG "${DISABLE_MSG}Eigen ")
  endif()
  if(NOT GLOG_FOUND)
    set(DISABLE_MSG "${DISABLE_MSG}Glog ")
  endif()
  if(NOT GFLAGS_FOUND)
    set(DISABLE_MSG "${DISABLE_MSG}Gflags ")
  endif()
  message(STATUS ${DISABLE_MSG})
  ocv_module_disable(sfm)
  return()
endif()

set(the_description "SFM algorithms")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Winvalid-pch -Wshadow -Wsign-compare -Wmissing-declarations -Wunused-but-set-variable -Wunused-parameter -Wunused-function)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
      # CANNOT COMPILE WITH CXX
      #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
      set(CMAKE_CXX_FLAGS "${CMAKE_CSS_FLAGS} -fPIC")
  endif()
endif()


### LIBMV LIGHT DEFINITIONS ###

set(LIBMV_LIGHT_INCLUDES
  src/libmv_light
  src/libmv_light/third_party/ssba
  "${OpenCV_SOURCE_DIR}/include/opencv"
)

set(LIBMV_LIGHT_LIBS
  reconstruction
  correspondence
  camera
  multiview
  multiview_test_data
  numeric
)


### LIBMV LIGHT EXTERNAL DEPENDENCIES ###

find_package( Ceres QUIET )

if ( Ceres_FOUND )
  add_definitions("-DCERES_FOUND=1")
  list(APPEND LIBMV_LIGHT_LIBS simple_pipeline)
else ()
  add_definitions("-DCERES_FOUND=0")
  message(STATUS "CERES support is disabled. Ceres Solver for reconstruction API is required.")
endif ()


### DEFINE OPENCV SFM MODULE DEPENDENCIES ###

set(OPENCV_SFM_DEPS
  opencv_core
  opencv_calib3d
  opencv_features2d
  opencv_xfeatures2d
)


### CREATE OPENCV SFM MODULE ###

ocv_add_module(sfm ${OPENCV_SFM_DEPS})

ocv_include_directories( ${LIBMV_LIGHT_INCLUDES} )
ocv_module_include_directories()

# source files
FILE(GLOB OPENCV_SFM_SRC src/*.cpp)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB OPENCV_SFM_HDRS include/opencv2/sfm/*.hpp)

ocv_set_module_sources( HEADERS ${OPENCV_SFM_HDRS}
                        SOURCES ${OPENCV_SFM_SRC} )

ocv_create_module()

# build libmv_light
if(NOT CMAKE_VERSION VERSION_LESS 2.8.11) # See ocv_target_include_directories() implementation
  if(TARGET ${the_module})
    get_target_property(__include_dirs ${the_module} INCLUDE_DIRECTORIES)
    include_directories(${__include_dirs})
  endif()
endif()
include_directories(${OCV_TARGET_INCLUDE_DIRS_${the_module}})
add_subdirectory(src/libmv_light)

ocv_target_link_libraries(${the_module} ${LIBMV_LIGHT_LIBS})


### CREATE OPENCV SFM TESTS ###

ocv_add_accuracy_tests()


### CREATE OPENCV SFM SAMPLES ###

if ( Ceres_FOUND )
  ocv_add_samples(opencv_viz)
endif ()