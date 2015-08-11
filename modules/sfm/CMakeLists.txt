set(the_description "SFM algorithms")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Winvalid-pch)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    # needed since we are linking static libs
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()


### LIBMV LIGHT DEFINITIONS ###

set(LIBMV_LIGHT_INCLUDES
  src/libmv_light
  src/libmv_light/third_party/ssba
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

find_package( Ceres )

if ( Ceres_FOUND )
  add_definitions("-DCERES_FOUND=1")
  list(APPEND LIBMV_LIGHT_LIBS simple_pipeline)
else ()
  add_definitions("-DCERES_FOUND=0")
endif ()


### CHECK OPENCV SFM MODULE DEPENDENCIES ###

set(OPENCV_SFM_DEPS
  opencv_core
  opencv_ts
  opencv_calib3d
  opencv_features2d
  opencv_xfeatures2d
  opencv_viz
)

ocv_check_dependencies(${OPENCV_SFM_DEPS})
ocv_include_directories("${OpenCV_SOURCE_DIR}/include/opencv")
ocv_include_modules_recurse(${OPENCV_SFM_DEPS})


### CREATE OPENCV SFM MODULE ###

ocv_add_module(sfm ${OPENCV_SFM_DEPS})
ocv_include_directories( ${LIBMV_LIGHT_INCLUDES} )

# build libmv_light
add_subdirectory(src/libmv_light)

# source files
FILE(GLOB OPENCV_SFM_SRC src/*.cpp)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB OPENCV_SFM_HDRS include/opencv2/sfm/*.hpp)

ocv_set_module_sources( HEADERS ${OPENCV_SFM_HDRS} SOURCES ${OPENCV_SFM_SRC} )

ocv_module_include_directories()
ocv_create_module( ${LIBMV_LIGHT_LIBS} )


### OPENCV SFM TESTS ###

ocv_add_accuracy_tests()


### OPENCV SFM SAMPLES ###

if ( Ceres_FOUND )
  ocv_add_samples(opencv_viz)
endif ()