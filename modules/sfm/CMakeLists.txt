set(the_description "SFM algorithms")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Werror=non-virtual-dtor -Wshadow -Wunused-parameter -Wsign-compare -Wmissing-declarations -Wunused-variable -Werror=sign-promo -Wsign-promo -Wunused-but-set-variable -Winvalid-pch)

ocv_add_module(sfm opencv_core opencv_ts)

set(LIBMV_LIGHT_INCLUDES
    ./libmv_light
	./libmv_light/third_party/gflags
	./libmv_light/third_party/glog/src
	./libmv_light/third_party/ssba
	./libmv_light/third_party/ceres/include
    )

ocv_include_directories(${LIBMV_LIGHT_INCLUDES})

add_subdirectory(libmv_light)

ocv_clear_vars(LIBMV_LIGHT_LIBS)

list(APPEND LIBMV_LIGHT_LIBS reconstruction)
#list(APPEND LIBMV_LIGHT_LIBS correspondence)
#list(APPEND LIBMV_LIGHT_LIBS camera)
#list(APPEND LIBMV_LIGHT_LIBS multiview)
#list(APPEND LIBMV_LIGHT_LIBS numeric)
#list(APPEND LIBMV_LIGHT_LIBS simple_pipeline)

# source files
FILE(GLOB OPENCV_SFM_SRC src/*.cpp)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB OPENCV_SFM_HDRS include/opencv2/sfm/*.hpp)

ocv_set_module_sources( HEADERS ${OPENCV_SFM_HDRS} SOURCES ${OPENCV_SFM_SRC} )

ocv_module_include_directories()

ADD_DEFINITIONS(-DTEST_DATA_DIR="\\"${CMAKE_CURRENT_SOURCE_DIR}/testdata/cv/sfm/\\"")

ocv_create_module() 

#ocv_add_accuracy_tests()

#ocv_define_module(sfm opencv_core opencv_ts)
#ocv_create_module(sfm)