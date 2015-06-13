set(the_description "SFM algorithms")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Werror=non-virtual-dtor -Wshadow -Wunused-parameter -Wsign-compare -Wmissing-declarations -Wunused-variable -Werror=sign-promo -Wsign-promo -Wunused-but-set-variable -Winvalid-pch)

set(LIBMV_LIGHT_INCLUDES
	include
  ./libmv_light
	./libmv_light/third_party/gflags
	./libmv_light/third_party/glog/src
	./libmv_light/third_party/ssba
	./libmv_light/third_party/ceres/include
)

ocv_include_directories(${LIBMV_LIGHT_INCLUDES})

ocv_add_module(sfm opencv_core opencv_ts)

# build libmv_light
add_subdirectory(libmv_light)

# source files
FILE(GLOB OPENCV_SFM_SRC src/*.cpp)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB OPENCV_SFM_HDRS include/opencv2/sfm/*.hpp)

#ocv_add_library(opencv_sfm ${OPENCV_SFM_SRC} ${OPENCV_SFM_HDRS})

ocv_set_module_sources( HEADERS ${OPENCV_SFM_HDRS} SOURCES ${OPENCV_SFM_SRC} )

ocv_module_include_directories()

ocv_create_module()

ocv_target_link_libraries(opencv_sfm  ${OpenCV_LIBRARIES}
                                  reconstruction
                                  correspondence
                                  camera
                                  multiview
                                  #multiview_test_data
                                  numeric
                                  simple_pipeline
)

install(TARGETS opencv_sfm
        DESTINATION lib
)

install(DIRECTORY include/opencv2/sfm
        DESTINATION include/opencv2
        COMPONENT main
)

## TESTS ##

ADD_DEFINITIONS(-DTEST_DATA_DIR="\\"${CMAKE_CURRENT_SOURCE_DIR}/testdata/cv/sfm/\\"")

#ocv_add_accuracy_tests()