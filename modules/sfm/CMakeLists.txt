set(the_description "SFM algorithms")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Werror=non-virtual-dtor -Wshadow -Wunused-parameter -Wsign-compare -Wmissing-declarations -Wunused-variable -Werror=sign-promo -Wsign-promo -Wunused-but-set-variable -Winvalid-pch)

ocv_include_directories(include
                    ./libmv_light
					./libmv_light/third_party/gflags
					./libmv_light/third_party/glog/src
					./libmv_light/third_party/ssba
					./libmv_light/third_party/ceres/include
)

ADD_DEFINITIONS(-DTEST_DATA_DIR="\\"${CMAKE_CURRENT_SOURCE_DIR}/testdata/cv/sfm/\\"")

ocv_define_module(sfm opencv_core opencv_ts)
#add_subdirectory(libmv_light)

#ocv_target_link_libraries(opencv_sfm ${OpenCV_LIBRARIES}
#                                     reconstruction
#                                     correspondence
#                                     camera
#                                     multiview
#                                     multiview_test_data
#                                     numeric
#                                     simple_pipeline
#)