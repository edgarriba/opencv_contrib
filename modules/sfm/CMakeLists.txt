set(the_description "SFM algorithms")

##############
#  opencv_sfm
##############

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Werror=non-virtual-dtor -Wshadow -Wunused-parameter -Wsign-compare -Wmissing-declarations -Wunused-variable -Werror=sign-promo -Wsign-promo -Wunused-but-set-variable)

ocv_define_module(sfm opencv_core opencv_ts)

add_subdirectory(src)


##############
#  sfm_tests
##############

ADD_DEFINITIONS(-DTEST_DATA_DIR="\\"${CMAKE_CURRENT_SOURCE_DIR}/testdata/cv/sfm/\\"")


