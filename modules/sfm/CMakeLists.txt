set(the_description "SFM algorithms")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Werror=non-virtual-dtor -Wshadow -Wunused-parameter -Wsign-compare -Wmissing-declarations -Wunused-variable -Werror=sign-promo -Wsign-promo -Wunused-but-set-variable)

add_subdirectory(libmv)

ocv_define_module(sfm opencv_core opencv_ts)

#add_subdirectory(src)

