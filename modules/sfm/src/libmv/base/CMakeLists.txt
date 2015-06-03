# define the source files
#SET(BASE_SRC vector_test.cc)
SET(BASE_SRC )

# define the header files (make the headers appear in IDEs.)
FILE(GLOB BASE_HDRS *.h)

ADD_LIBRARY(base ${BASE_SRC} ${BASE_HDRS})

SET_TARGET_PROPERTIES(base PROPERTIES LINKER_LANGUAGE C)

#LIBMV_TEST(vector numeric)
