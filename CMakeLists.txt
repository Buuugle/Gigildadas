cmake_minimum_required(VERSION 3.31)
project(Gigildadas C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
add_library(Container SHARED
        src/Module.c
        src/Container.c
        src/Header.c
        src/Utils.c
        src/Sections.c
)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
target_include_directories(Container PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
)
target_link_libraries(Container PRIVATE ${Python3_LIBRARIES})
set_target_properties(Container PROPERTIES
        OUTPUT_NAME "container"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../gigildadas"
)