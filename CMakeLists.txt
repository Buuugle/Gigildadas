cmake_minimum_required(VERSION 3.31)
project(Gigildas C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(Container SHARED
        Module.c
        Container.c
        Header.c
        Utils.c
        Sections.c
)

target_include_directories(Container PRIVATE
        "/usr/include/python3.13"
)
set_target_properties(Container PROPERTIES
        OUTPUT_NAME "container"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../gigildas"
)