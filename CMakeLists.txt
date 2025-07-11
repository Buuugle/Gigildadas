cmake_minimum_required(VERSION 3.31)
project(GigildasContainer C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(${PROJECT_NAME} SHARED
        Module.c
        Container.c
        Header.c
        Utils.c
        Sections.h
        Sections.c
)

target_include_directories(${PROJECT_NAME} PRIVATE
        "/usr/include/python3.13"
)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "container")