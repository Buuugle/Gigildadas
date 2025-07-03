cmake_minimum_required(VERSION 3.31)
project(PythonGildasContainer C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(${PROJECT_NAME} SHARED
        Module.c
        ContainerObject.c
        HeaderObject.c
        Utils.c
)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "gildascontainer")