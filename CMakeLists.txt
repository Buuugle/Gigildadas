cmake_minimum_required(VERSION 3.31)
project(PythonClassEditor C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(${PROJECT_NAME} SHARED
        Module.c
        FileEditor.c
)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "classeditor")