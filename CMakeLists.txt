cmake_minimum_required(VERSION 3.31)
project(Gigildadas C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Détection automatique des headers Python
if(NOT DEFINED PYTHON_INCLUDE_DIR)
    find_package(Python3 COMPONENTS Development)
    if(Python3_FOUND)
        set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
    else()
        # Fallback pour les systèmes Unix-like
        set(PYTHON_INCLUDE_DIR "/usr/include/python3.13")
    endif()
endif()

add_library(Container SHARED
        src/Module.c
        src/Container.c
        src/Header.c
        src/Utils.c
        src/Sections.c
)

target_include_directories(Container PRIVATE
        ${PYTHON_INCLUDE_DIR}
)

set_target_properties(Container PROPERTIES
        OUTPUT_NAME "container"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../gigildadas"
)