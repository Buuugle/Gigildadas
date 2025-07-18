cmake_minimum_required(VERSION 3.31)
project(Gigildas C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Détection automatique de la plateforme
if(WIN32)
    set(LIB_EXTENSION ".dll")
elseif(APPLE)
    set(LIB_EXTENSION ".dylib")
else()
    set(LIB_EXTENSION ".so")
endif()

add_library(Container SHARED
        src/Module.c
        src/Container.c
        src/Header.c
        src/Utils.c
        src/Sections.c
)

# Configuration Python multiplateforme avec détection automatique
find_package(Python3 COMPONENTS Development QUIET)

if(Python3_FOUND)
    target_include_directories(Container PRIVATE ${Python3_INCLUDE_DIRS})
    message(STATUS "Found Python3 includes: ${Python3_INCLUDE_DIRS}")
else()
    # Fallback vers chemins hardcodés
    if(WIN32)
        # Windows
        find_path(PYTHON_INCLUDE_DIR Python.h
                PATHS
                "$ENV{USERPROFILE}/AppData/Local/Programs/Python/Python313/include"
                "C:/Python313/include"
                "C:/Python/Python313/include"
        )
    elseif(APPLE)
        # macOS
        find_path(PYTHON_INCLUDE_DIR Python.h
                PATHS
                "/opt/homebrew/include/python3.13"
                "/usr/local/include/python3.13"
                "/Library/Frameworks/Python.framework/Versions/3.13/include/python3.13"
        )
    else()
        # Linux
        find_path(PYTHON_INCLUDE_DIR Python.h
                PATHS
                "/usr/include/python3.13"
                "/usr/local/include/python3.13"
        )
    endif()

    if(PYTHON_INCLUDE_DIR)
        target_include_directories(Container PRIVATE ${PYTHON_INCLUDE_DIR})
        message(STATUS "Found Python includes: ${PYTHON_INCLUDE_DIR}")
    else()
        message(WARNING "Python headers not found, build may fail")
    endif()
endif()

set_target_properties(Container PROPERTIES
        OUTPUT_NAME "container"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../gigildadas"
)