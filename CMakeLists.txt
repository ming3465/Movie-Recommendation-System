cmake_minimum_required(VERSION 3.14)
project(recsys
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "C++17 movie recommender (k-NN collaborative filtering)"
)

include(FetchContent)

# doctest v2.4.11 and CLI11 v2.4.2 still declare cmake_minimum_required < 3.5,
# which CMake 4 rejects. Override the policy floor for the fetched subprojects.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# --- Third-party dependencies (fetched at configure time) --------------------

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(doctest CLI11)

# --- Shared warning flags ----------------------------------------------------

add_library(recsys_warnings INTERFACE)
if(MSVC)
    target_compile_options(recsys_warnings INTERFACE /W4 /utf-8 /permissive-)
else()
    target_compile_options(recsys_warnings INTERFACE -Wall -Wextra -Wpedantic)
endif()

# --- recsys library ----------------------------------------------------------

add_library(recsys STATIC
    src/dataset.cpp
)
target_compile_features(recsys PUBLIC cxx_std_17)
target_include_directories(recsys PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(recsys PRIVATE recsys_warnings)

# --- CLI executable ----------------------------------------------------------

add_executable(recsys_cli apps/recsys_cli.cpp)
target_compile_features(recsys_cli PRIVATE cxx_std_17)
target_link_libraries(recsys_cli PRIVATE recsys CLI11::CLI11 recsys_warnings)

# Statically link the GCC runtime on MinGW so the binary runs without the
# MSYS2 bin dir on PATH.
if(MINGW)
    target_link_options(recsys_cli PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# --- MovieLens 100K download target ------------------------------------------
# Run with: cmake --build build --target download_movielens_100k

add_custom_target(download_movielens_100k
    COMMAND ${CMAKE_COMMAND}
        -DDEST=${CMAKE_SOURCE_DIR}/data
        -P ${CMAKE_SOURCE_DIR}/cmake/DownloadMovieLens.cmake
    COMMENT "Downloading MovieLens 100K into data/"
    VERBATIM
)

# --- Testing -----------------------------------------------------------------

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${doctest_SOURCE_DIR}/scripts/cmake)
include(doctest)
