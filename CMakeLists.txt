cmake_minimum_required(VERSION 3.10)
project(ML_ITMO)

set(CMAKE_CXX_STANDARD 17)

include_directories(extern/pybind11/include)

add_executable(A codeforces/A.cpp test/main.cpp)
add_executable(B KNN/ConfusionMatrix.cpp)
add_executable(C KNN/main.cpp)
add_executable(C_old KNN/main_old.cpp)

add_subdirectory(extern/pybind11)
pybind11_add_module(KNNRegressor KNN/main.cpp)
target_compile_definitions(KNNRegressor PRIVATE -DLIB)
pybind11_add_module(ConfusionMatrix KNN/ConfusionMatrix.cpp)
target_compile_definitions(ConfusionMatrix PRIVATE -DLIB)
