cmake_minimum_required(VERSION 3.22)
project(NN)

set(CMAKE_CXX_STANDARD 17)

add_executable(NN main.cpp NN.cpp TrainingSet.cpp TrainingSet.h)
