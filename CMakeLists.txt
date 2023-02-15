cmake_minimum_required(VERSION 3.22)
project(NN)

set(CMAKE_CXX_STANDARD 17)

add_executable(NN main.cpp MatrixNeuralNetwork.cpp MatrixNeuralNetwork.h DataSet.cpp DataSet.h StopWatch.cpp NeuralNetworkBase.h NeuralNetworkManager.cpp NeuralNetworkManager.h NeuralNetworkBase.cpp)
