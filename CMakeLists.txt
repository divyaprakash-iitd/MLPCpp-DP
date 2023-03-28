cmake_minimum_required(VERSION 3.13)  # CMake version check
project(simple_example)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard

set(project_name "MLPCpp")
set(app_name ${project_name})
project(${project_name})

file(GLOB APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
set(APP_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories(${APP_INCLUDE_DIRS})

#set(MLPCPP_APP_SOURCES
#    ${APP_SOURCES}
#    CACHE LIST "List of CPP sources needed for MLPCpp")
#set(MLPCPP_INCLUDE_DIRECTORIES
#    ${APP_INCLUDE_DIRS}
#    CACHE LIST "List of include directories needed for MLPCpp")

# Add main.cpp file of project root directory as source file
# set(SOURCE_FILES src/CIOMap.cpp src/CReadNeuralNetwork.cpp src/CLayer.cpp src/CNeuralNetwork.cpp src/CLookUp_ANN.cpp)

# Add executable target with source files listed in SOURCE_FILES variable
#add_executable(mlpcpp ${SOURCE_FILES})
add_executable(${app_name} ${APP_SOURCES})