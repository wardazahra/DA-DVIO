cmake_minimum_required(VERSION 3.5)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})

add_library(dpviewer MODULE viewer.cpp viewer_cuda.cu)
target_link_libraries(dpviewer PRIVATE pybind11::module ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${Python3_LIBRARIES})
set_target_properties(dpviewer PROPERTIES 
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
