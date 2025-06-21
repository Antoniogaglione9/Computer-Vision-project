cmake_minimum_required(VERSION 3.10)

project(opencv_project VERSION 1.0)

set(CMAKE_TOOLCHAIN_FILE "C:/Users/asus/Desktop/free/C++ CV/build/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

set(OpenCV_DIR "C:/OpenCV-MinGW-Build-OpenCV-4.5.5-x64")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(my_program src/main.cpp src/help.cpp pugixml/src/pugixml.cpp)




add_library(pugixml STATIC build/pugixml/src/pugixml.cpp)



target_include_directories(my_program PRIVATE ${OpenCV_INCLUDE_DIRS} build/pugixml/src)


target_link_libraries(my_program PRIVATE ${OpenCV_LIBS})

include_directories("pugixml")

set_target_properties(my_program PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
