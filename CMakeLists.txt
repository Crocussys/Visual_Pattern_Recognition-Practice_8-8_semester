cmake_minimum_required(VERSION 3.5)

project(Visual_Pattern_Recognition-Practice_8-8_semester LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Visual_Pattern_Recognition-Practice_8-8_semester main.cpp)

include(GNUInstallDirs)
install(TARGETS Visual_Pattern_Recognition-Practice_8-8_semester
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Visual_Pattern_Recognition-Practice_8-8_semester ${OpenCV_LIBS})
