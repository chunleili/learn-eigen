cmake_minimum_required(VERSION 3.21)
project(learn-eigen LANGUAGES CXX C) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin)

add_executable(learn-eigen main.cpp)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
if(TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 v${EIGEN3_VERSION_STRING} found in ${EIGEN3_INCLUDE_DIR}")
endif()
target_link_libraries(learn-eigen Eigen3::Eigen)


file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/data/*.mtx")

add_custom_command(
  TARGET learn-eigen POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${DATA_FILES}   ${CMAKE_SOURCE_DIR}/bin
  VERBATIM
  )