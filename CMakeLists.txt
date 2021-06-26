cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
    add_definitions("-Wall -Wextra")
else()
    add_definitions("-WX -W2")
endif()

if (NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr" CACHE STRING "install prefix" FORCE)
endif()

message(STATUS "install-prefix: ${CMAKE_INSTALL_PREFIX}")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)

project(MlpackSample)

list(APPEND CMAKE_MODULE_PATH /usr/local/share/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(PkgConfig REQUIRED)
find_package(MLPACK REQUIRED)
find_package(Boost 1.58.0 COMPONENTS system thread timer REQUIRED )
find_package(Armadillo 10.5.2 REQUIRED)  
find_package(Eigen3 REQUIRED)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(Eigen QUIET)
  if(NOT EIGEN_FOUND)
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
  else()
    set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
    set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})
  endif()
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()


pkg_check_modules(OGRE OGRE)
 
include_directories(
   include
   ${Boost_INCLUDE_DIRS}
   ${Armadillo_INCLUDE_DIR} 
    ${Boost_INCLUDE_DIR}/eigen3
   ${EIGEN3_INCLUDE_DIRS}/eigen3
   ${OGRE_INCLUDE_DIRS}   
   /usr/local/include/OGRE
   /usr/local/include/OGRE/Paging 
)

 
 
add_executable(Geeks_ML_code  src/Geeks_ML_code.cpp)
target_link_libraries(Geeks_ML_code ${MLPACK_LIBRARY})

add_executable(mlpack_sample  src/mlpack_sample.cpp)
target_link_libraries(mlpack_sample ${MLPACK_LIBRARY} ${ARMADILLO_LIBRARIES} )
