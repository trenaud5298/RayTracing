## Basic Project Setup
cmake_minimum_required(VERSION 3.25)
project(Application LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "CMake Source Directory: ${CMAKE_CURRENT_SOURCE_DIR}")
## Collect All Source Files For Application (No CUDA Files Here)
file(GLOB_RECURSE APP_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE APP_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

##Collects All .DLL Files Needed For Application
file(GLOB_RECURSE DLL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/libs/*.dll")

## Create the Executable for Application
add_executable(Application ${APP_SRC_FILES} ${APP_HEADER_FILES})

## Set Include Directories
target_include_directories(Application PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

## Visual Studio Setup
source_group("Source Files" FILES ${APP_SRC_FILES})
source_group("Header Files" FILES ${APP_HEADER_FILES})

## Link Application with Shader Library (Static Library)
target_link_libraries(Application PRIVATE ShaderLibrary)

##Handles SDL Library (Manual Setup in Green)
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/SDL3/cmake" ${CMAKE_PREFIX_PATH})
find_package(SDL3 REQUIRED CONFIG)
target_link_libraries(Application PRIVATE SDL3::SDL3)

##Handles SDL_ttf
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/SDL_ttf/cmake" ${CMAKE_PREFIX_PATH})
find_package(SDL3_ttf REQUIRED CONFIG)
target_link_libraries(Application PRIVATE SDL3_ttf::SDL3_ttf)

##Copies All .dll FIles In libs To Output Directory
foreach(DLL ${DLL_FILES})
    message(STATUS "Copying DLL: ${DLL} to $<TARGET_FILE_DIR:Application>")
    add_custom_command(TARGET Application POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${DLL} $<TARGET_FILE_DIR:Application>
    )
endforeach()