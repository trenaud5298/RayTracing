##Basic Project Properties
cmake_minimum_required(VERSION 3.25)
project(RayTracing LANGUAGES CXX CUDA)

##Get Source And Header Paths As Well As All .dll Files
file(GLOB_RECURSE SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE HEADER_FILES "${CMAKE_SOURCE_DIR}/include/*.h" "${CMAKE_SOURCE_DIR}/include/*.cuh")
file(GLOB_RECURSE DLL_FILES "${CMAKE_SOURCE_DIR}/libs/*.dll")
message(STATUS "Collected DLLs: ${DLL_FILES}")

##Handles Executable With Source Files And Header Files
add_executable(RayTracing ${SRC_FILES} ${HEADER_FILES})

##Handles Visual Studio Setup
source_group("Source Files" FILES ${SRC_FILES})     ##Sets Source File Folder
source_group("Header Files" FILES ${HEADER_FILES})  ##Sets Header File Folder
target_include_directories(RayTracing PUBLIC "${CMAKE_SOURCE_DIR}/include")  ##Sets Include Directory For Local <> Includes
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT RayTracing)      ##Sets Startup Project For Visual Studio To RayTracing

##Handles SDL Library (Manual Setup in Green)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libs/SDL3/cmake" ${CMAKE_PREFIX_PATH})
find_package(SDL3 REQUIRED CONFIG)
target_link_libraries(RayTracing PRIVATE SDL3::SDL3)

##Handles SDL_ttf
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libs/SDL_ttf/cmake" ${CMAKE_PREFIX_PATH})
find_package(SDL3_ttf REQUIRED CONFIG)
target_link_libraries(RayTracing PRIVATE SDL3_ttf::SDL3_ttf)

#Handles STB (Only A Single Header File Is Used)
target_include_directories(RayTracing PUBLIC "${CMAKE_SOURCE_DIR}/libs/STB/include")

##CUDA Properties
set(CUDA_ARCHITECTURES 75)
set_target_properties(RayTracing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # Enable separable compilation for CUDA
)

##Copies All .dll FIles In libs To Output Directory
foreach(DLL ${DLL_FILES})
    message(STATUS "Copying DLL: ${DLL} to $<TARGET_FILE_DIR:RayTracing>")
    add_custom_command(TARGET RayTracing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${DLL} $<TARGET_FILE_DIR:RayTracing>
    )
endforeach()