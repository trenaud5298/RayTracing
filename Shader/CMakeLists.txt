##Handle Initial Project Setup
cmake_minimum_required(VERSION 3.25)
project(Shader LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

##Collect All Header And Source Files For Shader Library
file(GLOB_RECURSE SHADER_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh")
file(GLOB_RECURSE SHADER_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

##Creates The Static Library
add_library(ShaderLibrary STATIC ${SHADER_SRC_FILES} ${SHADER_HEADER_FILES})

target_include_directories(ShaderLibrary PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/interface")  ## Includes Shader headers
target_include_directories(ShaderLibrary PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/internal")  ## Includes Shader headers

## Visual Studio Setup
source_group("Source Files" FILES ${SHADER_SRC_FILES})
source_group("Header Files" FILES ${SHADER_HEADER_FILES})

##Set CUDA Properties
set(CUDA_ARCHITECTURES 75)
set_target_properties(ShaderLibrary PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(ShaderLibrary
    PRIVATE $<IF:$<CONFIG:Debug>,ENABLE_LOGGING,>
)


