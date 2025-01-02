cmake_minimum_required(VERSION 3.25)
project(RayTracing LANGUAGES CUDA CXX)

##Adds The Shader Static Library As A Sub Directory
add_subdirectory(Shader)

##Adds The Application Executable As A Sub Directory
add_subdirectory(Application)