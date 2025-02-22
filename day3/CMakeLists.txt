cmake_minimum_required(VERSION 3.18)
project(cuda_app LANGUAGES CUDA C)

# CUDA 설정
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/EHsc")

# Debug / Release 모드별 컴파일 플래그 설정
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "/Zi /D_DEBUG /EHsc /RTC1")  # Debug 모드에서 /RTC1 사용
    set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=/Zi -Xcompiler=/D_DEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "/O2 /Ob2 /DNDEBUG")  # Release 모드에서 /O2 사용
    set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=/O2 -Xcompiler=/DNDEBUG")
endif()

# 인클루드 디렉토리 추가
include_directories(${CMAKE_SOURCE_DIR}/include)    

# CPU 실행 파일 (C 언어 사용)
add_executable(cpu_app src/main_cpu.c)
set_target_properties(cpu_app PROPERTIES LINKER_LANGUAGE C)
target_compile_options(cpu_app PRIVATE -Wall)

# GPU 실행 파일 (CUDA 사용)
add_executable(gpu_app src/main_gpu.cu)
set_target_properties(gpu_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(gpu_app PRIVATE -Xcompiler=-Wall)
