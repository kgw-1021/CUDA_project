cmake_minimum_required(VERSION 3.18)
project(cuda_app LANGUAGES CUDA)

# CUDA 설정
set(CMAKE_CUDA_STANDARD 17)  # CUDA 17 사용 가능
set(CMAKE_CUDA_ARCHITECTURES 75)  # SM_75 아키텍처 지정
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/O2 -Xcompiler=/EHsc")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS_DEBUG "/Zi /D_DEBUG /EHsc")
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=/Zi -Xcompiler=/D_DEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=/O2 -Xcompiler=/DNDEBUG")

# 소스 파일 지정
set(SRC_FILES
    src/main.cu
    src/kernels.cu
)

# 인클루드 디렉토리 추가
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA 실행 파일 생성
add_executable(cuda_app ${SRC_FILES})

# 컴파일러 플래그 추가 (-Wall, -O2 적용)
target_compile_options(cuda_app PRIVATE -Xcompiler=-Wall -Xcompiler=-O2)

# nvcc 설정
set_target_properties(cuda_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


