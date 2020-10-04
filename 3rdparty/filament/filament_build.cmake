include(ExternalProject)

set(FILAMENT_ROOT "${CMAKE_BINARY_DIR}/filament-binaries")

if("${PROCESSOR_ARCH}" STREQUAL "armv8")
    set(FILAMENT_GIT_REPOSITORY "https://github.com/yxlao/filament.git")
    set(FILAMENT_GIT_TAG "aarch64-dev")
else()
    set(FILAMENT_GIT_REPOSITORY "https://github.com/google/filament.git")
    set(FILAMENT_GIT_TAG "v1.8.1")
endif()

ExternalProject_Add(
    ext_filament
    PREFIX filament
    GIT_REPOSITORY ${FILAMENT_GIT_REPOSITORY}
    GIT_TAG ${FILAMENT_GIT_TAG}
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCCACHE_PROGRAM=OFF  # enables ccache, "launch-cxx" is not working.
        -DFILAMENT_ENABLE_JAVA=OFF
        -DCMAKE_C_COMPILER=${FILAMENT_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${FILAMENT_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${FILAMENT_ROOT}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DUSE_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}
        -DUSE_STATIC_LIBCXX=ON
        -DFILAMENT_SUPPORTS_VULKAN=OFF
        -DFILAMENT_SKIP_SAMPLES=ON
)

set(filament_LIBRARIES
    filameshio
    filament
    filamat_lite
    filamat
    filaflat
    filabridge
    geometry
    backend
    bluegl
    ibl
    image
    meshoptimizer
    smol-v
    utils
)
