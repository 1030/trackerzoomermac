cmake_minimum_required(VERSION 3.28...3.30)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" ON)
option(ENABLE_QT "Use Qt functionality" OFF)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

# Ensure OBS headers from bundled deps are visible (incl. obs-frontend-api.h)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.deps/include")

# --- AprilTag (AprilRobotics/apriltag) ---
# Vendored for cross-platform builds (macOS + Windows/MSVC).
set(APRILTAG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/apriltag")
add_library(apriltag STATIC
    "${APRILTAG_DIR}/apriltag.c"
    "${APRILTAG_DIR}/apriltag_pose.c"
    "${APRILTAG_DIR}/apriltag_quad_thresh.c"

    "${APRILTAG_DIR}/common/g2d.c"
    "${APRILTAG_DIR}/common/homography.c"
    "${APRILTAG_DIR}/common/image_u8.c"
    "${APRILTAG_DIR}/common/image_u8_parallel.c"
    "${APRILTAG_DIR}/common/image_u8x3.c"
    "${APRILTAG_DIR}/common/image_u8x4.c"
    "${APRILTAG_DIR}/common/matd.c"
    "${APRILTAG_DIR}/common/pthreads_cross.c"
    "${APRILTAG_DIR}/common/svd22.c"
    "${APRILTAG_DIR}/common/time_util.c"
    "${APRILTAG_DIR}/common/unionfind.c"
    "${APRILTAG_DIR}/common/workerpool.c"
    "${APRILTAG_DIR}/common/zarray.c"
    "${APRILTAG_DIR}/common/zhash.c"
    "${APRILTAG_DIR}/common/zmaxheap.c"

    # Tag family: match Python default (tag16h5)
    "${APRILTAG_DIR}/tag16h5.c"
)

target_include_directories(apriltag PUBLIC "${APRILTAG_DIR}" "${APRILTAG_DIR}/common")

# Needed for linking apriltag (static) into our plugin module on ELF platforms.
# (Fixes: relocation ... can not be used when making a shared object; recompile with -fPIC)
set_target_properties(apriltag PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(apriltag PRIVATE _CRT_SECURE_NO_WARNINGS)
if(MSVC)
  # Our CI enables warnings-as-errors globally; apriltag upstream produces a number
  # of harmless conversion/macro warnings under MSVC. Disable "treat warnings as errors"
  # and silence the noisy ones for this vendored library.
  target_compile_options(apriltag PRIVATE /WX- /wd4005 /wd4068 /wd4244 /wd4267)
else()
  # Our project uses very strict warnings-as-errors; apriltag upstream is clean
  # but can trip on warnings treated as errors under Clang.
  target_compile_options(apriltag PRIVATE -Wno-shorten-64-to-32 -Wno-comma -Wno-unreachable-code)
endif()

# Ensure apriltag is not built with warnings-as-errors under Xcode
set_target_properties(apriltag PROPERTIES
  XCODE_ATTRIBUTE_GCC_TREAT_WARNINGS_AS_ERRORS "NO"
  XCODE_ATTRIBUTE_CLANG_TREAT_WARNINGS_AS_ERRORS "NO"
)

# Our plugin links apriltag statically.
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE apriltag)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
  target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAVE_OBS_FRONTEND_API=1)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>
  )
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
endif()

target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/plugin-main.c src/trackerzoomer-filter.c)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})
