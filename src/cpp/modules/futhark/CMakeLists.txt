cmake_minimum_required (VERSION 3.7)

project("ADBench_futhark" CXX)

set(FUTHARK_BACKEND "c" CACHE STRING "Futhark compiler backend")

add_custom_command(
     OUTPUT
         ${CMAKE_CURRENT_BINARY_DIR}/omnibus.c
         ${CMAKE_CURRENT_BINARY_DIR}/omnibus.h
     COMMAND futhark ${FUTHARK_BACKEND}
         --library omnibus.fut
         -o ${CMAKE_CURRENT_BINARY_DIR}/omnibus
     DEPENDS ba.fut gmm.fut hand.fut
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
     VERBATIM
)

add_library("Futhark" MODULE)
target_sources("Futhark" PRIVATE "${CMAKE_SOURCE_DIR}/src/cpp/shared/utils.cpp")
target_sources("Futhark" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/FutharkBA.cpp")
target_sources("Futhark" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/FutharkGMM.cpp")
target_sources("Futhark" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/FutharkHand.cpp")
target_sources("Futhark" PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/omnibus.c")
target_sources("Futhark" PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/omnibus.h")
target_include_directories("Futhark" PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if (${FUTHARK_BACKEND} STREQUAL "c")
  # Nothing to do.
elseif (${FUTHARK_BACKEND} STREQUAL "multicore")
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries("Futhark" PRIVATE Threads::Threads)
elseif (${FUTHARK_BACKEND} STREQUAL "opencl")
  find_package(OpenCL REQUIRED)
  target_link_libraries("Futhark" OpenCL::OpenCL)
elseif (${FUTHARK_BACKEND} MATCHES "c")
    # Nothing special!
else ()
  message(FATAL_ERROR "Unsupported Futhark backend: ${FUTHARK_BACKEND}")
endif()
