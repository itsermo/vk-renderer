cmake_minimum_required(VERSION 3.19.6)
project(vkr VERSION 1.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "(cmake)")
function(get_platform_prefix __platform_prefix)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(${__platform_prefix} "win")
        if(CMAKE_CL_64)
            set(__bitness 64)
        else()
            set(__bitness 32)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(${__platform_prefix} "macos")
        set(__bitness 64)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(${__platform_prefix} "linux")
    else()
        message(FATAL_ERROR "Unsupported platform ${CMAKE_SYSTEM_NAME}")
    endif()

    set(${__platform_prefix} "${${__platform_prefix}}${__bitness}" PARENT_SCOPE)
endfunction(get_platform_prefix)

get_platform_prefix(PLATFORM_DIR)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/lib/${PLATFORM_DIR}
CACHE PATH "Output directory for static libraries.")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/lib/${PLATFORM_DIR}
CACHE PATH "Output directory for shared libraries.")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/bin/${PLATFORM_DIR}
CACHE PATH "Output directory for executables and DLL's.")

foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_CURRENT_LIST_DIR}/build/bin/${PLATFORM_DIR}" CACHE PATH "" FORCE)
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_CURRENT_LIST_DIR}/build/lib/${PLATFORM_DIR}" CACHE PATH "" FORCE)
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_CURRENT_LIST_DIR}/build/lib/${PLATFORM_DIR}" CACHE PATH "" FORCE)
endforeach()

option(BUILD_VK_RENDERER_LIB_STATIC "Builds the vk-renderer library as a static lib" ON)

# 3rdparty libraries
add_library(vkr-linalg                   INTERFACE  "${CMAKE_CURRENT_LIST_DIR}/3rdparty/linalg/include/linalg.h")
target_include_directories(vkr-linalg    INTERFACE  "${CMAKE_CURRENT_LIST_DIR}/3rdparty/linalg/include")
set_target_properties(vkr-linalg         PROPERTIES FOLDER "3rdparty")

add_library(vkr-stb-image                INTERFACE  "${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb-image/include/stb_image.h")
target_include_directories(vkr-stb-image INTERFACE  "${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb-image/include")
set_target_properties(vkr-stb-image      PROPERTIES FOLDER "3rdparty")

add_library(vkr-tiny-obj-loader                INTERFACE  "${CMAKE_CURRENT_LIST_DIR}/3rdparty/tiny-obj-loader/include/tiny_obj_loader.h")
target_include_directories(vkr-tiny-obj-loader INTERFACE  "${CMAKE_CURRENT_LIST_DIR}/3rdparty/tiny-obj-loader/include")
set_target_properties(vkr-tiny-obj-loader      PROPERTIES FOLDER "3rdparty")

# Vulkan
find_package(Vulkan REQUIRED FATAL_ERROR)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS    OFF)
set(GLFW_BUILD_DOCS     OFF)
set(GLFW_INSTALL        OFF)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/3rdparty/glfw)
set_target_properties(glfw PROPERTIES FOLDER "3rdparty")

if(BUILD_VK_RENDERER_LIB_STATIC)
    add_library(vkr STATIC)
else()
    add_library(vkr SHARED)
endif()

target_compile_features(vkr PUBLIC cxx_std_17)

target_sources(vkr PRIVATE    ${CMAKE_CURRENT_LIST_DIR}/src/vk-renderer/vk-renderer.cpp
                   PUBLIC     ${CMAKE_CURRENT_LIST_DIR}/src/vk-renderer/vk-renderer-common.hpp
                   PUBLIC     ${CMAKE_CURRENT_LIST_DIR}/src/vk-renderer/vk-renderer-utils.hpp
                   PUBLIC     ${CMAKE_CURRENT_LIST_DIR}/src/vk-renderer/vk-renderer.hpp
)

get_filename_component(VULKAN_LIB_DIR ${Vulkan_LIBRARY} DIRECTORY)

add_library(vkr-shaderc UNKNOWN IMPORTED)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_property(TARGET vkr-shaderc PROPERTY IMPORTED_LOCATION   "${VULKAN_LIB_DIR}/shaderc_shared.lib")
    set_property(TARGET vkr-shaderc PROPERTY IMPORTED_LOCATION_DEBUG   "${VULKAN_LIB_DIR}/shaderc_shared.lib")
    set_property(TARGET vkr-shaderc PROPERTY IMPORTED_LOCATION_RELEASE "${VULKAN_LIB_DIR}/shaderc_combined.lib")
else()
    set_property(TARGET vkr-shaderc PROPERTY IMPORTED_LOCATION "${VULKAN_LIB_DIR}/libshaderc_combined.a")
 endif()
 set_property(TARGET vkr-shaderc PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}")

target_link_libraries(vkr PUBLIC vkr-linalg vkr-stb-image vkr-tiny-obj-loader Vulkan::Vulkan vkr-shaderc glfw)

add_executable(vkr-test ${CMAKE_CURRENT_LIST_DIR}/src/vk-renderer-test/vk-renderer-test.cpp)
target_link_libraries(vkr-test PUBLIC vkr)
target_compile_features(vkr-test PUBLIC cxx_std_17)

# Automatically compile shaders to validate them upon building vkr
file(GLOB_RECURSE GLSL_VERT_FILES
    "${CMAKE_CURRENT_LIST_DIR}/resources/shaders/*.vs*"
    )

file(GLOB_RECURSE GLSL_FRAG_FILES
    "${CMAKE_CURRENT_LIST_DIR}/resources/shaders/*.fs*"
    )

foreach(GLSL ${GLSL_VERT_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME_WLE)
    set(SPIRV "${CMAKE_CURRENT_LIST_DIR}/build/resources/shaders/cache/${FILE_NAME}.spv")
    add_custom_command(
      OUTPUT ${SPIRV}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_LIST_DIR}/build/resources/shaders/cache"
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=vert ${GLSL} -o ${SPIRV}
      DEPENDS ${GLSL})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

foreach(GLSL ${GLSL_FRAG_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME_WLE)
    set(SPIRV "${CMAKE_CURRENT_LIST_DIR}/build/resources/shaders/cache/${FILE_NAME}.spv")
    add_custom_command(
      OUTPUT ${SPIRV}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_LIST_DIR}/build/resources/shaders/cache"
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=frag ${GLSL} -o ${SPIRV}
      DEPENDS ${GLSL})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    vkr-test-shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    SOURCES ${GLSL_VERT_FILES} ${GLSL_FRAG_FILES}
    )

set_source_files_properties(${GLSL_VERT_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${GLSL_FRAG_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

add_dependencies(vkr-test vkr-test-shaders)

## resources
add_custom_command(
        TARGET vkr-test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/resources ${CMAKE_CURRENT_LIST_DIR}/build/resources
)
                