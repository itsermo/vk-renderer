cmake_minimum_required(VERSION 3.19.6)
project(vkr VERSION 1.0)

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

option(BUILD_VK_RENDERER_LIB_STATIC "Builds the vk-renderer library as a static lib" ON)

# 3rdparty libraries
add_library(vkr-linalg          INTERFACE "${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/linalg.h")
add_library(vkr-stb-image       INTERFACE "${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/stb_image.h")
add_library(vkr-tiny-obj-loader INTERFACE "${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/tiny_obj_loader.h")

add_library(vkr-vulkan SHARED IMPORTED)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_property(TARGET vkr-vulkan PROPERTY IMPORTED_IMPLIB "${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib/${PLATFORM_DIR}/vulkan-1.lib")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set_property(TARGET vkr-vulkan PROPERTY IMPORTED_LOCATION "/usr/local/lib/libvulkan.dylib")
    set_property(TARGET vkr-vulkan PROPERTY INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include")
endif()

add_library(vkr-shaderc STATIC IMPORTED)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_property(TARGET vkr-shaderc PROPERTY IMPORTED_IMPLIB "${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib/${PLATFORM_DIR}/$<$<CONFIG:Debug>:shaderc_combined_debug>$<$<CONFIG:Release>:shaderc_combined>.lib")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set_property(TARGET vkr-shaderc PROPERTY IMPORTED_LOCATION "/usr/local/lib/libshaderc_combined.a")
    set_property(TARGET vkr-shaderc PROPERTY INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include")
endif()

add_library(vkr-glfw STATIC IMPORTED)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_property(TARGET vkr-glfw PROPERTY IMPORTED_IMPLIB "${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib/${PLATFORM_DIR}/glfw3.lib")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set_property(TARGET vkr-glfw PROPERTY IMPORTED_LOCATION "/usr/local/lib/libglfw.dylib")
    set_property(TARGET vkr-glfw PROPERTY INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include")
endif()

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
target_include_directories(vkr PUBLIC ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include)
target_include_directories(vkr PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

target_link_libraries(vkr INTERFACE vkr-linalg vkr-stb-image vkr-tiny-obj-loader
                          INTERFACE vkr-vulkan vkr-shaderc vkr-glfw)

add_executable(vkr-test ${CMAKE_CURRENT_LIST_DIR}/src/vk-renderer-test/vk-renderer-test.cpp)
target_link_libraries(vkr-test PUBLIC vkr)
target_compile_features(vkr-test PUBLIC cxx_std_17)