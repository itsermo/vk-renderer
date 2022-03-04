#[=======================================================================[.rst:
VulkanShaders
------------------

Overview
^^^^^^^^

This module enables target compilation of GLSL & HLSL shader sources to 
SPIR-V/SPIR-V ASM opcode in ASCII format.

It can also be used to preprocess shader code for compile-time error checking
of GLSL/HLSL code, not generating SPIR-V opcode.

The following shows a typical example of creating a GLSL vertex & frag shader
target rule, that embeds the shader code into an header file for compile-time
access to the shader opcode in the target ``MyVulkanTarget``:

.. code-block:: cmake

    target_shaders(
        MyVulkanTarget
        EMBED
        PRIVATE
        PREPROC rimlight-shader-preamble.glsl 
        VERT    rimlight-shader.vs.glsl
        FRAG    rimlight-shader.fs.glsl
        )


Commands
^^^^^^^^

Creating Shader Compile Target Rules
""""""""""""""""""""""""""""""""""""

.. command:: target_shaders

  .. code-block:: cmake

    target_shaders(<target> [EMBED] [PRIVATE|INTERFACE|PUBLIC] [SPVASM] [GLSL_VERSION] [TARGET_ENV] <PREPROC|VERT|FRAG|TESSCONTROL|TESSEVAL|GEOMETRY|COMPUTE> <SHADER_SOURCES>)

  The ``target_shaders()`` function adds a rule to the ``<target>`` which
  will compile the shader into SPIR-V binary using `glslc` command, as
  set by the `FindVulkan` package in variable ${Vulkan_GLSLC_EXECUTABLE}.
  
  If ``[SPVASM]`` option is used, SPIR-V assmebly files in ASCII format
  will be generated instead of binary opcode.

  The shader language is is determiend by input file extensions
  `.glsl` and `.hlsl`, for GLSL and HLSL respectively.

  Using the ``[EMBED]`` option, it will generate a rule to embed the compiled
  shader SPV/SPVASM into the ``<target>`` in the form of a header file,
  as an inlined std::string or std::array<unsigned char> variable.

  ``[GLSL_VERSION]`` is set by default to 450, and ignored for HLSL files.

  ``[TARGET_ENV]`` is by default vulkan1.0, and can be any of the following:
    `vulkan1.0`, `vulkan1.1`, `vulkan1.2`, `opengl4.5`, or `opengl_compat`

  Below is a typical example of embedding a shader into a target,
  using a specific GLSL version, targeting Vulkan 1.0.

  .. code-block:: cmake

    target_shaders(
        MyVulkanTarget
        EMBED
        PRIVATE
        GLSL_VERSION 450
        TARGET_ENV vulkan1.0
        PREPROC rimlight-shader-preamble.glsl 
        VERT    rimlight-shader.vs.glsl
        FRAG    rimlight-shader.fs.glsl
        )

#]=======================================================================]

include(EmbedFile)

function(target_vulkan_shaders target required_arg_1)
    cmake_parse_arguments(
        PARSE_ARGV
        1
        SHADER_STAGE
        "EMBED;PRIVATE;PUBLIC;INTERFACE;SPVASM"
        "TARGET_ENV;GLSL_VERSION"
        "PREPROC;VERT;FRAG;TESSCONTROL;TESSEVAL;GEOMETRY;COMPUTE"
    )

    if(SHADER_STAGE_EMBED)
        include(EmbedFile)
    endif()

    if(NOT GLSL_VERSION)
        set(GLSL_VERSION 450)
    endif()

    if(NOT TARGET_ENV)
        set(TARGET_ENV vulkan1.0)
    endif()

    if(SHADER_STAGE_SPVASM)
        set(SPIRV_SUFFIX "spvasm")
        set(EMBED_MODE "ascii")
    else()
        set(SPIRV_SUFFIX "spv")
        set(EMBED_MODE "binary")
    endif()
    
    macro(generate_spirv_from_shader_sources SHADER_FILE_LIST SHADER_STAGE)
        foreach(SHADER_FILE ${SHADER_FILE_LIST})
            get_filename_component(FILE_NAME ${SHADER_FILE} NAME_WLE)
            get_filename_component(FILE_NAME_EXT ${SHADER_FILE} LAST_EXT)
            if(${SHADER_STAGE} MATCHES "preproc")
                set(SHADER_FILE_PREPROC "${CMAKE_BINARY_DIR}/shaders/vulkan/glsl/${FILE_NAME}${FILE_NAME_EXT}")
                add_custom_command(
                    OUTPUT  ${SHADER_FILE_PREPROC}
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders/glsl"
                    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -E ${SHADER_FILE} -std=${GLSL_VERSION} --target-env=${TARGET_ENV} > ${SHADER_FILE_PREPROC}
                    DEPENDS ${SHADER_FILE})
                if(SHADER_STAGE_EMBED)
                    embed_file(${target} PRIVATE ASCII CXX INPUT ${SHADER_FILE_PREPROC} OUTPUT ${CMAKE_BINARY_DIR}/include/shaders/vulkan/${FILE_NAME}.hpp)
                endif()
            else()
                set(SPIRV "${CMAKE_BINARY_DIR}/shaders/spv/${FILE_NAME}.${SPIRV_SUFFIX}")
                add_custom_command(
                    OUTPUT ${SPIRV}
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders/spv"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders/include/shaders/vulkan"
                    COMMAND ${Vulkan_GLSLC_EXECUTABLE} $<IF:$<BOOL:${SHADER_STAGE_SPVASM}>,-S,-c> $<IF:$<CONFIG:Release>,-O,-O0> -fshader-stage=${SHADER_STAGE} -std=${GLSL_VERSION} --target-env=${TARGET_ENV} ${SHADER_FILE} -o ${SPIRV}
                    DEPENDS ${SHADER_FILE})
                if(SHADER_STAGE_EMBED)
                    embed_file(${target} PRIVATE BINARY CXX INPUT ${SPIRV} OUTPUT ${CMAKE_BINARY_DIR}/shaders/include/shaders/vulkan/${FILE_NAME}.hpp)
                endif()
            endif()

            source_group("shaders/vulkan" FILES ${CMAKE_BINARY_DIR}/shaders/include/shaders/vulkan/${FILE_NAME}.hpp)
        endforeach(SHADER_FILE)

        
        if(${SHADER_STAGE_PRIVATE})
            target_sources(${target} PRIVATE ${SHADER_FILE_LIST})
            if(SHADER_STAGE_EMBED)
                target_include_directories(${target} PRIVATE ${CMAKE_BINARY_DIR}/shaders/include)
            endif()
        elseif(${SHADER_STAGE_INTERFACE})
            target_sources(${target} INTERFACE ${SHADER_FILE_LIST})
            if(SHADER_STAGE_EMBED)
                target_include_directories(${target} INTERFACE ${CMAKE_BINARY_DIR}/shaders/include)
            endif()
            
        elseif(${SHADER_STAGE_PUBLIC})
        
            target_sources(${target} PUBLIC ${SHADER_FILE_LIST})
            if(SHADER_STAGE_EMBED)
                target_include_directories(${target} PUBLIC ${CMAKE_BINARY_DIR}/shaders/include)
            endif()
        else()
            message(FATAL_ERROR "Must use PRIVATE | PUBLIC | INTERFACE with target_shaders() function")
        endif()
        set_source_files_properties(${SHADER_FILE_LIST} PROPERTIES HEADER_FILE_ONLY TRUE)
        source_group("shaders/vulkan/src" FILES ${SHADER_FILE_LIST})
    endmacro(generate_spirv_from_shader_sources)

    generate_spirv_from_shader_sources("${SHADER_STAGE_PREPROC}"     preproc    )
    generate_spirv_from_shader_sources("${SHADER_STAGE_VERT}"        vertex     )
    generate_spirv_from_shader_sources("${SHADER_STAGE_FRAG}"        fragment   )
    generate_spirv_from_shader_sources("${SHADER_STAGE_TESSCONTROL}" tesscontrol)
    generate_spirv_from_shader_sources("${SHADER_STAGE_TESSEVAL}"    tesseval   )
    generate_spirv_from_shader_sources("${SHADER_STAGE_GEOMETRY}"    geometry   )
    generate_spirv_from_shader_sources("${SHADER_STAGE_COMPUTE}"     compute    )

endfunction(target_vulkan_shaders)