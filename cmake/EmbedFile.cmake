#[=======================================================================[.rst:
EmbedFile
------------------

Overview
^^^^^^^^

This module enables embedding of contents of a file on disk into a CPP header file.

The use case is to embed ASCII or BINARY data into a std::string or std::array,
respectively, so it can be accessible during compile-time as a variable.

The following shows a typical example of a embedding a binary file in a C++ header:

.. code-block:: cmake

  embed_file(
    MyTarget
    PRIVATE
    BINARY
    CXX
    INPUT my-binary-file.bin
    OUTPUT my-symbol-file
  )

For ascii files, the same can be done, but with the ``ASCII`` option:

.. code-block:: cmake

  embed_file(
    MyTarget
    PRIVATE
    ASCII
    CXX
    INPUT my-ascii-file.txt
    OUTPUT my-symbol-file
  )

Commands
^^^^^^^^

Embedding a File Into a Target
""""""""""""""""""""""""""""""

.. command:: embed_file

  .. code-block:: cmake

    embed_file(<target> <PRIVATE|INTERFACE|PUBLIC> <BINARY|ASCII> <CXX|C> INPUT OUTPUT)

  The ``embed_file()`` function generates a project called `embed-file`, which adds
  a rule to generate a .h or .hpp header file ``OUTPUT``, from a file on disk ``INPUT``.

  A ``CXX`` header file will be generated with a .hpp extension, and will define an
  inline constexpr std::array<> or inline constexpr std::string_view<> variable containing
  the binary or ascii data of the ``INPUT`` file, respectively, with the variable name being
  derived from ``OUTPUT``.

  A ``C`` header file will be generated with a .h extension, and will define an
  inline const unsigned char[] or inline const char * variable containing
  the binary or ascii data of the ``INPUT`` file, respectively, with the variable name being
  derived from ``OUTPUT``.

  A rule is created for the ``OUTPUT`` file to be added to the target sources
  of designated ``<target>``.

  Below is a typical example of embedding a binary file to target `my_target`:

  .. code-block:: cmake

      embed_file(
        MyTarget
        PRIVATE
        BINARY
        CXX
        INPUT my_binary_file.bin
        OUTPUT my_symbol_file
      )
#]=======================================================================]

set(EMBED_FILE_C_CODE [=[
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

FILE* open_or_exit(const char* fname, const char* mode) {
  FILE* f = fopen(fname, mode)\;
  if (f == NULL) { perror(fname)\; exit(EXIT_FAILURE)\; }
  return f\;
}

int main(int argc, char** argv)
{
  if (argc < 4 || argc > 6) {
    fprintf(stderr, "usage: embed-file <target_dir> <target_file_prefix> <source_file> [ascii] [c-header]\n\n")\;
    fprintf(stderr, "desc:  Embeds the contents of <source_file> into a C++ header <target_dir>/<target_file_prefix>.hpp\n"
                    "       Contents are embedded as a std::array<unsigned char>, or std::string if [ascii] flag is set.\n")\;

    return EXIT_FAILURE\;
  }

  const char* dir_out = argv[1]\;
  char* sym = argv[2]\;

  int is_ascii = (argc > 4 && (strcmp(argv[4], "ascii") == 0) || argc > 5 && (strcmp(argv[5], "ascii") == 0)) ? 1 : 0\;
  int is_c_header = (argc > 4 && (strcmp(argv[4], "c-header") == 0) || argc > 5 && (strcmp(argv[5], "c-header") == 0))? 1 : 0\;
  FILE* in = open_or_exit(argv[3], is_ascii ? "r" : "rb")\;

  char symfile[256], sym_upper[256]\;
  snprintf(symfile, sizeof(symfile), (is_c_header ? "%s/%s.h" : "%s/%s.hpp"), dir_out, sym)\;

  printf((is_c_header ? "Embedding %s into %s/%s.h...\n" : "Embedding %s into %s/%s.hpp...\n"), argv[3], dir_out, sym)\;

  FILE* out = open_or_exit(symfile,"w")\;

  // If first character is a number, replace with _
  // Replace any non-alpha character to _
  if(isdigit(sym[0])) sym[0] = '_'\;
  for (int i = 0\; i < strlen(sym)\; i++) {
    if (!isalpha(sym[i]) && !isdigit(sym[i])) sym[i] = '_'\;
    sym_upper[i] = isalpha(sym[i]) ? toupper(sym[i]) : sym[i]\;
  }

  // Get uppercase symbol name
  int len = strlen(sym)\;
  sym_upper[len] = '\0'\;

  // Get file size
  fseek(in, 0L, SEEK_END)\;
  long file_size = ftell(in)\;
  rewind(in)\;

  fprintf(out, "#ifndef %s_H\n#define %s_H\n\n",sym_upper,sym_upper)\;

  printf("Embed Data Type   : %s\n", is_ascii ? "ascii" : "binary")\;
  printf("Header Language   : %s\n", is_c_header ? "c" : "c++")\;
  if(is_ascii)
  {
    printf("Symbol Declaration: %s %s\n", (is_c_header ? "inline const char*" : "inline const std::string"), sym_upper)\;
    fprintf(out, "#include <stddef.h>\n")\;
    fprintf(out, (is_c_header ? "\n" : "#include <string_view>\n\n"))\;
    fprintf(out, (is_c_header ? "inline const char * %s = \"" : "inline constexpr std::string_view %s { R\"%s-ascii("),sym_upper,sym_upper)\;

    while (1) {
      int c = fgetc(in)\;
      if(feof(in)) break\;
      if(c == '\r') { file_size--\; continue\; }
      if(is_c_header && c == '\n') fprintf(out, "\\n\\")\;
      fprintf(out, "%c", c)\;
    }

    if (!is_c_header) fprintf(out, ")%s-ascii\" }\;\n", sym_upper)\;
    else fprintf(out,"\"\;\n")\;

    fprintf(out, (is_c_header ? "\ninline const size_t %s_SIZE = %lu\;\n" : "\ninline constexpr size_t %s_SIZE = %lu\;\n"), sym_upper, is_c_header ? file_size + 1 : file_size)\;
  }
  else
  {
    
    printf("Symbol Declaration: inline const std::array<unsigned char,%s_SIZE> %s\n", sym_upper, sym_upper)\;
    fprintf(out, (is_c_header ? "#include <stddef.h>\n\n" : "#include <stddef.h>\n#include <array>\n\n"))\;
    fprintf(out, (is_c_header ? "inline const size_t %s_SIZE = %lu\;\n" : "inline constexpr size_t %s_SIZE = %lu\;\n"), sym_upper, file_size)\;
    fprintf(out, (is_c_header ? "inline const unsigned char %s[%s_SIZE] = {\n": "inline constexpr std::array<unsigned char,%s_SIZE> %s {\n"),sym_upper, sym_upper)\;
    unsigned char buf[256]\;
    size_t nread = 0, linecount = 0\;
    do {
      nread = fread(buf, 1, sizeof(buf), in)\;
      if (nread == 0 && ferror(in)) perror("Error parsing file: ")\;
      size_t i\;
      for (i=0\; i < nread\; i++) {
        fprintf(out, (nread < sizeof(buf) && i == nread-1 ? "0x%02x" : "0x%02x, "), buf[i])\;
        if (++linecount == 10) { fprintf(out, "\n")\; linecount = 0\; }
      }
    } while (nread > 0)\;
    if (linecount > 0) fprintf(out, "\n")\;
    fprintf(out, "}\;\n")\;
  }

  

  fprintf(out, "\n#endif\n")\;
  fclose(in)\;
  fclose(out)\;

  return EXIT_SUCCESS\;
}
]=])

function(embed_file TARGET required_arg_1)
    cmake_parse_arguments(
        PARSE_ARGV
        1
        EMBED_FILE_ARG
        "ASCII;BINARY;C;CXX;PRIVATE;PUBLIC;INTERFACE"
        "INPUT;OUTPUT"
        ""
    )

    if(NOT EMBED_FILE_ARG_INPUT)
      message(FATAL_ERROR "Function embed_file() requires INPUT argument")
    elseif(NOT EMBED_FILE_ARG_OUTPUT)
      message(FATAL_ERROR "Function embed_file() requires OUTPUT argument")
    endif()

    if(EMBED_FILE_ARG_ASCII)
      set(EMBED_MODE "ascii")
    elseif(EMBED_FILE_ARG_BINARY)
      set(EMBED_MODE "binary")
    else()
      message(FATAL_ERROR "Function embed_file() requires ASCII or BINARY option to embed")
    endif()

    if(EMBED_FILE_ARG_C)
      set(EMBED_LANGUAGE "cpp-header")
    elseif(EMBED_FILE_ARG_CXX)
      set(EMBED_LANGUAGE "cpp-header")
    else()
      message(FATAL_ERROR "Function embed_file() requires CXX or C language declaration to generate header")
    endif()

    get_filename_component(INPUT_FILE_DIR ${EMBED_FILE_ARG_INPUT} DIRECTORY)
    if(NOT INPUT_FILE_DIR)
      set(INPUT_FILE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
      set(EMBED_FILE_ARG_INPUT ${INPUT_FILE_DIR}/${EMBED_FILE_ARG_INPUT})
    endif()

    get_filename_component(OUTPUT_FILE_DIR ${EMBED_FILE_ARG_OUTPUT} DIRECTORY)
    if(NOT OUTPUT_FILE_DIR)
      set(OUTPUT_FILE_DIR ${CMAKE_CURRENT_BINARY_DIR})
      set(EMBED_FILE_ARG_OUTPUT ${OUTPUT_FILE_DIR}/${EMBED_FILE_ARG_OUTPUT})
    endif()
    get_filename_component(OUTPUT_FILE_NAME ${EMBED_FILE_ARG_OUTPUT} NAME_WLE)

    file(WRITE ${CMAKE_BINARY_DIR}/src/embed-file.c ${EMBED_FILE_C_CODE})
    if(NOT TARGET embed-file)
        add_executable(embed-file ${CMAKE_BINARY_DIR}/src/embed-file.c)
    endif()

    add_custom_command(
        OUTPUT  ${EMBED_FILE_ARG_OUTPUT}
        COMMAND embed-file ${OUTPUT_FILE_DIR} ${OUTPUT_FILE_NAME} ${EMBED_FILE_ARG_INPUT} ${EMBED_MODE} ${EMBED_LANGUAGE}
        WORKING_DIRECTORY ${OUTPUT_FILE_DIR}
        DEPENDS ${EMBED_FILE_ARG_INPUT}
    )

    if(${EMBED_FILE_ARG_PRIVATE})
        target_sources(${TARGET} PRIVATE ${EMBED_FILE_ARG_OUTPUT})
    elseif(${EMBED_FILE_ARG_PUBLIC})
        target_sources(${TARGET} PUBLIC ${EMBED_FILE_ARG_OUTPUT})
    elseif(${EMBED_FILE_ARG_INTERFACE})
        target_sources(${TARGET} INTERFACE ${EMBED_FILE_ARG_OUTPUT})
    else()
        message(FATAL_ERROR "embed_file() function requires use of PRIVATE, PUBLIC, or INTERFACE parameter")
    endif()

    file(MAKE_DIRECTORY ${OUTPUT_FILE_DIR})
    file(TOUCH ${EMBED_FILE_ARG_OUTPUT})
endfunction(embed_file)