CC=g++
GLSLC=glslc
CFLAGS=-std=c++17 -g -Wall -DNDEBUG -DGLFW_INCLUDE_VULKAN -DSTB_IMAGE_IMPLEMENTATION -DTINYOBJLOADER_IMPLEMENTATION -I 3rdparty/include
LDFLAGS=-lglfw -lvulkan -lshaderc_shared

PROJ_NAME=vkTest
RESOURCES_PATH=resources/
MODELS_PATH=$(RESOURCES_PATH)models/
SHADER_PATH=$(RESOURCES_PATH)shaders/
TEXTURES_PATH=$(RESOURCES_PATH)textures/
VERT_SHADER_SUFFIX=.vs.glsl
FRAG_SHADER_SUFFIX=.fs.glsl
VERT_SHADERS:= $(notdir $(shell find $(SHADER_PATH) -name '*$(VERT_SHADER_SUFFIX)'))
FRAG_SHADERS:= $(notdir $(shell find $(SHADER_PATH) -name '*$(FRAG_SHADER_SUFFIX)'))
SOURCES=src/vktest.cpp
VERT_SHADER_OBJECTS=$(VERT_SHADERS:.glsl=.spv)
FRAG_SHADER_OBJECTS=$(FRAG_SHADERS:.glsl=.spv)
OBJECTS=$(SOURCES:.cpp=.o)
OUTPUT_PATH=bin64/
EXECUTABLE=$(PROJ_NAME)
PROJ_TARGET=$(addprefix $(OUTPUT_PATH), $(EXECUTABLE))

all: $(SOURCES) vert_shaders frag_shaders
	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) -o $(addprefix $(OUTPUT_PATH), $(EXECUTABLE))
	cp -fR $(RESOURCES_PATH) $(addprefix $(OUTPUT_PATH),resources)

vert_shaders: 
	$(GLSLC) -fshader-stage=vert $(addprefix $(SHADER_PATH), $(VERT_SHADERS)) -o $(addprefix $(OUTPUT_PATH),resources/shaders/$(VERT_SHADER_OBJECTS))

frag_shaders:
	$(GLSLC) -fshader-stage=frag $(addprefix $(SHADER_PATH), $(FRAG_SHADERS)) -o $(addprefix $(OUTPUT_PATH),resources/shaders/$(FRAG_SHADER_OBJECTS))

clean:
	rm -fR $(OUTPUT_PATH)*
