CC=g++
GLSLC=glslc
CFLAGS=-std=c++17 -g -Wall -DNDEBUG -DGLFW_INCLUDE_NONE -DGLFW_INCLUDE_VULKAN -DSTB_IMAGE_IMPLEMENTATION -DTINYOBJLOADER_IMPLEMENTATION -I 3rdparty/include
LDFLAGS=-lglfw -lvulkan -lshaderc_shared

PROJ_NAME=vk-renderer-test
RESOURCES_PATH=resources
MODELS_PATH=$(RESOURCES_PATH)/models
SHADER_PATH=$(RESOURCES_PATH)/shaders
TEXTURES_PATH=$(RESOURCES_PATH)/textures
VERT_SHADER_SUFFIX=.vs.glsl
FRAG_SHADER_SUFFIX=.fs.glsl
VERT_SHADERS:= $(notdir $(shell find $(SHADER_PATH) -name '*$(VERT_SHADER_SUFFIX)'))
FRAG_SHADERS:= $(notdir $(shell find $(SHADER_PATH) -name '*$(FRAG_SHADER_SUFFIX)'))
SOURCES=src/vk-renderer-test/vk-renderer-test.cpp
VERT_SHADER_OBJECTS=$(VERT_SHADERS:.glsl=.spv)
FRAG_SHADER_OBJECTS=$(FRAG_SHADERS:.glsl=.spv)
OBJECTS=$(SOURCES:.cpp=.o)
BUILD_PATH=build
OUTPUT_PATH=$(BUILD_PATH)/bin/linux64
EXECUTABLE=$(PROJ_NAME)
PROJ_TARGET=$(addprefix $(OUTPUT_PATH), $(EXECUTABLE))
SHADER_CACHE_PATH=$(BUILD_PATH)/resources/shaders/cache

all: $(SOURCES) resources vert_shaders frag_shaders
	mkdir -p $(OUTPUT_PATH)
	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) -o $(addprefix $(OUTPUT_PATH)/, $(EXECUTABLE))

resources: $(SOURCES)
	@echo "Creating build path '$(BUILD_PATH)'..."
	mkdir -p $(BUILD_PATH)
	@echo "Creating resources path '$(addprefix $(BUILD_PATH)/,resources)'..."
	mkdir -p $(addprefix $(BUILD_PATH)/,resources)
	@echo "Copying resources to '$(addprefix $(BUILD_PATH)/,resources)'..."
	cp -fR $(RESOURCES_PATH)/* $(addprefix $(BUILD_PATH)/,resources)

vert_shaders:
	$(GLSLC) -fshader-stage=vert $(addprefix $(SHADER_PATH)/, $(VERT_SHADERS)) -o $(addprefix $(SHADER_CACHE_PATH)/,$(VERT_SHADER_OBJECTS))

frag_shaders:
	$(GLSLC) -fshader-stage=frag $(addprefix $(SHADER_PATH)/, $(FRAG_SHADERS)) -o $(addprefix $(BUILD_PATH)/,resources/shaders/cache/$(FRAG_SHADER_OBJECTS))

clean:
	rm -fR $(BUILD_PATH)*
