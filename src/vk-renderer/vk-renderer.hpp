#pragma once
#ifndef VK_RENDERER_HPP
#define VK_RENDERER_HPP
#include "vk-renderer-common.hpp"
#include "vk-renderer-utils.hpp"
#include <string>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <linalg.h>
#include <shaderc/shaderc.hpp>
#include <fstream>
#include <chrono>
#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#endif
#include <GLFW/glfw3.h>

#include <unordered_map>
using namespace linalg::aliases;

// Checks vulkan return code, throws runtime error exception if not VK_SUCCESS
#define CHECK_VK(vk_call, fail_str)												\
do {																			\
	VkResult vk_call_result = vk_call;											\
	if(vk_call_result != VK_SUCCESS)											\
	{																			\
		std::ostringstream ss;													\
		ss << #vk_call << " returned " << vk_call_result << ": " << fail_str;	\
		const auto & ss_str = ss.str();											\
		throw std::runtime_error(ss_str.c_str());								\
	}																			\
} while(false)

namespace vk_renderer {

	class vk_renderer {

	public:

		//const std::vector<vertex> VERTICES = {
		//    {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
		//    {{0.5f, -0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
		//    {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
		//    {{-0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
		//    {{-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
		//    {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
		//    {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
		//    {{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
		//};

		//const std::vector<uint16_t> INDICES = {

		//    // front face
		//    0,1,2,
		//    2,3,0,

		//    // back face
		//    6,5,4,
		//    4,7,6,

		//    // top face
		//    4,1,0,
		//    5,1,4,

		//    // bottom face
		//    6,3,2,
		//    7,3,6,

		//    // left face
		//    0,3,7,
		//    7,4,0,

		//    // right face
		//    1,5,6,
		//    6,2,1
		//};

		struct QueueFamilyIndices {
			std::optional<uint32_t> graphics_family;
			std::optional<uint32_t> present_family;

			bool is_complete() {
				return graphics_family.has_value() && present_family.has_value();
			}
		};

		struct SwapChainSupportDetails {
			VkSurfaceCapabilitiesKHR capabilities;
			std::vector<VkSurfaceFormatKHR> formats;
			std::vector<VkPresentModeKHR> present_modes;
		};

		void init_vulkan() {
			init_vulkan(nullptr, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, WINDOW_WIDTH, WINDOW_HEIGHT);
		}

		void init_vulkan(void * logical_device, int x_pos, int y_pos, int width, int height) {

			if (!logical_device)
			{
				glfwInit();
				create_vulkan_instance();
				setup_debug_callback();
				init_window(x_pos, y_pos, width, height);
				create_surface();
				select_physical_device();
				create_logical_device();
			}

			create_swap_chain();
			create_image_views();
			create_render_pass();
			create_descriptor_set_layout();
			create_graphics_pipeline();
			create_command_pool();
			create_color_resources();
			create_depth_resources();
			create_framebuffers();
			setup_camera();
			create_command_buffers();
			create_sync_objects();
			create_descriptor_pool();
			//create_descriptor_sets();
			//create_uniform_buffers();
		}

		void begin_frame()
		{
			vkDeviceWaitIdle(logical_device);

			reset_command_buffers();
			model_draw_queue.clear();
			//cleanup_uniform_buffers();
		}

		void add_model(const model & the_model)
		{
			for (auto it = model_cache.begin(); it != model_cache.end(); it++)
			{
				auto m = *it;
				if (the_model.id == m->id) {
					if (
						m->vertices.size() != the_model.vertices.size() ||
						m->indices.size() != the_model.indices.size() ||
						m->mip_levels != the_model.mip_levels ||
						m->texture_width != the_model.texture_width ||
						m->texture_height != the_model.texture_height ||
						m->texture_data != the_model.texture_data ||
						m->texture_num_chan != the_model.texture_num_chan
						)
					{
						model_cache.erase(it);
						break;
					}

					m->transform = the_model.transform;
					model_draw_queue.push_back(m);
					return;
				}
			}

			std::shared_ptr<engine_model> the_engine_model{ new engine_model { the_model } };

			std::tie(the_engine_model->vertex_buffer, the_engine_model->vertex_buffer_memory) = create_vertex_buffer(the_engine_model->vertices);
			std::tie(the_engine_model->index_buffer, the_engine_model->index_buffer_memory) = create_index_buffer(the_engine_model->indices);

			std::tie(the_engine_model->texture_image, the_engine_model->texture_image_memory) = create_texture_image(the_engine_model->texture_data, the_engine_model->texture_width, the_engine_model->texture_height, the_engine_model->texture_num_chan, the_engine_model->texture_data.size(), the_engine_model->mip_levels);
			the_engine_model->texture_image_view = create_image_view(the_engine_model->texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, the_engine_model->mip_levels);
			the_engine_model->texture_sampler = create_texture_sampler(the_engine_model->mip_levels);

			the_engine_model->descriptor_sets = create_descriptor_sets();
			std::tie(the_engine_model->uniform_buffers, the_engine_model->uniform_buffer_memories) = create_uniform_buffer();

			model_cache.push_back(the_engine_model);
			model_draw_queue.push_back(the_engine_model);
		}

		void set_view_proj(const float4x4 & view_matrix, const float4x4 & projection_matrix)
		{
			this->view_matrix = view_matrix;
			this->projection_matrix = projection_matrix;
		}

		void set_view_proj_from_camera()
		{
			view_matrix = cam.get_view_matrix();
			projection_matrix = cam.get_projection_matrix();
		}

		void end_frame()
		{
			for (size_t i = 0; i < model_draw_queue.size(); i++)
			{
				update_descriptor_sets(i);
			}

			fill_command_buffers();
		}

		void draw_frame()
		{
			CHECK_VK(vkWaitForFences(logical_device, 1, &in_flight_fences[current_frame], VK_TRUE, std::numeric_limits<uint64_t>::max()), "Could not wait for vulkan fences at top of draw call");

			uint32_t image_index{};
			auto vk_result = vkAcquireNextImageKHR(logical_device, swap_chain, std::numeric_limits<uint64_t>::max(), image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);
			if (vk_result == VK_ERROR_OUT_OF_DATE_KHR || framebuffer_resized) {
				framebuffer_resized = false;
				recreate_swap_chain();
				return;
			}
			else if (vk_result != VK_SUCCESS && vk_result != VK_SUBOPTIMAL_KHR) {
				throw std::runtime_error("Could not acquire vulkan swap chain image");
			}

			VkSemaphore signal_semaphores[] = { render_finished_semaphores[current_frame] };
			VkSemaphore wait_semaphores[] = { image_available_semaphores[current_frame] };
			VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

			for (size_t m = 0; m < model_draw_queue.size(); m++)
			{
				update_uniform_buffer(m, image_index);
			}

			VkSubmitInfo submit_info = {};
			submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submit_info.waitSemaphoreCount = 1;
			submit_info.pWaitSemaphores = wait_semaphores;
			submit_info.pWaitDstStageMask = wait_stages;
			submit_info.commandBufferCount = 1;
			submit_info.pCommandBuffers = &command_buffers[image_index];
			submit_info.signalSemaphoreCount = 1;
			submit_info.pSignalSemaphores = signal_semaphores;

			CHECK_VK(vkResetFences(logical_device, 1, &in_flight_fences[current_frame]), "Could not reset vulkan fences before queue submission");

			CHECK_VK(vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame]), "Could not submit vulkan graphics queue in draw call");

			VkSwapchainKHR swap_chains[] = { swap_chain };
			VkPresentInfoKHR present_info = {};
			present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
			present_info.waitSemaphoreCount = 1;
			present_info.pWaitSemaphores = signal_semaphores;
			present_info.swapchainCount = 1;
			present_info.pSwapchains = swap_chains;
			present_info.pImageIndices = &image_index;
			present_info.pResults = nullptr; // Optional

			vk_result = vkQueuePresentKHR(present_queue, &present_info);
			if (vk_result == VK_ERROR_OUT_OF_DATE_KHR || vk_result == VK_SUBOPTIMAL_KHR || framebuffer_resized) {
				framebuffer_resized = false;
				recreate_swap_chain();
			}
			else if (vk_result != VK_SUCCESS) {
				throw std::runtime_error("Could not present vulkan swap chain image");
			}

			//vkQueueWaitIdle(present_queue);

			current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
		}

		void cleanup() {

			vkDeviceWaitIdle(logical_device);

			cleanup_sync_objects();
			cleanup_models();

			if (descriptor_pool) cleanup_descriptor_pool();

			cleanup_descriptor_set_layout();

			cleanup_swap_chain();
			vkDestroySwapchainKHR(logical_device, swap_chain, nullptr);

			cleanup_command_pool();
			cleanup_logical_device();
			cleanup_surface();

			cleanup_validation_layers();

			cleanup_vulkan_instance();

			glfwDestroyWindow(window);

			glfwTerminate();
		}

		void init_window(int x_pos, int y_pos, int width, int height) {

			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
			glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
			glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);

			window = glfwCreateWindow(width, height, "vk_renderer test", nullptr, nullptr);
			glfwSetWindowPos(window, x_pos, y_pos);

			glfwSetWindowUserPointer(window, this);
			glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
			glfwSetScrollCallback(window, mouse_scroll_callback);
			glfwSetCursorPosCallback(window, mouse_pos_callback);
			glfwSetKeyCallback(window, key_callback);
		}

		bool poll_events()
		{
			glfwPollEvents();

			float current_frame_timestamp = static_cast<float>(glfwGetTime());
			delta_time = current_frame_timestamp - previous_frame_timestamp;
			previous_frame_timestamp = current_frame_timestamp;

			process_input();

			return !glfwWindowShouldClose(window);
		}

		void main_loop() {
			while (!glfwWindowShouldClose(window)) {
				glfwPollEvents();

				float current_frame_timestamp = static_cast<float>(glfwGetTime());
				delta_time = current_frame_timestamp - previous_frame_timestamp;
				previous_frame_timestamp = current_frame_timestamp;

				process_input();

				draw_frame();
			}

			vkDeviceWaitIdle(logical_device);
		}

		void run() {
			init_window(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, WINDOW_WIDTH, WINDOW_HEIGHT);
			init_vulkan();
			main_loop();
			cleanup();
		}

		const coord_system & get_coordinate_system() const { return engine_coordinate_system; }

		pose get_camera_pose() const { return cam.get_pose(); }

		void set_camera_fov_deg(float fov_deg) { this->cam.fov_deg = fov_deg; }
		float get_camera_fov_deg() const { return cam.fov_deg; }

	private:

		float4x4 projection_matrix{ linalg::identity };
		float4x4 view_matrix{ linalg::identity };

		static VkVertexInputBindingDescription get_vertex_binding_description() {
			VkVertexInputBindingDescription binding_description = {};
			binding_description.binding = 0;
			binding_description.stride = sizeof(vertex);
			binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
			return binding_description;
		}
		static std::array<VkVertexInputAttributeDescription, 3> get_vertex_attribute_description() {
			std::array<VkVertexInputAttributeDescription, 3> attribute_descriptions = {};
			attribute_descriptions[0].binding = 0;
			attribute_descriptions[0].location = 0;
			attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
			attribute_descriptions[0].offset = offsetof(vertex, pos);

			attribute_descriptions[1].binding = 0;
			attribute_descriptions[1].location = 1;
			attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			attribute_descriptions[1].offset = offsetof(vertex, color);

			attribute_descriptions[2].binding = 0;
			attribute_descriptions[2].location = 2;
			attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
			attribute_descriptions[2].offset = offsetof(vertex, tex_coord);

			return attribute_descriptions;
		}

		struct engine_model : model {
			std::vector<VkBuffer> uniform_buffers{};
			std::vector<VkDeviceMemory> uniform_buffer_memories{};
			std::vector<VkDescriptorSet> descriptor_sets{};
			VkBuffer vertex_buffer{ VK_NULL_HANDLE };
			VkDeviceMemory vertex_buffer_memory{ VK_NULL_HANDLE };
			VkBuffer index_buffer{ VK_NULL_HANDLE };
			VkDeviceMemory index_buffer_memory{ VK_NULL_HANDLE };
			VkImage texture_image{ VK_NULL_HANDLE };
			VkDeviceMemory texture_image_memory{ VK_NULL_HANDLE };
			VkImageView texture_image_view{ VK_NULL_HANDLE };
			VkSampler texture_sampler{ VK_NULL_HANDLE };
		};

		const coord_system vk_coordinate_system{ coord_axis::right, coord_axis::down, coord_axis::forward };
		const int WINDOW_WIDTH{ 800 };
		const int WINDOW_HEIGHT{ 600 };
		const float QUEUE_PRIORITY{ 1.0f };
		const size_t MAX_FRAMES_IN_FLIGHT = 2;

		GLFWwindow* window{ nullptr };
		VkInstance vulkan_instance{ VK_NULL_HANDLE };
		VkDebugUtilsMessengerEXT callback{ VK_NULL_HANDLE };
		VkPhysicalDevice physical_device{ VK_NULL_HANDLE };
		VkDevice logical_device{ VK_NULL_HANDLE };
		VkQueue graphics_queue{ VK_NULL_HANDLE };
		VkSurfaceKHR surface{ VK_NULL_HANDLE };
		VkQueue present_queue{ VK_NULL_HANDLE };
		VkSwapchainKHR swap_chain{ VK_NULL_HANDLE };
		std::vector<VkImage> swap_chain_images{};
		VkFormat swap_chain_image_format{};
		VkExtent2D swap_chain_extent{};
		std::vector<VkImageView> swap_chain_image_views{};
		VkShaderModule vert_shader_module{ VK_NULL_HANDLE };
		VkShaderModule frag_shader_module{ VK_NULL_HANDLE };
		const char * VERT_SHADER_FILENAME = "../../resources/shaders/shader.vs.glsl";
		const char * FRAG_SHADER_FILENAME = "../../resources/shaders/shader.fs.glsl";
		VkRenderPass render_pass{ VK_NULL_HANDLE };
		VkDescriptorSetLayout descriptor_set_layout{ VK_NULL_HANDLE };
		VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };
		VkPipeline graphics_pipeline{ VK_NULL_HANDLE };
		std::vector<VkFramebuffer> swap_chain_framebuffers{};
		VkCommandPool command_pool{};
		std::vector<VkCommandBuffer> command_buffers{};

		size_t current_frame{ 0 };
		std::vector<VkSemaphore> image_available_semaphores{};
		std::vector<VkSemaphore> render_finished_semaphores{};
		std::vector<VkFence> in_flight_fences{};

		bool framebuffer_resized{ false };

		VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT;

		VkImage color_image{ VK_NULL_HANDLE };
		VkDeviceMemory color_image_memory{ VK_NULL_HANDLE };
		VkImageView color_image_view{ VK_NULL_HANDLE };
		VkImage depth_image{ VK_NULL_HANDLE };
		VkDeviceMemory depth_image_memory{ VK_NULL_HANDLE };
		VkImageView depth_image_view{ VK_NULL_HANDLE };

		VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };

		std::vector<std::shared_ptr<engine_model>> model_cache{};
		std::vector<std::shared_ptr<engine_model>> model_draw_queue{};

		// camera
		uniform_buffer_object mvp{};
		//float fov_deg{ 45.0f };
		camera cam;
		float delta_time{};
		float previous_frame_timestamp{};

		const std::vector<const char*> VALIDATION_LAYERS = { "VK_LAYER_LUNARG_standard_validation" };
		const std::vector<const char*> DEVICE_EXTENSIONS = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
#ifdef NDEBUG
		const bool enable_validation_layers = false;
#else
		const bool enable_validation_layers = true;
#endif

		bool check_validation_layer_support() {
			uint32_t layer_count;
			vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

			std::vector<VkLayerProperties> available_layers(layer_count);
			vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

			for (const char* layer_name : VALIDATION_LAYERS) {
				bool layer_found = false;

				for (const auto& layer_properties : available_layers) {
					if (strcmp(layer_name, layer_properties.layerName) == 0) {
						layer_found = true;
						break;
					}
				}

				if (!layer_found) {
					return false;
				}
			}

			return true;
		}

		std::vector<const char*> get_required_extensions() {
			uint32_t glfw_extension_count = 0;
			const char** glfw_extensions;
			glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

			std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

			if (enable_validation_layers) {
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}

			return extensions;
		}

		static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
			VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
			VkDebugUtilsMessageTypeFlagsEXT message_type,
			const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
			void* pUserData) {

			std::cerr << "Validation layer: " << callback_data->pMessage << std::endl;

			return VK_FALSE;
		}

		static void framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
			auto app = reinterpret_cast<vk_renderer*>(glfwGetWindowUserPointer(window));
			app->framebuffer_resized = true;
		}

		static void mouse_scroll_callback(GLFWwindow* window, double x, double y)
		{
			auto app = reinterpret_cast<vk_renderer*>(glfwGetWindowUserPointer(window));

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS)
			{
				auto cam_pose = app->get_camera_pose();
				cam_pose.pos.y -= static_cast<float>(y) * 0.05f;
				cam_pose.pos.x -= static_cast<float>(x) * 0.05f;
				app->cam.position = cam_pose.pos;
			}
			else
			{
				auto fov = app->get_camera_fov_deg();

				if (fov >= 1.0f && fov <= 90.0f)
					fov -= static_cast<float>(y);
				if (fov <= 1.0f)
					fov = 1.0f;
				if (fov >= 90.0f)
					fov = 90.0f;

				app->set_camera_fov_deg(fov);
			}

		}

		static void mouse_pos_callback(GLFWwindow* window, double x, double y)
		{
			const float MOUSE_SENSITIVITY = 0.001f;
			static bool mouse_was_down{ false };
			static float2 prev_mouse_pos{};
			static float2 mouse_down_pos{};
			auto app = reinterpret_cast<vk_renderer*>(glfwGetWindowUserPointer(window));

			if (glfwGetMouseButton(window, 0) == GLFW_PRESS)
			{
				float2 mouse_motion{};
				float2 mouse_pos = float2{ static_cast<float>(x), static_cast<float>(y) };

				if (!mouse_was_down)
				{
					mouse_down_pos = mouse_pos;
					mouse_motion = { 0,0 };
				}
				else
					mouse_motion = mouse_pos - prev_mouse_pos;

				app->cam.yaw -= mouse_motion.x*MOUSE_SENSITIVITY;
				app->cam.pitch -= mouse_motion.y*MOUSE_SENSITIVITY;

				mouse_was_down = true;
				prev_mouse_pos = mouse_pos;
			}
			else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS)
			{
				float2 mouse_motion{};
				const float2 mouse_pos = float2{ static_cast<float>(x), static_cast<float>(y) };
				if (!mouse_was_down)
				{
					mouse_down_pos = mouse_pos;
					mouse_motion = { 0,0 };
				}
				else
					mouse_motion = mouse_pos - prev_mouse_pos;

				const auto & coords = app->get_coordinate_system();
				auto cam_pose = app->get_camera_pose();
				cam_pose.pos += MOUSE_SENSITIVITY * (mouse_motion.y* linalg::qrot(cam_pose.rot, coords.get_down()) + mouse_motion.x * linalg::qrot(cam_pose.rot, coords.get_right()));
				app->cam.position = cam_pose.pos;

				mouse_was_down = true;
				prev_mouse_pos = mouse_pos;
			}
			else
			{
				mouse_was_down = false;
				prev_mouse_pos = { 0,0 };
			}

		}

		static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
		{
			//const auto& app = reinterpret_cast<vk_renderer*>(glfwGetWindowUserPointer(window));

			if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
				glfwSetWindowShouldClose(window, true);

		}

		VkSampleCountFlagBits get_max_usable_sample_count() {
			VkPhysicalDeviceProperties physical_device_properties{};
			vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

			VkSampleCountFlags counts = std::min(physical_device_properties.limits.framebufferColorSampleCounts, physical_device_properties.limits.framebufferDepthSampleCounts);
			if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
			if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
			if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
			if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
			if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
			if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

			return VK_SAMPLE_COUNT_1_BIT;
		}

		void create_vulkan_instance()
		{
			if (enable_validation_layers && !check_validation_layer_support()) {
				throw std::runtime_error("Validation layers requested, but not available!");
			}

			uint32_t extension_count = 0;
			CHECK_VK(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr), "Could not enumerate instance extension properties");
			std::vector<VkExtensionProperties> extensions(extension_count);
			CHECK_VK(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data()), "Could not enumerate instance extension properties");

			std::cout << "Available extensions:" << std::endl;
			for (const auto& extension : extensions) {
				std::cout << "\t" << extension.extensionName << std::endl;
			}

			VkApplicationInfo app_info = {};
			app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			app_info.pApplicationName = "Vulkan Renderer";
			app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
			app_info.pEngineName = "ErmalEngine";
			app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
			app_info.apiVersion = VK_API_VERSION_1_0;

			VkInstanceCreateInfo create_info = {};
			create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			create_info.pApplicationInfo = &app_info;

			auto required_extensions = get_required_extensions();

			create_info.enabledExtensionCount = static_cast<uint32_t>(required_extensions.size());
			create_info.ppEnabledExtensionNames = required_extensions.data();

			if (enable_validation_layers) {
				create_info.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
				create_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();
			}
			else {
				create_info.enabledLayerCount = 0;
			}

			CHECK_VK(vkCreateInstance(&create_info, nullptr, &vulkan_instance), "Could not create vulkan instance");
		}

		void setup_debug_callback() {
			if (!enable_validation_layers) return;

			VkDebugUtilsMessengerCreateInfoEXT create_info = {};
			create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			create_info.pfnUserCallback = debug_callback;
			create_info.pUserData = nullptr; // Optional

			CHECK_VK(CreateDebugUtilsMessengerEXT(vulkan_instance, &create_info, nullptr, &callback), "Could not create debug callback");
		}

		void create_surface()
		{
			CHECK_VK(glfwCreateWindowSurface(vulkan_instance, window, nullptr, &surface), "Could not create vulkan window surface using GLFW");
		}

		VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback) {
			auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
			if (func != nullptr) {
				return func(instance, pCreateInfo, pAllocator, pCallback);
			}
			else {
				return VK_ERROR_EXTENSION_NOT_PRESENT;
			}
		}

		void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator) {
			auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
			if (func != nullptr) {
				func(instance, callback, pAllocator);
			}
		}

		void process_input()
		{
			float camera_speed = 2.5f * delta_time;
			if (glfwGetKey(window, GLFW_KEY_W)) cam.move_local(camera_speed * engine_coordinate_system.get_forward());
			if (glfwGetKey(window, GLFW_KEY_S)) cam.move_local(camera_speed * engine_coordinate_system.get_back());
			if (glfwGetKey(window, GLFW_KEY_A)) cam.move_local(camera_speed * engine_coordinate_system.get_left());
			if (glfwGetKey(window, GLFW_KEY_D)) cam.move_local(camera_speed * engine_coordinate_system.get_right());
			if (glfwGetKey(window, GLFW_KEY_Q)) cam.move_local(camera_speed * engine_coordinate_system.get_up());
			if (glfwGetKey(window, GLFW_KEY_Z)) cam.move_local(camera_speed * engine_coordinate_system.get_down());
		}

		std::set<std::string> check_device_extensions_support(VkPhysicalDevice device)
		{
			uint32_t extension_count;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

			std::vector<VkExtensionProperties> available_extensions(extension_count);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

			std::set<std::string> required_extensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());

			for (const auto& extension : available_extensions) {
				required_extensions.erase(extension.extensionName);
			}

			return required_extensions;
		}

		SwapChainSupportDetails query_swap_chain_support(VkPhysicalDevice device)
		{
			SwapChainSupportDetails details{};

			CHECK_VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities), "Could not get physical device surface capabilities");

			uint32_t format_count{ 0 };
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);

			if (format_count != 0) {
				details.formats.resize(format_count);
				vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
			}

			uint32_t present_mode_count{ 0 };
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);

			if (present_mode_count != 0) {
				details.present_modes.resize(present_mode_count);
				vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.present_modes.data());
			}

			return details;
		}

		int rate_device_suitability(VkPhysicalDevice device)
		{
			VkPhysicalDeviceProperties device_properties;
			VkPhysicalDeviceFeatures device_features;
			vkGetPhysicalDeviceProperties(device, &device_properties);
			vkGetPhysicalDeviceFeatures(device, &device_features);

			int score = 0;

			// Check to see if we can send graphics commands
			QueueFamilyIndices indices = find_queue_families(device);

			bool swap_chain_adequate = false;
			if (check_device_extensions_support(device).empty()) {
				SwapChainSupportDetails swap_chain_support = query_swap_chain_support(device);
				swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.present_modes.empty();
			}

			if (!indices.is_complete() || !swap_chain_adequate) return 0;

			// Discrete GPUs have a significant performance advantage
			if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
				score += 1000;
			}

			// Maximum possible size of textures affects graphics quality
			score += device_properties.limits.maxImageDimension2D;

			// Application can't function without sampler anisotropy
			if (!device_features.fillModeNonSolid || !device_features.samplerAnisotropy || !device_features.sampleRateShading) {
				return 0;
			}


			return score;
		}

		uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties)
		{
			VkPhysicalDeviceMemoryProperties mem_properties;
			vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

			for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
				if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
					return i;
				}
			}

			throw std::runtime_error("find_memory_type(...): Could not find suitable memory for vulkan physical device");
		}

		void select_physical_device()
		{
			uint32_t device_count{ 0 };
			CHECK_VK(vkEnumeratePhysicalDevices(vulkan_instance, &device_count, nullptr), "Could not enumerate vulkan physical devices");
			if (device_count == 0) throw std::runtime_error("Could not find any GPUs with vulkan support");
			std::vector<VkPhysicalDevice> devices(device_count);
			CHECK_VK(vkEnumeratePhysicalDevices(vulkan_instance, &device_count, devices.data()), "Could not enumerate vulkan physical devices");

			// Use an ordered map to automatically sort candidates by increasing score
			std::multimap<int, VkPhysicalDevice> candidates;

			for (const auto& device : devices) {
				int score = rate_device_suitability(device);
				candidates.insert(std::make_pair(score, device));
			}

			// Check if the best candidate is suitable at all
			if (candidates.rbegin()->first > 0) {
				physical_device = candidates.rbegin()->second;
				msaa_samples = get_max_usable_sample_count();
			}
			else {
				throw std::runtime_error("Could not find a suitable vulkan GPU");
			}

		}

		QueueFamilyIndices find_queue_families(VkPhysicalDevice device) {
			QueueFamilyIndices indices;

			uint32_t queue_family_count = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

			std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

			int i = 0;
			for (const auto& queue_family : queue_families) {
				if (queue_family.queueCount > 0 && queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
					indices.graphics_family = i;
				}

				VkBool32 present_support = false;
				vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);

				if (queue_family.queueCount > 0 && present_support) {
					indices.present_family = i;
				}

				if (indices.is_complete()) {
					break;
				}

				i++;
			}

			return indices;
		}

		VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats) {
			if (available_formats.size() == 1 && available_formats[0].format == VK_FORMAT_UNDEFINED) {
				return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
			}

			for (const auto& available_format : available_formats) {
				if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
					return available_format;
				}
			}

			return available_formats[0];
		}

		VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR> available_present_modes) {
			VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;

			for (const auto& available_present_mode : available_present_modes) {
				if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
					return available_present_mode;
				}
				else if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
					best_mode = available_present_mode;
				}
			}

			return best_mode;
		}

		VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities) {
			if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
				return capabilities.currentExtent;
			}
			else {

				int width, height;
				glfwGetFramebufferSize(window, &width, &height);

				VkExtent2D actual_extent = {
					static_cast<uint32_t>(width),
					static_cast<uint32_t>(height)
				};

				actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
				actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

				return actual_extent;
			}
		}

		void create_logical_device()
		{
			QueueFamilyIndices indices = find_queue_families(physical_device);

			std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
			std::set<uint32_t> unique_queue_families = { indices.graphics_family.value(), indices.present_family.value() };

			for (uint32_t queueFamily : unique_queue_families) {
				VkDeviceQueueCreateInfo queue_create_info = {};
				queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queue_create_info.queueFamilyIndex = queueFamily;
				queue_create_info.queueCount = 1;
				queue_create_info.pQueuePriorities = &QUEUE_PRIORITY;
				queue_create_infos.push_back(queue_create_info);
			}

			VkPhysicalDeviceFeatures device_features = {};
			device_features.fillModeNonSolid = true;
			device_features.samplerAnisotropy = VK_TRUE;
			device_features.sampleRateShading = VK_TRUE;

			VkDeviceCreateInfo create_info = {};
			create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
			create_info.pQueueCreateInfos = queue_create_infos.data();
			create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
			create_info.pEnabledFeatures = &device_features;
			create_info.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
			create_info.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

			if (enable_validation_layers) {
				create_info.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
				create_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();
			}
			else {
				create_info.enabledLayerCount = 0;
			}

			CHECK_VK(vkCreateDevice(physical_device, &create_info, nullptr, &logical_device), "Could not create vulkan logical device");

			vkGetDeviceQueue(logical_device, indices.graphics_family.value(), 0, &graphics_queue);
			vkGetDeviceQueue(logical_device, indices.present_family.value(), 0, &present_queue);
		}

		void create_swap_chain()
		{
			VkSwapchainKHR old_swap_chain = swap_chain;
			SwapChainSupportDetails swap_chain_support = query_swap_chain_support(physical_device);

			VkSurfaceFormatKHR surface_format = choose_swap_surface_format(swap_chain_support.formats);
			VkPresentModeKHR present_mode = choose_swap_present_mode(swap_chain_support.present_modes);
			VkExtent2D extent = choose_swap_extent(swap_chain_support.capabilities);

			uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
			if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount) {
				image_count = swap_chain_support.capabilities.maxImageCount;
			}

			VkSwapchainCreateInfoKHR create_info = {};
			create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			create_info.surface = surface;
			create_info.minImageCount = image_count;
			create_info.imageFormat = surface_format.format;
			create_info.imageColorSpace = surface_format.colorSpace;
			create_info.imageExtent = extent;
			create_info.imageArrayLayers = 1;
			create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

			QueueFamilyIndices indices = find_queue_families(physical_device);
			uint32_t queueFamilyIndices[] = { indices.graphics_family.value(), indices.present_family.value() };

			if (indices.graphics_family != indices.present_family) {
				create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
				create_info.queueFamilyIndexCount = 2;
				create_info.pQueueFamilyIndices = queueFamilyIndices;
			}
			else {
				create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
				create_info.queueFamilyIndexCount = 0; // Optional
				create_info.pQueueFamilyIndices = nullptr; // Optional
			}

			create_info.preTransform = swap_chain_support.capabilities.currentTransform;
			create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			create_info.presentMode = present_mode;
			create_info.clipped = VK_TRUE;
			create_info.oldSwapchain = old_swap_chain;

			CHECK_VK(vkCreateSwapchainKHR(logical_device, &create_info, nullptr, &swap_chain), "Could not create vulkan swap chain");

			if (old_swap_chain != VK_NULL_HANDLE)
				vkDestroySwapchainKHR(logical_device, old_swap_chain, nullptr);

			CHECK_VK(vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count, nullptr), "Could not get vulkan swap chain images");
			swap_chain_images.resize(image_count);
			CHECK_VK(vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count, swap_chain_images.data()), "Could not get vulkan swap chain images");

			swap_chain_image_format = surface_format.format;
			swap_chain_extent = extent;

			cam.aspect = static_cast<float>(extent.width) / extent.height;
		}

		void create_image_views()
		{
			swap_chain_image_views.resize(swap_chain_images.size());

			for (size_t i = 0; i < swap_chain_images.size(); i++) {
				swap_chain_image_views[i] = create_image_view(swap_chain_images[i], swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT, 1);
			}

		}

		static std::vector<char> read_file(const std::string& filename) {
			std::ifstream file(filename, std::ios::ate | std::ios::binary);

			if (!file.is_open()) {
				throw std::runtime_error("Failed to open file " + filename);
			}

			size_t file_size = (size_t)file.tellg();
			std::vector<char> buffer(file_size);

			file.seekg(0);
			file.read(buffer.data(), file_size);

			file.close();

			return buffer;
		}

		VkShaderModule create_shader_module(const std::vector<uint32_t>& shader_bin)
		{
			VkShaderModuleCreateInfo create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			create_info.codeSize = shader_bin.size() * sizeof(uint32_t);
			create_info.pCode = shader_bin.data();

			VkShaderModule shader_module;
			CHECK_VK(vkCreateShaderModule(logical_device, &create_info, nullptr, &shader_module), "Could not create vulkan shader module");

			return shader_module;
		}

		void create_render_pass()
		{
			// Create color attachment with sample count from msaa_samples
			VkAttachmentDescription color_attachment = {};
			color_attachment.format = swap_chain_image_format;
			color_attachment.samples = msaa_samples;
			color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			// Create depth attachment with sample count from msaa_samples
			VkAttachmentDescription depth_attachment = {};
			depth_attachment.format = find_depth_format();
			depth_attachment.samples = msaa_samples;
			depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			// Create a color attachment resolve with sample count of 1 and VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
			// Need to resolve the color image to the framebuffer size from MSAA color attachment
			VkAttachmentDescription color_attachment_resolve = {};
			color_attachment_resolve.format = swap_chain_image_format;
			color_attachment_resolve.samples = VK_SAMPLE_COUNT_1_BIT;
			color_attachment_resolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			color_attachment_resolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			color_attachment_resolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			color_attachment_resolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			color_attachment_resolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			color_attachment_resolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			VkAttachmentReference color_attachment_ref = {};
			color_attachment_ref.attachment = 0;
			color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkAttachmentReference depth_attachment_ref = {};
			depth_attachment_ref.attachment = 1;
			depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkAttachmentReference color_attachment_resolve_ref = {};
			color_attachment_resolve_ref.attachment = 2;
			color_attachment_resolve_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpass = {};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = 1;
			subpass.pColorAttachments = &color_attachment_ref;
			subpass.pDepthStencilAttachment = &depth_attachment_ref;
			subpass.pResolveAttachments = &color_attachment_resolve_ref;

			VkSubpassDependency dependency = {};
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			std::array<VkAttachmentDescription, 3> attachments = { color_attachment, depth_attachment, color_attachment_resolve };

			VkRenderPassCreateInfo render_pass_info = {};
			render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
			render_pass_info.pAttachments = attachments.data();
			render_pass_info.subpassCount = 1;
			render_pass_info.pSubpasses = &subpass;
			render_pass_info.dependencyCount = 1;
			render_pass_info.pDependencies = &dependency;

			CHECK_VK(vkCreateRenderPass(logical_device, &render_pass_info, nullptr, &render_pass), "Could not create vulkan render pass");
		}

		void create_descriptor_set_layout()
		{
			VkDescriptorSetLayoutBinding ubo_layout_binding = {};
			ubo_layout_binding.binding = 0;
			ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			ubo_layout_binding.descriptorCount = 1;
			ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
			ubo_layout_binding.pImmutableSamplers = nullptr; // Optional

			VkDescriptorSetLayoutBinding sampler_layout_binding = {};
			sampler_layout_binding.binding = 1;
			sampler_layout_binding.descriptorCount = 1;
			sampler_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			sampler_layout_binding.pImmutableSamplers = nullptr;
			sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			std::array<VkDescriptorSetLayoutBinding, 2> bindings = { ubo_layout_binding, sampler_layout_binding };
			VkDescriptorSetLayoutCreateInfo layout_info = {};
			layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layout_info.bindingCount = static_cast<uint32_t>(bindings.size());;
			layout_info.pBindings = bindings.data();

			CHECK_VK(vkCreateDescriptorSetLayout(logical_device, &layout_info, nullptr, &descriptor_set_layout), "Could not create vulkan descriptor set layout");
		}

		void create_graphics_pipeline()
		{
			const auto & vert_shader = read_file(VERT_SHADER_FILENAME);
			const auto & frag_shader = read_file(FRAG_SHADER_FILENAME);

			shaderc::CompileOptions options{};
			shaderc::Compiler compiler{};

			// Compile vert shader
			auto compilation_result = compiler.CompileGlslToSpv(std::string(vert_shader.begin(), vert_shader.end()), shaderc_shader_kind::shaderc_glsl_vertex_shader, VERT_SHADER_FILENAME, options);
			if (compilation_result.GetCompilationStatus() != shaderc_compilation_status_success)
				throw std::runtime_error("Could not compile shader file '" + std::string(VERT_SHADER_FILENAME) + "': " + compilation_result.GetErrorMessage());

			const std::vector<uint32_t> vert_bin(compilation_result.begin(), compilation_result.end());

			// Create vert shader module
			vert_shader_module = create_shader_module(vert_bin);

			// Compile frag shader
			compilation_result = compiler.CompileGlslToSpv(std::string(frag_shader.begin(), frag_shader.end()), shaderc_shader_kind::shaderc_glsl_fragment_shader, FRAG_SHADER_FILENAME, options);
			if (compilation_result.GetCompilationStatus() != shaderc_compilation_status_success)
				throw std::runtime_error("Could not compile shader file '" + std::string(VERT_SHADER_FILENAME) + "': " + compilation_result.GetErrorMessage());

			const std::vector<uint32_t> frag_bin(compilation_result.begin(), compilation_result.end());

			// Create frag shader module
			frag_shader_module = create_shader_module(frag_bin);

			// Ready to create the pipeline
			VkPipelineShaderStageCreateInfo vert_shader_stage_info = {};
			vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
			vert_shader_stage_info.module = vert_shader_module;
			vert_shader_stage_info.pName = "main";

			VkPipelineShaderStageCreateInfo frag_shader_stage_info = {};
			frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			frag_shader_stage_info.module = frag_shader_module;
			frag_shader_stage_info.pName = "main";

			VkPipelineShaderStageCreateInfo shader_stages[] = { vert_shader_stage_info, frag_shader_stage_info };

			const auto & vertex_binding_description = get_vertex_binding_description();
			const auto & vertex_attribute_description = get_vertex_attribute_description();

			// Create vertex input layout
			VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
			vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertex_input_info.vertexBindingDescriptionCount = 1;
			vertex_input_info.pVertexBindingDescriptions = &vertex_binding_description; // Optional
			vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_attribute_description.size());
			vertex_input_info.pVertexAttributeDescriptions = vertex_attribute_description.data(); // Optional

			// Create input assembly
			VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
			input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			input_assembly.primitiveRestartEnable = VK_FALSE;

			// Create viewport
			VkViewport viewport = {};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = static_cast<float>(swap_chain_extent.width);
			viewport.height = static_cast<float>(swap_chain_extent.height);
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			// Create scissor rectangle (same as viewport)
			VkRect2D scissor = {};
			scissor.offset = { 0, 0 };
			scissor.extent = swap_chain_extent;

			VkPipelineViewportStateCreateInfo viewport_state = {};
			viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewport_state.viewportCount = 1;
			viewport_state.pViewports = &viewport;
			viewport_state.scissorCount = 1;
			viewport_state.pScissors = &scissor;

			// Create rasterizer
			VkPipelineRasterizationStateCreateInfo rasterizer = {};
			rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterizer.depthClampEnable = VK_FALSE;
			rasterizer.rasterizerDiscardEnable = VK_FALSE;
			rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
			rasterizer.lineWidth = 1.0f;
			rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
			rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
			rasterizer.depthBiasEnable = VK_FALSE;
			rasterizer.depthBiasConstantFactor = 0.0f; // Optional
			rasterizer.depthBiasClamp = 0.0f; // Optional
			rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

			// Multi-sampling
			VkPipelineMultisampleStateCreateInfo multisampling = {};
			multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			multisampling.sampleShadingEnable = VK_TRUE;
			multisampling.rasterizationSamples = msaa_samples;
			multisampling.minSampleShading = 1.0f; // Optional
			multisampling.pSampleMask = nullptr; // Optional
			multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
			multisampling.alphaToOneEnable = VK_FALSE; // Optional

			VkPipelineColorBlendAttachmentState color_blend_attachment = {};
			color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			color_blend_attachment.blendEnable = VK_FALSE;
			color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
			color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
			color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
			color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
			color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
			color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

			VkPipelineColorBlendStateCreateInfo color_blending = {};
			color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			color_blending.logicOpEnable = VK_FALSE;
			color_blending.logicOp = VK_LOGIC_OP_COPY; // Optional
			color_blending.attachmentCount = 1;
			color_blending.pAttachments = &color_blend_attachment;
			color_blending.blendConstants[0] = 0.0f; // Optional
			color_blending.blendConstants[1] = 0.0f; // Optional
			color_blending.blendConstants[2] = 0.0f; // Optional
			color_blending.blendConstants[3] = 0.0f; // Optional

			VkPipelineDepthStencilStateCreateInfo depth_stencil = {};
			depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
			depth_stencil.depthTestEnable = VK_TRUE;
			depth_stencil.depthWriteEnable = VK_TRUE;
			depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
			depth_stencil.depthBoundsTestEnable = VK_FALSE;
			depth_stencil.minDepthBounds = 0.0f; // Optional
			depth_stencil.maxDepthBounds = 1.0f; // Optional
			depth_stencil.stencilTestEnable = VK_FALSE;
			depth_stencil.front = {}; // Optional
			depth_stencil.back = {}; // Optional

			std::array<VkDynamicState, 2> dynamic_states = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_LINE_WIDTH
			};

			VkPipelineDynamicStateCreateInfo dynamic_state = {};
			dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
			dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
			dynamic_state.pDynamicStates = dynamic_states.data();

			VkPipelineLayoutCreateInfo pipeline_layout_info = {};
			pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipeline_layout_info.setLayoutCount = 1; // Optional
			pipeline_layout_info.pSetLayouts = &descriptor_set_layout; // Optional
			pipeline_layout_info.pushConstantRangeCount = 0; // Optional
			pipeline_layout_info.pPushConstantRanges = nullptr; // Optional

			CHECK_VK(vkCreatePipelineLayout(logical_device, &pipeline_layout_info, nullptr, &pipeline_layout), "Could not create vulkan pipeline layout");

			VkGraphicsPipelineCreateInfo pipeline_info = {};
			pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			pipeline_info.stageCount = 2;
			pipeline_info.pStages = shader_stages;
			pipeline_info.pVertexInputState = &vertex_input_info;
			pipeline_info.pInputAssemblyState = &input_assembly;
			pipeline_info.pViewportState = &viewport_state;
			pipeline_info.pRasterizationState = &rasterizer;
			pipeline_info.pMultisampleState = &multisampling;
			pipeline_info.pDepthStencilState = &depth_stencil; // Optional
			pipeline_info.pColorBlendState = &color_blending;
			pipeline_info.pDynamicState = &dynamic_state; // Optional
			pipeline_info.layout = pipeline_layout;
			pipeline_info.renderPass = render_pass;
			pipeline_info.subpass = 0;
			pipeline_info.basePipelineHandle = VK_NULL_HANDLE; // Optional
			pipeline_info.basePipelineIndex = -1; // Optional

			CHECK_VK(vkCreateGraphicsPipelines(logical_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline), "Could not create vulkan graphics pipeline");

			vkDestroyShaderModule(logical_device, frag_shader_module, nullptr);
			vkDestroyShaderModule(logical_device, vert_shader_module, nullptr);
		}

		void create_framebuffers()
		{
			swap_chain_framebuffers.resize(swap_chain_image_views.size());

			for (size_t i = 0; i < swap_chain_image_views.size(); i++) {

				std::array<VkImageView, 3> attachments = { color_image_view, depth_image_view, swap_chain_image_views[i] };

				VkFramebufferCreateInfo framebuffer_info = {};
				framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebuffer_info.renderPass = render_pass;
				framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());;
				framebuffer_info.pAttachments = attachments.data();
				framebuffer_info.width = swap_chain_extent.width;
				framebuffer_info.height = swap_chain_extent.height;
				framebuffer_info.layers = 1;

				CHECK_VK(vkCreateFramebuffer(logical_device, &framebuffer_info, nullptr, &swap_chain_framebuffers[i]), "Could not create vulkan framebuffers");

			}
		}

		void create_command_pool()
		{
			QueueFamilyIndices queue_family_indices = find_queue_families(physical_device);

			VkCommandPoolCreateInfo pool_info = {};
			pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			pool_info.queueFamilyIndex = queue_family_indices.graphics_family.value();
			pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Optional

			CHECK_VK(vkCreateCommandPool(logical_device, &pool_info, nullptr, &command_pool), "Could not create vulkan command pool");
		}

		VkCommandBuffer begin_single_time_commands() {
			VkCommandBufferAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			alloc_info.commandPool = command_pool;
			alloc_info.commandBufferCount = 1;

			VkCommandBuffer command_buffer;
			vkAllocateCommandBuffers(logical_device, &alloc_info, &command_buffer);

			VkCommandBufferBeginInfo begin_info = {};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(command_buffer, &begin_info);

			return command_buffer;
		}

		void end_single_time_commands(VkCommandBuffer command_buffer) {
			vkEndCommandBuffer(command_buffer);

			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &command_buffer;

			vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphics_queue);

			vkFreeCommandBuffers(logical_device, command_pool, 1, &command_buffer);
		}

		void create_buffer(VkDeviceSize mem_size, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags mem_properties, VkBuffer& buffer, VkDeviceMemory& device_memory)
		{
			VkBufferCreateInfo buffer_info = {};
			buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			buffer_info.size = mem_size;
			buffer_info.usage = usage_flags;
			buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			CHECK_VK(vkCreateBuffer(logical_device, &buffer_info, nullptr, &buffer), "Could not create vulkan vertex buffer");

			VkMemoryRequirements mem_requirements = {};
			vkGetBufferMemoryRequirements(logical_device, buffer, &mem_requirements);

			VkMemoryAllocateInfo allocInfo = {};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = mem_requirements.size;
			allocInfo.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, mem_properties);

			CHECK_VK(vkAllocateMemory(logical_device, &allocInfo, nullptr, &device_memory), "Could not allocate vulkan physical device memory for vertex buffer");

			vkBindBufferMemory(logical_device, buffer, device_memory, 0);
		}

		void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {

			VkCommandBuffer command_buffer = begin_single_time_commands();

			VkBufferCopy copy_region = {};
			copy_region.srcOffset = 0; // Optional
			copy_region.dstOffset = 0; // Optional
			copy_region.size = size;
			vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

			end_single_time_commands(command_buffer);
		}

		void create_image(uint32_t width, uint32_t height, uint32_t mip_levels, VkSampleCountFlagBits num_samples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& image_memory) {
			VkImageCreateInfo image_info = {};
			image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			image_info.imageType = VK_IMAGE_TYPE_2D;
			image_info.extent.width = width;
			image_info.extent.height = height;
			image_info.extent.depth = 1;
			image_info.mipLevels = mip_levels;
			image_info.arrayLayers = 1;
			image_info.format = format;
			image_info.tiling = tiling;
			image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			image_info.usage = usage;
			image_info.samples = num_samples;
			image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			CHECK_VK(vkCreateImage(logical_device, &image_info, nullptr, &image), "Could not create vulkan image");

			VkMemoryRequirements mem_requirements;
			vkGetImageMemoryRequirements(logical_device, image, &mem_requirements);

			VkMemoryAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			alloc_info.allocationSize = mem_requirements.size;
			alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, properties);

			CHECK_VK(vkAllocateMemory(logical_device, &alloc_info, nullptr, &image_memory), "Could not allocate vulkan image memory");

			vkBindImageMemory(logical_device, image, image_memory, 0);
		}

		void transition_image_layout(VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout, uint32_t mip_levels) {
			VkCommandBuffer command_buffer = begin_single_time_commands();
			VkImageMemoryBarrier barrier = {};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = old_layout;
			barrier.newLayout = new_layout;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = image;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = mip_levels;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;

			VkPipelineStageFlags source_stage;
			VkPipelineStageFlags destination_stage;

			if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
				barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

				if (has_stencil_component(format)) {
					barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
				}
			}
			else {
				barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			}

			if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

				source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			}
			else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
				destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			}
			else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

				source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			}
			else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
				source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destination_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			}
			else {
				throw std::invalid_argument("unsupported layout transition!");
			}

			vkCmdPipelineBarrier(
				command_buffer,
				source_stage, destination_stage,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);

			end_single_time_commands(command_buffer);
		}

		void copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
			VkCommandBuffer command_buffer = begin_single_time_commands();

			VkBufferImageCopy region = {};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;

			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount = 1;

			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = {
				width,
				height,
				1
			};

			vkCmdCopyBufferToImage(
				command_buffer,
				buffer,
				image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&region
			);

			end_single_time_commands(command_buffer);
		}

		void generate_mipmaps(VkImage image, VkFormat image_format, int32_t tex_width, int32_t tex_height, uint32_t mip_levels) {

			// Check if image format supports linear blitting
			VkFormatProperties format_properties = {};
			vkGetPhysicalDeviceFormatProperties(physical_device, image_format, &format_properties);
			if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
				throw std::runtime_error("vulkan device does does not support linear blitting");
			}

			VkCommandBuffer command_buffer = begin_single_time_commands();

			VkImageMemoryBarrier barrier = {};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.image = image;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.subresourceRange.levelCount = 1;

			int32_t mip_width = tex_width;
			int32_t mip_height = tex_height;

			for (uint32_t i = 1; i < mip_levels; i++) {
				barrier.subresourceRange.baseMipLevel = i - 1;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

				vkCmdPipelineBarrier(command_buffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
					0, nullptr,
					0, nullptr,
					1, &barrier);

				VkImageBlit blit = {};
				blit.srcOffsets[0] = { 0, 0, 0 };
				blit.srcOffsets[1] = { mip_width, mip_height, 1 };
				blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.srcSubresource.mipLevel = i - 1;
				blit.srcSubresource.baseArrayLayer = 0;
				blit.srcSubresource.layerCount = 1;
				blit.dstOffsets[0] = { 0, 0, 0 };
				blit.dstOffsets[1] = { mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1 };
				blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.dstSubresource.mipLevel = i;
				blit.dstSubresource.baseArrayLayer = 0;
				blit.dstSubresource.layerCount = 1;

				vkCmdBlitImage(command_buffer,
					image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1, &blit,
					VK_FILTER_LINEAR);

				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				vkCmdPipelineBarrier(command_buffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
					0, nullptr,
					0, nullptr,
					1, &barrier);

				if (mip_width > 1) mip_width /= 2;
				if (mip_height > 1) mip_height /= 2;
			}

			barrier.subresourceRange.baseMipLevel = mip_levels - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(command_buffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			end_single_time_commands(command_buffer);
		}



		std::pair<VkImage, VkDeviceMemory> create_texture_image(std::string texture_file_path, int& tex_width, int& tex_height, int& num_chan, size_t& image_size, uint32_t mip_levels, VkBuffer staging_buffer = VK_NULL_HANDLE, VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE)
		{
			std::vector<uint8_t> pixels = utils::load_image_file_to_memory(texture_file_path, tex_width, tex_height, num_chan, image_size);
			return create_texture_image(pixels, tex_width, tex_height, num_chan, image_size, mip_levels, staging_buffer, staging_buffer_memory);
		}

		std::pair<VkImage, VkDeviceMemory> create_texture_image(std::vector<uint8_t> pixels, const int& tex_width, const int& tex_height, const int& num_chan, const size_t& image_size, const uint32_t mip_levels, VkBuffer staging_buffer = VK_NULL_HANDLE, VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE)
		{
			if ((staging_buffer && !staging_buffer_memory) || (!staging_buffer && staging_buffer_memory))
				throw std::invalid_argument("Don't know what to do when staging_buffer and staging_buffer_memory are not consistently null or existing");

			VkImage texture_image{ VK_NULL_HANDLE };
			VkDeviceMemory texture_image_memory{ VK_NULL_HANDLE };
			bool should_cleanup_staging{ false };

			if (!staging_buffer)
			{
				create_buffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);
				should_cleanup_staging = true;
			}

			void* data;
			vkMapMemory(logical_device, staging_buffer_memory, 0, image_size, 0, &data);
			memcpy(data, pixels.data(), static_cast<size_t>(image_size));
			vkUnmapMemory(logical_device, staging_buffer_memory);

			create_image(tex_width, tex_height, mip_levels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture_image, texture_image_memory);

			transition_image_layout(texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mip_levels);

			copy_buffer_to_image(staging_buffer, texture_image, static_cast<uint32_t>(tex_width), static_cast<uint32_t>(tex_height));

			if (should_cleanup_staging)
			{
				vkDestroyBuffer(logical_device, staging_buffer, nullptr);
				vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
			}

			generate_mipmaps(texture_image, VK_FORMAT_R8G8B8A8_UNORM, tex_width, tex_height, mip_levels);

			return { texture_image, texture_image_memory };
		}

		VkImageView create_image_view(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags, uint32_t mip_levels) {
			VkImageViewCreateInfo view_info = {};
			view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			view_info.image = image;
			view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
			view_info.format = format;
			view_info.subresourceRange.aspectMask = aspect_flags;
			view_info.subresourceRange.baseMipLevel = 0;
			view_info.subresourceRange.levelCount = mip_levels;
			view_info.subresourceRange.baseArrayLayer = 0;
			view_info.subresourceRange.layerCount = 1;

			VkImageView image_view;
			CHECK_VK(vkCreateImageView(logical_device, &view_info, nullptr, &image_view), "Could not create vulkan image view");
			return image_view;
		}

		//VkImageView create_texture_image_view(VkImage texture_image)
		//{
		//	return create_image_view(texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
		//}

		std::tuple<VkImageView, VkImage, VkDeviceMemory> create_texture_image_with_view(std::string texture_file_path, int& tex_width, int& tex_height, int& num_chan, size_t &image_size, VkBuffer staging_buffer = VK_NULL_HANDLE, VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE)
		{
			uint32_t mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(tex_width, tex_height)))) + 1;
			auto texture = create_texture_image(texture_file_path, tex_width, tex_height, num_chan, image_size, mip_levels, staging_buffer, staging_buffer_memory);
			auto view = create_image_view(texture.first, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, mip_levels);
			return { view, texture.first, texture.second };
		}

		VkSampler create_texture_sampler(uint32_t mip_levels)
		{
			VkSampler texture_sampler = { VK_NULL_HANDLE };
			VkSamplerCreateInfo sampler_info = {};
			sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			sampler_info.magFilter = VK_FILTER_LINEAR;
			sampler_info.minFilter = VK_FILTER_LINEAR;
			sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			sampler_info.anisotropyEnable = VK_TRUE;
			sampler_info.maxAnisotropy = 16;
			sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			sampler_info.unnormalizedCoordinates = VK_FALSE;
			sampler_info.compareEnable = VK_FALSE;
			sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
			sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			sampler_info.mipLodBias = 0.0f;
			sampler_info.minLod = 0.0f;
			sampler_info.maxLod = static_cast<float>(mip_levels);

			CHECK_VK(vkCreateSampler(logical_device, &sampler_info, nullptr, &texture_sampler), "Could not create vulkan texture sampler");

			return texture_sampler;
		}

		template<class T>
		std::pair <VkBuffer, VkDeviceMemory> create_vertex_buffer(const std::vector<T> &src, VkBuffer staging_buffer = VK_NULL_HANDLE, VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE)
		{
			if ((staging_buffer && !staging_buffer_memory) || (!staging_buffer && staging_buffer_memory))
				throw std::invalid_argument("Don't know what to do when staging_buffer and staging_buffer_memory are not consistently null or existing");

			bool should_cleanup_staging{ false };
			VkDeviceSize buffer_size = sizeof(src[0]) * src.size();

			if (!staging_buffer)
			{
				create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);
				should_cleanup_staging = true;
			}

			void* data;
			vkMapMemory(logical_device, staging_buffer_memory, 0, buffer_size, 0, &data);
			memcpy(data, src.data(), (size_t)buffer_size);
			vkUnmapMemory(logical_device, staging_buffer_memory);

			VkDeviceMemory buffer_memory{ VK_NULL_HANDLE };
			VkBuffer buffer{ VK_NULL_HANDLE };

			create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, buffer_memory);

			copy_buffer(staging_buffer, buffer, buffer_size);

			if (should_cleanup_staging)
			{
				vkDestroyBuffer(logical_device, staging_buffer, nullptr);
				vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
			}

			return { buffer, buffer_memory };
		}

		template<class T>
		std::pair <VkBuffer, VkDeviceMemory> create_index_buffer(const std::vector<T> &src, VkBuffer staging_buffer = VK_NULL_HANDLE, VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE)
		{
			if ((staging_buffer && !staging_buffer_memory) || (!staging_buffer && staging_buffer_memory))
				throw std::invalid_argument("Don't know what to do when staging_buffer and staging_buffer_memory are not consistently null or existing");

			bool should_cleanup_staging{ false };
			VkDeviceSize buffer_size = sizeof(src[0]) * src.size();

			if (!staging_buffer)
			{
				create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);
				should_cleanup_staging = true;
			}

			void* data;
			vkMapMemory(logical_device, staging_buffer_memory, 0, buffer_size, 0, &data);
			memcpy(data, src.data(), (size_t)buffer_size);
			vkUnmapMemory(logical_device, staging_buffer_memory);

			VkDeviceMemory buffer_memory{ VK_NULL_HANDLE };
			VkBuffer buffer{ VK_NULL_HANDLE };

			create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, buffer_memory);

			copy_buffer(staging_buffer, buffer, buffer_size);

			if (should_cleanup_staging)
			{
				vkDestroyBuffer(logical_device, staging_buffer, nullptr);
				vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
			}

			return { buffer, buffer_memory };
		}

		std::pair<std::vector<VkBuffer>,std::vector<VkDeviceMemory>> create_uniform_buffer() {
			VkDeviceSize buffer_size = sizeof(uniform_buffer_object);

			std::vector<VkBuffer> uniform_buffers(swap_chain_images.size());
			std::vector<VkDeviceMemory> uniform_buffer_memories(swap_chain_images.size());

			for (size_t i = 0; i < swap_chain_images.size(); i++) {
				create_buffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniform_buffers[i], uniform_buffer_memories[i]);
			}

			return { uniform_buffers, uniform_buffer_memories };
		}

		VkFormat find_supported_format(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
		{
			for (VkFormat format : candidates) {
				VkFormatProperties props;
				vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

				if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
					return format;
				}
				else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
					return format;
				}
			}

			throw std::runtime_error("could not find vulkan supported format");
		}

		VkFormat find_depth_format() {
			return find_supported_format(
				{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
				VK_IMAGE_TILING_OPTIMAL,
				VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
			);
		}

		bool has_stencil_component(VkFormat format) {
			return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
		}

		void create_color_resources()
		{
			VkFormat color_format = swap_chain_image_format;

			create_image(swap_chain_extent.width, swap_chain_extent.height, 1, msaa_samples, color_format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, color_image, color_image_memory);
			color_image_view = create_image_view(color_image, color_format, VK_IMAGE_ASPECT_COLOR_BIT, 1);

			transition_image_layout(color_image, color_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1);
		}

		void create_depth_resources()
		{
			VkFormat depth_format = find_depth_format();
			create_image(swap_chain_extent.width, swap_chain_extent.height, 1, msaa_samples, depth_format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth_image, depth_image_memory);
			depth_image_view = create_image_view(depth_image, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
			transition_image_layout(depth_image, depth_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
		}

		void create_descriptor_pool() {

			VkPhysicalDeviceProperties physical_device_properties{};
			vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

			std::array<VkDescriptorPoolSize, 2> pool_sizes = {};
			pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			pool_sizes[0].descriptorCount = static_cast<uint32_t>(256);
			pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			pool_sizes[1].descriptorCount = static_cast<uint32_t>(256);

			VkDescriptorPoolCreateInfo pool_info = {};
			pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
			pool_info.pPoolSizes = pool_sizes.data();
			pool_info.maxSets = static_cast<uint32_t>(256);
			pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

			CHECK_VK(vkCreateDescriptorPool(logical_device, &pool_info, nullptr, &descriptor_pool), "Could not create vulkan descriptor pool");
		}

		std::vector<VkDescriptorSet> create_descriptor_sets()
		{
			std::vector<VkDescriptorSet> descriptor_sets(swap_chain_images.size());
			std::vector<VkDescriptorSetLayout> layouts(swap_chain_images.size(), descriptor_set_layout);
			VkDescriptorSetAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			alloc_info.descriptorPool = descriptor_pool;
			alloc_info.descriptorSetCount = static_cast<uint32_t>(swap_chain_images.size());
			alloc_info.pSetLayouts = layouts.data();

			CHECK_VK(vkAllocateDescriptorSets(logical_device, &alloc_info, descriptor_sets.data()), "Could not allocate vulkan descriptor sets");

			return descriptor_sets;
		}

		void update_descriptor_sets(size_t model_index)
		{
			auto & m = model_draw_queue[model_index];
			for (size_t i = 0; i < swap_chain_images.size(); i++) {

				VkDescriptorBufferInfo buffer_info = {};
				buffer_info.buffer = m->uniform_buffers[i];
				buffer_info.offset = 0;
				buffer_info.range = sizeof(uniform_buffer_object);

				VkDescriptorImageInfo image_info = {};
				image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				image_info.imageView = model_draw_queue[model_index]->texture_image_view;
				image_info.sampler = model_draw_queue[model_index]->texture_sampler;

				std::array<VkWriteDescriptorSet, 2> descriptor_writes = {};

				descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptor_writes[0].dstSet = m->descriptor_sets[i];
				descriptor_writes[0].dstBinding = 0;
				descriptor_writes[0].dstArrayElement = 0;
				descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptor_writes[0].descriptorCount = 1;
				descriptor_writes[0].pBufferInfo = &buffer_info;

				descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptor_writes[1].dstSet = m->descriptor_sets[i];
				descriptor_writes[1].dstBinding = 1;
				descriptor_writes[1].dstArrayElement = 0;
				descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptor_writes[1].descriptorCount = 1;
				descriptor_writes[1].pImageInfo = &image_info;

				vkUpdateDescriptorSets(logical_device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
			}
		}

		void fill_command_buffers()
		{
			for (size_t i = 0; i < command_buffers.size(); i++) {
				VkCommandBufferBeginInfo begin_info = {};
				begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
				begin_info.pInheritanceInfo = nullptr; // Optional

				CHECK_VK(vkBeginCommandBuffer(command_buffers[i], &begin_info), "Could not begin recording of vulkan command buffer " + std::to_string(i));

				VkViewport viewport = {};
				viewport.height = static_cast<float>(swap_chain_extent.height);
				viewport.width = static_cast<float>(swap_chain_extent.width);
				viewport.x = 0;
				viewport.y = 0;
				viewport.minDepth = 0.0f;
				viewport.maxDepth = 1.0f;
				vkCmdSetViewport(command_buffers[i], 0, 1, &viewport);

				std::array<VkClearValue, 2> clear_values = {};
				clear_values[0].color = { 0.05f, 0.05f, 0.05f, 0.8f };
				clear_values[1].depthStencil = { 1.0f, 0 };

				VkRenderPassBeginInfo render_pass_info = {};
				render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				render_pass_info.renderPass = render_pass;
				render_pass_info.framebuffer = swap_chain_framebuffers[i];
				render_pass_info.renderArea.offset = { 0, 0 };
				render_pass_info.renderArea.extent = swap_chain_extent;
				render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
				render_pass_info.pClearValues = clear_values.data();

				vkCmdBeginRenderPass(command_buffers[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
				vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

				for (size_t m = 0; m < model_draw_queue.size(); m++)
				{
					VkBuffer vertex_buffers[] { model_draw_queue[m]->vertex_buffer };
					VkDeviceSize offsets[] { 0 };

					vkCmdBindVertexBuffers(command_buffers[i], 0, 1, vertex_buffers, offsets);

					//vkCmdDraw(command_buffers[i], static_cast<uint32_t>(VERTICES.size()), 1, 0, 0);
					vkCmdBindDescriptorSets(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &model_draw_queue[m]->descriptor_sets[i], 0, nullptr);

					vkCmdBindIndexBuffer(command_buffers[i], model_draw_queue[m]->index_buffer, 0, VK_INDEX_TYPE_UINT32);
					vkCmdDrawIndexed(command_buffers[i], static_cast<uint32_t>(model_draw_queue[m]->indices.size()), 1, 0, 0, 0);
				}

				vkCmdEndRenderPass(command_buffers[i]);

				CHECK_VK(vkEndCommandBuffer(command_buffers[i]), "Could not end vulkan command buffer " + std::to_string(i));
			}
		}

		void reset_command_buffers()
		{
			for (size_t i = 0; i < command_buffers.size(); i++) {
				CHECK_VK(vkResetCommandBuffer(command_buffers[i], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT), "Could not reset vulkan command buffer");
			}
		}

		void create_command_buffers()
		{
			command_buffers.resize(swap_chain_framebuffers.size());

			VkCommandBufferAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			alloc_info.commandPool = command_pool;
			alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			alloc_info.commandBufferCount = (uint32_t)command_buffers.size();

			CHECK_VK(vkAllocateCommandBuffers(logical_device, &alloc_info, command_buffers.data()), "Could not allocate vulkan command buffers");
		}

		void create_sync_objects()
		{
			image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
			render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
			in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

			VkSemaphoreCreateInfo semaphore_info = {};
			semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			VkFenceCreateInfo fence_info = {};
			fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				CHECK_VK(vkCreateSemaphore(logical_device, &semaphore_info, nullptr, &image_available_semaphores[i]), "Could not create vulkan image available synchronization semaphore");
				CHECK_VK(vkCreateSemaphore(logical_device, &semaphore_info, nullptr, &render_finished_semaphores[i]), "Could not create vulkan render fininshed synchronization semaphore");
				CHECK_VK(vkCreateFence(logical_device, &fence_info, nullptr, &in_flight_fences[i]), "Could not create vulkan in flight fence");
			}
		}

		void setup_camera()
		{
			cam.position = engine_coordinate_system.get_back() * 2.0f;
			cam.position += engine_coordinate_system.get_up() * 2.0f;
			cam.position += engine_coordinate_system.get_left() * 2.0f;
			cam.look_at({ 0,0,0 });
		}

		void update_uniform_buffer(size_t model_index, uint32_t current_image)
		{
			const auto & engine_to_vk = make_transform(engine_coordinate_system, vk_coordinate_system);
			const auto & engine_to_vk_mat = float4x4{
				float4 { engine_to_vk.x, 0 },
				float4 { engine_to_vk.y, 0 },
				float4 { engine_to_vk.z, 0 },
				float4 { 0, 0, 0, 1 }
			};

			mvp.model = model_draw_queue[model_index]->transform;
			mvp.view = linalg::mul(engine_to_vk_mat, view_matrix);
			mvp.projection = projection_matrix;

			void* uniform_buffer_data;
			CHECK_VK(vkMapMemory(logical_device, model_draw_queue[model_index]->uniform_buffer_memories[current_image], 0, sizeof(mvp), 0, &uniform_buffer_data), "Could not map vulkan uniform buffer memories");
			memcpy(uniform_buffer_data, &mvp, sizeof(mvp));
			vkUnmapMemory(logical_device, model_draw_queue[model_index]->uniform_buffer_memories[current_image]);
		}

		void cleanup_swap_chain() {

			vkDestroyImageView(logical_device, depth_image_view, nullptr);
			vkDestroyImage(logical_device, depth_image, nullptr);
			vkFreeMemory(logical_device, depth_image_memory, nullptr);

			vkDestroyImageView(logical_device, color_image_view, nullptr);
			vkDestroyImage(logical_device, color_image, nullptr);
			vkFreeMemory(logical_device, color_image_memory, nullptr);

			for (size_t i = 0; i < swap_chain_framebuffers.size(); i++) {
				vkDestroyFramebuffer(logical_device, swap_chain_framebuffers[i], nullptr);
			}

			vkFreeCommandBuffers(logical_device, command_pool, static_cast<uint32_t>(command_buffers.size()), command_buffers.data());

			vkDestroyPipeline(logical_device, graphics_pipeline, nullptr);
			vkDestroyPipelineLayout(logical_device, pipeline_layout, nullptr);
			vkDestroyRenderPass(logical_device, render_pass, nullptr);

			for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
				vkDestroyImageView(logical_device, swap_chain_image_views[i], nullptr);
			}

			//vkDestroySwapchainKHR(logical_device, swap_chain, nullptr);
		}

		void recreate_swap_chain() {

			int width = 0, height = 0;
			while (width == 0 || height == 0) {
				glfwGetFramebufferSize(window, &width, &height);
				glfwWaitEvents();
			}

			vkDeviceWaitIdle(logical_device);

			cleanup_swap_chain();

			create_swap_chain();
			create_image_views();
			create_render_pass();
			create_graphics_pipeline();
			create_color_resources();
			create_depth_resources();
			create_framebuffers();
			create_command_buffers();
		}

		void cleanup_models()
		{
			for (auto model : model_cache)
			{
				for (auto mem : model->uniform_buffer_memories)
				{
					vkFreeMemory(logical_device, mem, nullptr);
				}

				for (auto buff : model->uniform_buffers)
				{
					vkDestroyBuffer(logical_device, buff, nullptr);
				}

				vkFreeDescriptorSets(logical_device, descriptor_pool, static_cast<uint32_t>(model->descriptor_sets.size()), model->descriptor_sets.data());

				vkDestroyImageView(logical_device, model->texture_image_view, nullptr);

				vkDestroyImage(logical_device, model->texture_image, nullptr);
				vkFreeMemory(logical_device, model->texture_image_memory, nullptr);

				vkDestroyBuffer(logical_device, model->index_buffer, nullptr);
				vkFreeMemory(logical_device, model->index_buffer_memory, nullptr);

				vkDestroyBuffer(logical_device, model->vertex_buffer, nullptr);
				vkFreeMemory(logical_device, model->vertex_buffer_memory, nullptr);

				vkDestroySampler(logical_device, model->texture_sampler, nullptr);
			}

			model_cache.clear();
		}

		void cleanup_descriptor_pool()
		{
			vkDestroyDescriptorPool(logical_device, descriptor_pool, nullptr);
		}

		void cleanup_descriptor_set_layout()
		{
			vkDestroyDescriptorSetLayout(logical_device, descriptor_set_layout, nullptr);
		}

		void cleanup_sync_objects()
		{
			for (size_t i = 0; i < in_flight_fences.size(); i++) {
				vkDestroyFence(logical_device, in_flight_fences[i], nullptr);
				vkDestroySemaphore(logical_device, render_finished_semaphores[i], nullptr);
				vkDestroySemaphore(logical_device, image_available_semaphores[i], nullptr);
			}
		}

		void cleanup_command_pool()
		{
			vkDestroyCommandPool(logical_device, command_pool, nullptr);
		}

		void cleanup_command_buffers()
		{
			vkFreeCommandBuffers(logical_device, command_pool, static_cast<uint32_t>(command_buffers.size()), command_buffers.data());
		}

		void cleanup_logical_device()
		{
			vkDestroyDevice(logical_device, nullptr);
		}

		void cleanup_validation_layers()
		{
			if (enable_validation_layers) {
				DestroyDebugUtilsMessengerEXT(vulkan_instance, callback, nullptr);
			}
		}

		void cleanup_surface()
		{
			vkDestroySurfaceKHR(vulkan_instance, surface, nullptr);
		}

		void cleanup_vulkan_instance()
		{
			vkDestroyInstance(vulkan_instance, nullptr);
		}
	};
}


#endif
