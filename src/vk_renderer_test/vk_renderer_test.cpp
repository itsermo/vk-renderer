#include "../vk_renderer/vk_renderer.hpp"

const int WINDOW_WIDTH  { 800 };
const int WINDOW_HEIGHT { 600 };
const std::string MODEL_PATH = "resources/models/cube.obj";
const std::string TEXTURE_PATH = "resources/textures/statue.jpg";

int main() try
{
	vk_renderer::vk_renderer renderer;

	renderer.init_vulkan(nullptr, 300, 300, WINDOW_WIDTH, WINDOW_HEIGHT);

	auto cube_statue = vk_renderer::utils::load_obj_file_to_memory("cube statue", MODEL_PATH, TEXTURE_PATH);

	while (renderer.poll_events())
	{
		auto start_time = std::chrono::high_resolution_clock::now();

		renderer.begin_frame();
		renderer.set_view_proj_from_camera();
		renderer.add_model(cube_statue);
		renderer.end_frame();
		renderer.draw_frame();

		auto current_time = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - start_time).count();

		std::cout << "Rendering Speed: " << 1/time << " FPS\r";
	}

	renderer.cleanup();

	return EXIT_SUCCESS;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}