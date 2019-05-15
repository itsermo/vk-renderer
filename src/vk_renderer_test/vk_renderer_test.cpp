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
		renderer.begin_frame();
		renderer.add_model(cube_statue);
		renderer.end_frame();
		renderer.draw_frame();
	}

	renderer.cleanup();

	return EXIT_SUCCESS;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}