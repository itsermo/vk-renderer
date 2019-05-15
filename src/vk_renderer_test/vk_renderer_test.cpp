#include "../vk_renderer/vk_renderer.hpp"

const int WINDOW_WIDTH  { 800 };
const int WINDOW_HEIGHT { 600 };
const std::string MODEL_PATH = "resources/models/cube.obj";
const std::string TEXTURE_PATH = "resources/textures/statue.jpg";

int main() try
{
	vk_renderer::vk_renderer renderer;

	renderer.init_vulkan(nullptr, 300, 300, 1024, 768);

	auto cube_statue = vk_renderer::utils::load_obj_file_to_memory("cube venus", "resources/models/cube.obj", "resources/textures/statue.jpg");
	auto cube_venus = vk_renderer::utils::load_obj_file_to_memory("cube venus", "resources/models/cube.obj", "resources/textures/venus.jpg");

	while (renderer.poll_events())
	{
		renderer.begin_frame();
		renderer.add_model(cube_statue);
		renderer.add_model(cube_venus);
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