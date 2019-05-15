#include "../vk_renderer/vk_renderer.hpp"

const int WINDOW_WIDTH  { 800 };
const int WINDOW_HEIGHT { 600 };
const std::string MODEL_PATH = "resources/models/cube.obj";
const std::string TEXTURE_PATH = "resources/textures/statue.jpg";

int main() try
{
	vk_renderer::vk_renderer renderer;

	//renderer.init_window(WINDOW_WIDTH, WINDOW_HEIGHT);
	renderer.init_vulkan(nullptr, 300, 300, 1024, 768);

	while (true)
	{
		renderer.begin_frame();
		renderer.end_frame();
		renderer.swap_buffers();
	}

	// while true
	//   begin frame
	//   begin shader
	//   add model
	//   end shader
	//   end frame
	//   swap buffer
	
	// destroy window
	// destroy context

	//renderer.run();

	renderer.cleanup();

	return EXIT_SUCCESS;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}