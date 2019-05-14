#include "../vk_renderer/vk_renderer.hpp"


int main() try
{
	vk_renderer app;

	app.run();

	return EXIT_SUCCESS;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}