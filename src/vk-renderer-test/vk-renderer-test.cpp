#include "../vk-renderer/vk-renderer.hpp"

const int WINDOW_WIDTH  { 800 };
const int WINDOW_HEIGHT { 600 };
const std::string CUBE_MODEL_PATH = "../../resources/models/cube.obj";
const std::string CHALET_MODEL_PATH = "../../resources/models/chalet.obj";

const std::string STATUE_TEXTURE_PATH = "../../resources/textures/statue.jpg";
const std::string VENUS_TEXTURE_PATH = "../../resources/textures/venus.jpg";
const std::string CHALET_TEXTURE_PATH = "../../resources/textures/chalet.jpg";

int main() try
{
	vk_renderer::vk_renderer renderer;

	renderer.init_vulkan(nullptr, 300, 300, WINDOW_WIDTH, WINDOW_HEIGHT);

	auto cube_statue = vk_renderer::utils::load_obj_file_to_memory("cube statue", CUBE_MODEL_PATH, STATUE_TEXTURE_PATH, vk_renderer::coord_system{ vk_renderer::coord_axis::forward, vk_renderer::coord_axis::left, vk_renderer::coord_axis::up }, renderer.get_coordinate_system());
	auto cube_venus = vk_renderer::utils::load_obj_file_to_memory("cube venus", CUBE_MODEL_PATH, VENUS_TEXTURE_PATH, vk_renderer::coord_system{ vk_renderer::coord_axis::forward, vk_renderer::coord_axis::left, vk_renderer::coord_axis::up }, renderer.get_coordinate_system());
	auto chalet = vk_renderer::utils::load_obj_file_to_memory("chalet", CHALET_MODEL_PATH, CHALET_TEXTURE_PATH, vk_renderer::coord_system { vk_renderer::coord_axis::forward, vk_renderer::coord_axis::left, vk_renderer::coord_axis::up }, renderer.get_coordinate_system());

	while (renderer.poll_events())
	{
		static auto begin_first_frame_time = std::chrono::high_resolution_clock::now();
		auto begin_frame_time = std::chrono::high_resolution_clock::now();
        static float time_since_beginning_of_program = 0;
        
        // Rotate the cube by the amount we want
        cube_statue.transform = linalg::pose_matrix(linalg::rotation_quat(renderer.get_coordinate_system().get_up(), time_since_beginning_of_program * 90.0f * DEG_TO_RAD * 0.25f), float3{ -1.0f,0,0 });
        cube_venus.transform = linalg::pose_matrix(linalg::rotation_quat(renderer.get_coordinate_system().get_right(), time_since_beginning_of_program * 90.0f * DEG_TO_RAD * 0.25f), float3{ 0, 1.0f, -1.0f });
        chalet.transform = linalg::pose_matrix(linalg::rotation_quat(renderer.get_coordinate_system().get_down(), time_since_beginning_of_program * 90.0f * DEG_TO_RAD * 0.25f), float3{ 1.0f,0,0 });

		renderer.begin_frame();
		renderer.set_view_proj_from_camera();
		renderer.add_model(cube_statue);
		renderer.add_model(cube_venus);
		renderer.add_model(chalet);
		renderer.end_frame();
		renderer.draw_frame();

		// Calculate the time it took to draw the frame and time since the first frame was prepared
		auto current_time = std::chrono::high_resolution_clock::now();
		float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - begin_frame_time).count();
		time_since_beginning_of_program = std::chrono::duration<float, std::chrono::seconds::period>(current_time - begin_first_frame_time).count();


		std::cout << "Rendering Speed: " << 1/frame_time << " FPS\r";
	}

	renderer.cleanup();

	return EXIT_SUCCESS;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
