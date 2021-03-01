#include "../vk-renderer/vk-renderer.hpp"

const int WINDOW_WIDTH  { 800 };
const int WINDOW_HEIGHT { 600 };

const std::string VIKING_ROOM_MODEL_PATH = "../../resources/models/viking-room.obj";

const std::string VIKING_ROOM_TEXTURE_PATH = "../../resources/textures/viking-room.png";

int main() try
{
	vk_renderer::vk_renderer renderer;

	renderer.init_vulkan(nullptr, 300, 300, WINDOW_WIDTH, WINDOW_HEIGHT);

	auto viking_room = vk_renderer::utils::load_obj_file_to_memory("viking-room", VIKING_ROOM_MODEL_PATH, VIKING_ROOM_TEXTURE_PATH, vk_renderer::coord_system { vk_renderer::coord_axis::forward, vk_renderer::coord_axis::left, vk_renderer::coord_axis::up }, renderer.get_coordinate_system());

	while (renderer.poll_events())
	{
		static auto begin_first_frame_time = std::chrono::high_resolution_clock::now();
		auto begin_frame_time = std::chrono::high_resolution_clock::now();
        static float time_since_beginning_of_program = 0;
        
        viking_room.transform = linalg::pose_matrix(linalg::rotation_quat(renderer.get_coordinate_system().get_down(), time_since_beginning_of_program * 90.0f * DEG_TO_RAD * 0.25f), float3{ 0,0,0 });

		renderer.begin_frame();
		renderer.set_view_proj_from_camera();
		renderer.add_model(viking_room);
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
