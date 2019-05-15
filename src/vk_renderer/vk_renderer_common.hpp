#pragma once
#ifndef VK_RENDERER_COMMON
#define VK_RENDERER_COMMON
#include <linalg.h>
#include <vector>

using namespace linalg::aliases;

namespace vk_renderer
{
	struct vertex
	{
		float3 pos;
		float3 color;
		float2 tex_coord;

		bool operator==(const vertex& other) const {
			return pos == other.pos && color == other.color && tex_coord == other.tex_coord;
		}
	};

	// A value type representing an abstract direction vector in 3D space, independent of any coordinate system
	enum class coord_axis { forward, back, left, right, up, down, north = forward, east = right, south = back, west = left };
	static constexpr float dot(coord_axis a, coord_axis b)
	{
		float table[6][6]{ {+1,-1,0,0,0,0},{-1,+1,0,0,0,0},{0,0,+1,-1,0,0},{0,0,-1,+1,0,0},{0,0,0,0,+1,-1},{0,0,0,0,-1,+1} };
		return table[static_cast<int>(a)][static_cast<int>(b)];
	}

	// A concrete 3D coordinate system with defined x, y, and z axes
	struct coord_system
	{
		coord_axis x_axis, y_axis, z_axis;
		constexpr float3 get_axis(coord_axis a) const { return { dot(x_axis, a), dot(y_axis, a), dot(z_axis, a) }; }
		constexpr float3 get_left() const { return get_axis(coord_axis::left); }
		constexpr float3 get_right() const { return get_axis(coord_axis::right); }
		constexpr float3 get_up() const { return get_axis(coord_axis::up); }
		constexpr float3 get_down() const { return get_axis(coord_axis::down); }
		constexpr float3 get_forward() const { return get_axis(coord_axis::forward); }
		constexpr float3 get_back() const { return get_axis(coord_axis::back); }
	};
	inline float3x3 make_transform(const coord_system & from, const coord_system & to) { return { to.get_axis(from.x_axis), to.get_axis(from.y_axis), to.get_axis(from.z_axis) }; }

	constexpr coord_system engine_coordinate_system{ coord_axis::right, coord_axis::up, coord_axis::back };

	struct model
	{
		std::string id{};
		std::vector<vertex> vertices{};
		std::vector<uint32_t> indices{};
		std::vector<uint8_t> texture_data{};
		int texture_width{};
		int texture_height{};
		int texture_num_chan{};
		uint32_t mip_levels{};
		coord_system coordinate_system{ coord_axis::right, coord_axis::up, coord_axis::back };
		float3x4 transform;
		model* parent;
	};


	struct pose
	{
		float3 pos{ 0,0,0 };
		float4 rot{ 0,0,0,1 };
		float4x4 matrix_4x4() const { return linalg::pose_matrix(rot, pos); }
		float3x4 matrix_3x4() const { return {{linalg::qxdir(rot)}, {linalg::qydir(rot)}, {linalg::qzdir(rot)}, {pos}}; }

		static pose from_matrix(const float4x4 & pose_matrix) {
			return { { pose_matrix.w.x, pose_matrix.w.y, pose_matrix.w.z }, { linalg::rotation_quat(float3x3 { pose_matrix.x.xyz(), pose_matrix.y.xyz(), pose_matrix.z.xyz() } ) } };
		}
		static pose from_matrix(const float3x4 & pose_matrix) {
			return { { pose_matrix.w.x, pose_matrix.w.y, pose_matrix.w.z }, { linalg::rotation_quat(float3x3 { pose_matrix.x, pose_matrix.y, pose_matrix.z }) } };
		}
	};

	static float4 from_to(const float3 & from, const float3 & to) { return rotation_quat(normalize(cross(from, to)), angle(from, to)); }

	struct camera
	{
		float3 position;
		float pitch = 0, yaw = 0;

		float4 get_orientation() const { return qmul(rotation_quat(engine_coordinate_system.get_up(), yaw), rotation_quat(engine_coordinate_system.get_right(), pitch)); }
		pose get_pose() const { return { position, get_orientation() }; }
		float4x4 get_pose_matrix() const { return pose_matrix(get_orientation(), position); }
		float4x4 get_view_matrix() const { return inverse(get_pose_matrix()); }

		void move_local(const float3 & step)
		{
			position += qrot(get_orientation(), step);
		}

		void look_at(const float3 & center)
		{
			const auto up = engine_coordinate_system.get_up();

			const float3 fwd = normalize(center - position);
			const float3 flat_fwd = normalize(fwd - up * dot(fwd, up));
			const float4 yaw_quat = from_to(engine_coordinate_system.get_forward(), flat_fwd);

			const float3 pitch_fwd = qrot(qinv(yaw_quat), fwd);
			const float4 pitch_quat = from_to(engine_coordinate_system.get_forward(), pitch_fwd);

			pitch = qangle(pitch_quat) * dot(qaxis(pitch_quat), engine_coordinate_system.get_right());
			yaw = qangle(yaw_quat) * dot(qaxis(yaw_quat), engine_coordinate_system.get_up());
		}
	};

	struct uniform_buffer_object
	{
		float4x4 model{ linalg::identity };
		float4x4 view{ linalg::identity };
		float4x4 proj{ linalg::identity };
	};
}

namespace std {
	template<> struct hash<vk_renderer::vertex> {
		size_t operator()(vk_renderer::vertex const& vertex) const {
			return ((hash<float3>()(vertex.pos) ^
				(hash<float3>()(vertex.color) << 1)) >> 1) ^
				(hash<float2>()(vertex.tex_coord) << 1);
		}
	};
}
#endif