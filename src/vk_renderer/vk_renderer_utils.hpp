#pragma once
#ifndef VK_RENDERER_UTILS
#define VK_RENDERER_UTILS

#include "vk_renderer_common.hpp"

#include <linalg.h>
#include <vector>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include <stb_image.h>

#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif
#include <tiny_obj_loader.h>

#include <unordered_map>
#include <tuple>

using namespace linalg::aliases;

const float PI_CONST = 3.14159265358979323846f;
const float DEG_TO_RAD = 0.017453292519943295769236907684886f;

namespace vk_renderer::utils
{
	// Transform matrix "matrix" by a specific coordinate system transformation
	inline float3x3 transform_matrix(const float3x3 & coord_transform, const float3x3 & matrix) { return mul(coord_transform, matrix, inverse(coord_transform)); }

	// Convert angle axis to euler
	static float3 to_euler(const float3 & axis, const float & angle)
	{
		float yaw, pitch, roll;

		float s = sin(angle);
		float c = cos(angle);
		float t = 1 - c;

		if ((axis.x*axis.y*t + axis.z * s) > 0.998f) { // north pole singularity detected
			yaw = 2 * atan2(axis.x*sin(angle * 0.5f), cos(angle * 0.5f));
			pitch = PI_CONST * 0.5f;
			roll = 0;
			goto euler_end;
		}
		if ((axis.x*axis.y*t + axis.z * s) < -0.998f) { // south pole singularity detected
			yaw = -2 * atan2(axis.x*sin(angle * 0.5f), cos(angle *0.5f));
			pitch = -PI_CONST * 0.5f;
			roll = 0;
			goto euler_end;
		}
		yaw = atan2(axis.y * s - axis.x * axis.z * t, 1 - (axis.y*axis.y + axis.z * axis.z) * t);
		pitch = asin(axis.x * axis.y * t + axis.z * s);
		roll = atan2(axis.x * s - axis.y * axis.z * t, 1 - (axis.x*axis.x + axis.z * axis.z) * t);

	euler_end:
		return float3{ roll, yaw, pitch };
	}

	// Convert quaternion to euler
	static float3 to_euler(const float4 & q)
	{
		float pitch, yaw, roll;

		// roll (x-axis rotation)
		float sinr_cosp = +2.0f * (q.w * q.x + q.y * q.z);
		float cosr_cosp = +1.0f - 2.0f * (q.x * q.x + q.y * q.y);
		roll = atan2(sinr_cosp, cosr_cosp);

		// pitch (y-axis rotation)
		float sinp = +2.0f * (q.w * q.y - q.z * q.x);
		if (fabs(sinp) >= 1)
			pitch = copysign(PI_CONST / 2, sinp); // use 90 degrees if out of range
		else
			pitch = asin(sinp);

		// yaw (z-axis rotation)
		float siny_cosp = +2.0f * (q.w * q.z + q.x * q.y);
		float cosy_cosp = +1.0f - 2.0f * (q.y * q.y + q.z * q.z);
		yaw = atan2(siny_cosp, cosy_cosp);

		return { roll, pitch, yaw };
	}

	// Convert rotation matrix to euler
	static float3 to_euler(const float3x3 & rot_mat)
	{
		return to_euler(linalg::rotation_quat(rot_mat));
	}

	static inline std::vector<uint8_t> load_image_file_to_memory(const std::string & texture_file_path, int& tex_width, int& tex_height, int& num_chan, size_t& image_size)
	{
		stbi_uc* pixels = stbi_load(texture_file_path.c_str(), &tex_width, &tex_height, &num_chan, STBI_rgb_alpha);
		image_size = tex_width * tex_height * 4;

		if (!pixels) {
			throw std::runtime_error("failed to load texture image");
		}

		std::vector<uint8_t> pix_vec(image_size);
		memcpy(pix_vec.data(), pixels, image_size);
		stbi_image_free(pixels);

		return pix_vec;
	}

	static inline model load_obj_file_to_memory(const std::string & obj_id, const std::string & obj_file_path, const coord_system coords = { coord_axis::forward, coord_axis::left, coord_axis::up })
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_file_path.c_str())) {
			throw std::runtime_error("Error loading OBJ file '" + obj_file_path + "': " + warn + err);
		}

		model the_model{};
		the_model.id = obj_id;
		the_model.coordinate_system = coords;

		std::unordered_map<vertex, uint32_t> unique_vertices = {};
		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				vertex vert = {};

				vert.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vert.tex_coord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};

				vert.color = { 1.0f, 1.0f, 1.0f };

				if (unique_vertices.count(vert) == 0) {
					unique_vertices[vert] = static_cast<uint32_t>(the_model.vertices.size());
					the_model.vertices.push_back({ vert });
				}

				the_model.indices.push_back(unique_vertices[vert]);
			}
		}

		return the_model;
	}

	static inline model load_obj_file_to_memory(const std::string & obj_id, const std::string & obj_file_path, const std::string & texture_file_path, const coord_system coords = { coord_axis::forward, coord_axis::left, coord_axis::up })
	{
		auto the_model = load_obj_file_to_memory(obj_id, obj_file_path);

		size_t image_size{};
		the_model.texture_data = utils::load_image_file_to_memory(texture_file_path, the_model.texture_width, the_model.texture_height, the_model.texture_num_chan, image_size);
		the_model.mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(the_model.texture_width, the_model.texture_height)))) + 1;

		return the_model;
	}
}

#endif