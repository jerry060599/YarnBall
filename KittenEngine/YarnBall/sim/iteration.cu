#if !defined(USE_HIP)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif
#include "../YarnBall.h"

namespace YarnBall {
	// Converts velocity to initial guess
	__global__ void initItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		const float h = data->h;
		auto verts = data->d_verts;
		auto lastVels = data->d_lastVels;

		const vec3 g = data->gravity;
		const vec3 vel = data->d_vels[tid];

		vec3 dx = h * vel;
		vec3 lastVel = lastVels[tid];
		lastVels[tid] = vel;
		float padding = data->scaledDetectionRadius;
		auto y = dx + (h * h) * g;
		padding += length(y);

		if (verts[tid].invMass != 0) {
			// Compute y (inertial + accel position)
			// Store it in vel (The actual vel is no longer needed)
			data->d_vels[tid] = y;

			// Compute initial guess
			float g2 = length2(g);
			if (g2 > 0) {
				vec3 a = (vel - lastVel) / data->lastH;
				float s = clamp(dot(a, g) / g2, 0.f, 1.f);
				dx += (h * h * s) * g;
			}
		}
		data->d_paddingSize[tid] = padding;
		data->d_dx[tid] = dx;

		// Transfer segment data
		vec3 pos = verts[tid].pos;
		data->d_lastPos[tid] = pos;
	}

	void Sim::startIterate() {
		initItr <<<(meta.numVerts + 255) / 256, 256, 0, stream >>> (d_meta);
	}

	// Converts dx back to velocity and advects
	__global__ void endItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		const float h = data->h;
		const float invH = 1 / h;
		auto verts = data->d_verts;

		// Linear velocity
		vec3 dx = data->d_dx[tid];
		if (verts[tid].invMass != 0)
			data->d_vels[tid] = dx * invH * (1 - data->drag * h);
		verts[tid].pos += dx;
	}

	void Sim::endIterate() {
		endItr <<<(meta.numVerts + 255) / 256, 256, 0, stream >>> (d_meta);
	}
}