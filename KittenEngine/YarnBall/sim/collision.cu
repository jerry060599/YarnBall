#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include <thrust/swap.h>

#include "../YarnBall.h"

namespace YarnBall {
	template <bool USE_VEL_RADIUS>
	__global__ void buildAABBs(MetaData* data, int* errorReturn) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		auto verts = data->d_lastPos;
		auto flags = data->d_lastFlags[tid];
		Kit::LBVH::aabb aabb;

		auto p0 = verts[tid];
		if (flags & (uint32_t)VertexFlags::hasNext) {
			auto p1 = verts[tid + 1];

			aabb.absorb(p0);
			aabb.absorb(p1);

			if (USE_VEL_RADIUS) {
				auto dxs = data->d_dx;
				aabb.absorb(p0 + dxs[tid]);
				aabb.absorb(p1 + dxs[tid + 1]);
			}
			aabb.pad(data->scaledDetectionRadius);
		}

		data->d_bounds[tid] = aabb;
	}

	template <bool USE_VEL_RADIUS>
	__global__ void buildCollisionList(MetaData* data, int maxCols, int* errorReturn) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= maxCols) return;

		ivec2 ids = data->d_boundColList[tid];

		auto cids = data->d_lastCID;
		int c0 = cids[ids.x];
		int c1 = cids[ids.x + 1];

		// Exempt self-collisions due to glueing
		if (c0 == ids.y || c1 == ids.y || c0 == ids.y + 1 || c1 == ids.y + 1) return;
		// Exempt neighboring segments
		if (abs(ids.y - ids.x) <= 2) return;

		auto verts = data->d_lastPos;
		vec3 a0 = verts[ids.x];
		vec3 a1 = verts[ids.x + 1] - a0;
		vec3 b0 = verts[ids.y] - a0;
		vec3 b1 = verts[ids.y + 1] - a0;

		// Discrete collision detection

		vec2 uv = Kit::segmentClosestPoints(vec3(0), a1, b0, b1);
		if (!glm::isfinite(uv.x) || !glm::isfinite(uv.y))
			uv = vec2(0.5);

		// Remove duplicate collisions if there is a previous segment and the collision happens on the lower corner
		vec3 normal = uv.x * a1 - mix(b0, b1, uv.y);
		float d2 = Kit::length2(normal);

		float r = 2 * data->scaledDetectionRadius;
		if (USE_VEL_RADIUS) {
			auto dxs = data->d_dx;
			r += sqrt(glm::max(length2(dxs[ids.x]), length2(dxs[ids.x + 1])));
			r += sqrt(glm::max(length2(dxs[ids.y]), length2(dxs[ids.y + 1])));
		}

		float mr = 2 * data->radius;
		if (d2 < r * r) {
			if (d2 < mr * mr) // Report interpenetration
				errorReturn[1] = Sim::WARNING_SEGMENT_INTERPENETRATION;

			auto nCols = data->d_numCols;
			const auto collisions = data->d_collisions;
			int numCols = atomicAdd(&nCols[ids.x], 1);
			if (numCols >= MAX_COLLISIONS_PER_SEGMENT)
				errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
			else
				collisions[ids.x + numVerts * numCols] = ids.y;

			numCols = atomicAdd(&nCols[ids.y], 1);
			if (numCols >= MAX_COLLISIONS_PER_SEGMENT)
				errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
			else
				collisions[ids.y + numVerts * numCols] = ids.x;
		}
	}

	void Sim::detectCollisions() {
		// Rebuild bvh
		if (meta.useVelocityRadius)
			buildAABBs<true> << <(meta.numVerts + 255) / 256, 256 >> > (d_meta, d_error);
		else
			buildAABBs<false> << <(meta.numVerts + 255) / 256, 256 >> > (d_meta, d_error);

		if (lastBVHRebuild >= meta.bvhRebuildPeriod) {
			bvh.compute(meta.d_bounds, meta.numVerts);
			lastBVHRebuild = 0;
		}
		else {
			bvh.refit();
			lastBVHRebuild += meta.h * meta.detectionPeriod;
		}
		currentBounds = bvh.bounds();

		int numCols = bvh.query(meta.d_boundColList, meta.numVerts * MAX_COLLISIONS_PER_SEGMENT);

		// Build collision list
		cudaMemsetAsync(meta.d_numCols, 0, sizeof(int) * meta.numVerts, stream);
		if (meta.useVelocityRadius)
			buildCollisionList<true> << <(numCols + 127) / 128, 128, 0, stream >> > (d_meta, numCols, d_error);
		else
			buildCollisionList<false> << <(numCols + 127) / 128, 128, 0, stream >> > (d_meta, numCols, d_error);
	}

	template<bool LIMIT, bool USE_VEL_RADIUS>
	__global__ void recomputeStepLimitKernel(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= numVerts) return;

		float minDist = INFINITY;
		if (LIMIT && (bool)(data->d_lastFlags[tid] & (uint32_t)VertexFlags::hasNext)) {
			constexpr float SAFETY_MARGIN = 0.2f;

			// Linear change
			const auto verts = data->d_lastPos;
			vec3 a0 = verts[tid];
			vec3 a1 = verts[tid + 1] - a0;

			// This is the maximum move possible where the AABB query is still guaranteed to find the collision
			minDist = data->detectionRadius * (data->detectionScaler - 1);
			if (USE_VEL_RADIUS) 
				minDist += length(data->d_dx[tid]);

			// Collision energy of this segment
			const int numCols = data->d_numCols[tid];
			const auto collisions = data->d_collisions + tid;
			for (int i = 0; i < numCols; i++) {
				int oid = collisions[i * numVerts];
				vec3 b0 = verts[oid] - a0;
				vec3 b1 = verts[oid + 1] - a0;

				// Recompute contact data
				vec2 uv = Kit::segmentClosestPoints(vec3(0), a1, b0, b1);
				if (!glm::isfinite(uv.x) || !glm::isfinite(uv.y))
					uv = vec2(0.5);

				// Remove duplicate collisions if there is a previous segment and the collision happens on the lower corner
				vec3 normal = uv.x * a1 - mix(b0, b1, uv.y);
				float l = length(normal);
				minDist = min(minDist, ((1 - SAFETY_MARGIN) * 0.5f) * l);
			}
		}
		data->d_maxStepSize[tid] = minDist;
	}

	void Sim::recomputeStepLimit() {
		if (meta.useStepSizeLimit) {
			if (meta.useVelocityRadius)
				recomputeStepLimitKernel<true, true> << <(meta.numVerts + 127) / 128, 128, 0, stream >> > (d_meta);
			else recomputeStepLimitKernel<true, false> << <(meta.numVerts + 127) / 128, 128, 0, stream >> > (d_meta);
		}
		else {
			if (meta.useVelocityRadius)
				recomputeStepLimitKernel<false, true> << <(meta.numVerts + 127) / 128, 128, 0, stream >> > (d_meta);
			else recomputeStepLimitKernel<false, false> << <(meta.numVerts + 127) / 128, 128, 0, stream >> > (d_meta);
		}
	}
}