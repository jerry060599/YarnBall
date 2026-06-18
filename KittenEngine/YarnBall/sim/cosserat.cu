
// Jerry Hsu, jerry.hsu.research@gmail.com, 2025

#if !defined(USE_HIP)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif
#include "../YarnBall.h"

namespace YarnBall {
	using Kit::hess3;

	// The main Cosserat iteration is split into sectors because we are compute bound when computing collisions
	// The whole block is divided into THREADS_PER_VERTEX sectors. 
	// Each sector computes a portion of the collision energy and this is summed up into sector 0
	// Sector 0 then performs the actual update

#define BLOCK_SIZE (32 * 4)
#define THREADS_PER_VERTEX (2)
#define VERTEX_PER_BLOCK (BLOCK_SIZE / THREADS_PER_VERTEX)

	template<bool LIMIT>
	__global__ void cosseratItr(MetaData* data) {
		const int sid = threadIdx.x / VERTEX_PER_BLOCK;			// Sector id
		const int ltid = threadIdx.x - sid * VERTEX_PER_BLOCK;	// Local tid
		const int tid = (int)(blockIdx.x * (VERTEX_PER_BLOCK - 1) + ltid) - 1;	// Thread id

		const int numVerts = data->numVerts;
		if (tid >= numVerts || tid < 0) return;

		const float h = data->h;
		const float damping = data->damping / h;
		const auto lastPos = data->d_lastPos;
		const auto dxs = data->d_dx;

		// Linear change
		Vertex v0 = data->d_verts[tid];
		vec3 dx = dxs[tid];

		// First couple of warps are used to compute the needed energies. 
		hess3 H(0);
		vec3 f(0);
		if (!sid) {
			// Hessian H
			H = hess3(1 / (v0.invMass * h * h));
			// vel has been overwritten to contain y - pos
			f = 1 / (h * h * v0.invMass) * (data->d_vels[tid] - dx);

			// Special connections energy
			if (v0.connectionIndex >= 0) {
				constexpr float stiffness = 4e1;
				vec3 p0 = lastPos[v0.connectionIndex];
				vec3 p0dx = dxs[v0.connectionIndex];
				f -= stiffness * ((v0.pos - p0) + (dx - p0dx) + damping * dx);
				H.diag += stiffness * (1 + damping);
			}
		}

		// We need to store absolute position and position updates seperatly for floating point precision
		// If we added these together, the update could be small enough to be rounded out, causing stability issues
		vec3 p1, p1dx;
		vec3 f2(0);
		hess3 H2(0);

		if (v0.flags & (uint32_t)VertexFlags::hasNext) {
			p1 = lastPos[tid + 1];
			p1dx = dxs[tid + 1];

			// Cosserat stretching energy
			if (!sid) {
				float invl = 1 / v0.lRest;
				vec3 c = ((p1 - v0.pos) + (p1dx - dx)) * invl - data->d_qs[tid] * vec3(1, 0, 0);

				float k = v0.kStretch * invl;
				float d = k * invl;
				f += k * c - (damping * d) * dx;
				f2 += -k * c - (damping * d) * p1dx;
				d *= 1 + damping;
				H.diag += d;
				H2.diag += d;
			}

			const float fricK = data->kFriction;
			const float invb = 1 / data->barrierThickness;
			const float radius = 2 * data->radius;
			const float fricMu = data->frictionCoeff;
			const auto collisions = data->d_collisions;
			const float kCol = data->kCollision * invb;

			// Collision energy of this segment
			const int numCols = data->d_numCols[tid];
			for (int i = sid; i < numCols; i += THREADS_PER_VERTEX) {
				int colID = collisions[tid + i * numVerts];

				vec3 b0 = lastPos[colID];
				vec3 b1 = lastPos[colID + 1];
				vec3 db0 = dxs[colID];
				vec3 db1 = dxs[colID + 1];

				// Compute collision UV and normal
				vec2 uv = Kit::segmentClosestPoints(
					vec3(0), (p1 - v0.pos) + (p1dx - dx),
					(b0 - v0.pos) + (db0 - dx), (b1 - v0.pos) + (db1 - dx));

				vec3 dpos = mix(v0.pos, p1, uv.x) - mix(b0, b1, uv.y);
				vec3 ddpos = mix(dx, p1dx, uv.x) - mix(db0, db1, uv.y);
				vec3 normal = dpos + ddpos;
				float d = length(normal);
				normal *= (1 / d);

				uv.y = uv.x;
				uv.x = 1 - uv.x;

				// Compute penetration
				d = d - radius;
				d *= invb;
				if (d > 1) continue;	// Not touching
				d = max(d, 1e-3f);		// Clamp to some small value. This is a ratio of the barrier thickness.

				// IPC barrier energy
				float invd = 1 / d;
				float logd = log(d);

				float dH = (-3 + (2 + invd) * invd - 2 * logd) * kCol * invb;
				float ff = -(1 - d) * (d - 1 + 2 * d * logd) * invd * kCol;
				f += (ff * uv.x - damping * dH * uv.x * uv.x * dot(normal, dx)) * normal;
				f2 += (ff * uv.y - damping * dH * uv.y * uv.y * dot(normal, p1dx)) * normal;

				dH *= 1 + damping;
				hess3 op = hess3::outer(normal);
				H += op * (dH * uv.x * uv.x);
				H2 += op * (dH * uv.y * uv.y);

				// Friction
				vec3 u = ddpos - dot(normal, ddpos) * normal;
				float ul = length(u);
				if (ul > 0) {
					float f1 = glm::min(fricK, fricMu * ff / ul);

					op.diag -= 1;

					f -= f1 * uv.x * u;
					H -= op * (Kit::pow2(uv.x) * f1);

					f2 -= f1 * uv.y * u;
					H2 -= op * (Kit::pow2(uv.y) * f1);
				}
			}
		}

		__shared__ float sharedData[18 * VERTEX_PER_BLOCK];

		// Reduce forces to the lower threads
		vec3* f0s = (vec3*)sharedData;
		vec3* f1s = (vec3*)(sharedData + 3 * VERTEX_PER_BLOCK);
		hess3* h0s = (hess3*)(sharedData + 6 * VERTEX_PER_BLOCK);
		hess3* h1s = (hess3*)(sharedData + 12 * VERTEX_PER_BLOCK);

		if (sid) {
			f0s[ltid] = f;
			f1s[ltid] = f2;
			h0s[ltid] = H;
			h1s[ltid] = H2;
		}
		__syncthreads();

		if (!sid) {
			f += f0s[ltid];
			f2 += f1s[ltid];
			H += h0s[ltid];
			H2 += h1s[ltid];
		}
		__syncthreads();

		// Sum forces across the yarn segments
		if (!sid) {
			vec4* forces = (vec4*)sharedData;
			hess3* hessians = (hess3*)(sharedData + 4 * VERTEX_PER_BLOCK);

			float stepLimit = INFINITY;
			if (LIMIT) stepLimit = data->d_maxStepSize[tid];
			forces[threadIdx.x] = vec4(f2, stepLimit);
			hessians[threadIdx.x] = H2;

			__syncthreads();

			// No reason to keep thread 0 going anymore
			if (!threadIdx.x) return;

			if (v0.flags & (uint32_t)VertexFlags::hasPrev) {
				vec4 v = forces[threadIdx.x - 1];
				if (LIMIT) stepLimit = min(stepLimit, v.w);
				f += vec3(v);
				H += hessians[threadIdx.x - 1];
			}

			if (v0.invMass != 0) {
				// Local solve
				vec3 delta = data->accelerationRatio * (inverse((mat3)H) * f);
				dx += delta;

				if (LIMIT) {
					float l = length(dx);
					if (l > stepLimit && l > 0) dx *= stepLimit / l;
				}

				// Apply update
				dxs[tid] = dx;
			}
		}
	}

	__global__ void quaternionLambdaItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= numVerts || tid < 0) return;

		const auto verts = data->d_verts;
		const auto dxs = data->d_dx;

		// Linear change
		Vertex v0 = verts[tid];

		// Update segment orientation
		// This is done assuming some very very large invMoment (i.e. no inertia so static equilibrium)
		if (!(bool)(v0.flags & (uint32_t)VertexFlags::fixOrientation) != 0 && (v0.flags & (uint32_t)VertexFlags::hasNext)) {
			vec3 dx = dxs[tid];
			vec3 p1 = verts[tid + 1].pos;
			vec3 p1dx = dxs[tid + 1];

			// All this is from an alternate derivation from forced-base hair interpolation.
			v0.pos = ((p1 - v0.pos) + (p1dx - dx)) / v0.lRest;
			v0.pos *= -2 * v0.kStretch;

			vec4 b(0);
			auto qs = data->d_qs;
			auto qRests = data->d_qRests;
			auto q0 = qs[tid];
			if (v0.flags & (uint32_t)VertexFlags::hasPrev) {
				auto qRest = Kit::Rotor(qRests[tid - 1]);
				auto qq = qs[tid - 1];
				float s = dot((qq.inverse() * q0).v, qRest.v) > 0 ? 1 : -1;
				b += s * (qq * qRest).v;
			}

			if (v0.flags & (uint32_t)VertexFlags::hasNextOrientation) {
				auto qRest = Kit::Rotor(qRests[tid]);
				auto qq = qs[tid + 1];
				float s = dot((q0.inverse() * qq).v, qRest.v) > 0 ? 1 : -1;
				b += s * (qq * qRest.inverse()).v;
			}

			float lambda = length(v0.pos) + length(b);
			q0 = Kit::Rotor(normalize((Kit::Rotor(v0.pos) * Kit::Rotor(b) * Kit::Rotor(1)).v + lambda * b));
			qs[tid] = q0;
		}
	}

	void Sim::iterateCosserat() {
		if (meta.useStepSizeLimit)
			cosseratItr<true> <<<(meta.numVerts + VERTEX_PER_BLOCK - 2) / (VERTEX_PER_BLOCK - 1), BLOCK_SIZE, 0, stream >>> (d_meta);
		else
			cosseratItr<false> <<<(meta.numVerts + VERTEX_PER_BLOCK - 2) / (VERTEX_PER_BLOCK - 1), BLOCK_SIZE, 0, stream >>> (d_meta);
		quaternionLambdaItr <<<(meta.numVerts + 127) / 128, 128, 0, stream >>> (d_meta);
	}
}