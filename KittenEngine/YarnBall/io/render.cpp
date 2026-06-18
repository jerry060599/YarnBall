#include "../YarnBall.h"
#if !defined(USE_HIP)
#include <cuda.h>
#endif

namespace YarnBall {
	void Sim::startRender() {
		// The GPU<->GL buffer sharing path uses cuda*GL interop, which the ROCm
		// ComputeBuffer does not expose. Rendering is only reached in the GUI
		// (non-headless) loop; headless simulation never calls this.
#if !defined(USE_HIP)
		vertBuffer->cudaWriteGL(meta.d_verts);
		qBuffer->cudaWriteGL(meta.d_qs);
#endif
	}

	void Sim::render() {
		static auto segBase = Kit::get<Kit::Shader>("resources\\shaders\\yarnSegBase.glsl");
		static auto segForward = Kit::get<Kit::Shader>("resources\\shaders\\yarnSegForward.glsl");

		static auto shadedBase = Kit::get<Kit::Shader>("resources\\shaders\\yarnSegBase.glsl");
		static auto shadedForward = Kit::get<Kit::Shader>("resources\\shaders\\yarnForward.glsl");

		vertBuffer->bind(5);
		qBuffer->bind(6);

		auto base = renderShaded ? shadedBase : segBase;
		auto forward = renderShaded ? shadedForward : segForward;

		base->setInt("numVerts", meta.numVerts);
		forward->setInt("numVerts", meta.numVerts);

		float r = meta.radius + 0.5f * meta.barrierThickness;
		base->setFloat("radius", r);
		forward->setFloat("radius", r);

		Kit::renderInstancedForward(renderShaded ? cylMeshHiRes : cylMesh, meta.numVerts - 1, base, forward);
	}

	void Sim::renderShadows() {
		static auto segBase = Kit::get<Kit::Shader>("resources\\shaders\\yarnSegBase.glsl");

		vertBuffer->bind(5);
		segBase->setInt("numVerts", meta.numVerts);

		float r = meta.radius + 0.5f * meta.barrierThickness;
		segBase->setFloat("radius", r);

		Kit::renderInstancedShadows(cylMesh, meta.numVerts - 1, segBase);
	}
}