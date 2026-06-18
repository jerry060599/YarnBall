
#include "../YarnBall.h"

namespace YarnBall {
	void Sim::rebuildCUDAGraph() {
		// Graph is still good
		if (meta.numItr == lastItr)
			return;
		checkCudaErrors(cudaGetLastError());

		cudaStreamSynchronize(stream);
		// Build graph with detection
		{
			cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

			if (meta.useStepSizeLimit)
				recomputeStepLimit();

			for (size_t i = 0; i < meta.numItr; i++)
				iterateCosserat();

			endIterate();

			cudaGraph_t graph;
			cudaStreamEndCapture(stream, &graph);

			if (stepGraph) cudaGraphExecDestroy(stepGraph);
			cudaGraphInstantiate(&stepGraph, graph, NULL, NULL, 0);
			cudaGraphDestroy(graph);
		}

		checkCudaErrors(cudaGetLastError());

		lastItr = meta.numItr;
	}

	float Sim::advance(float h) {
		if (h <= 0) return 0;

		int steps = max(1, (int)ceil(h / maxH));
		meta.lastH = meta.h;
		meta.h = h / steps;

		rebuildCUDAGraph();
		uploadMeta();

		for (int s = 0; s < steps; s++, stepCounter++) {
			startIterate();
			if (meta.detectionPeriod > 0 && stepCounter % meta.detectionPeriod == 0)
				detectCollisions();
			cudaGraphLaunch(stepGraph, stream);
		}

		meta.time += h;
		checkErrors();
		return h;
	}

	void Sim::step(float h) {
		advance(maxH);
	}
}