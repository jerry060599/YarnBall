#include "../YarnBall.h"
#include <cuda.h>

namespace YarnBall {
	void Sim::printCollisionStats() {
		// Download collision counts
		int* numCols = new int[meta.numVerts];
		cudaMemcpy(numCols, meta.d_numCols, meta.numVerts * sizeof(int), cudaMemcpyDeviceToHost);

		size_t totCols = 0;
		int numSegWithCols = 0;

		for (int i = 0; i < meta.numVerts; i++)
			if (numCols[i] > 0) {
				totCols += numCols[i];
				numSegWithCols++;
			}

		printf("Total collisions: %zd\n", totCols);
		printf("Segments with collisions: %d/%d\n", numSegWithCols, meta.numVerts);

		delete[] numCols;
	}
}