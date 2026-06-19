#pragma once
// Jerry Hsu, 2021

#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>

#if defined(USE_HIP)
// cuda_to_hip.h (force-included) already mapped the cudaGraphics* names to
// hipGraphics*; pull the GL interop declarations now that glad is in scope.
#include <hip/hip_gl_interop.h>
#elif __has_include("cuda_runtime.h")
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

namespace Kitten {
	class ComputeBuffer {
	public:
		unsigned int glHandle;
		size_t elementSize, size;
		GLenum usage;

		ComputeBuffer() = delete;
		ComputeBuffer(size_t elementSize, size_t count, GLenum usage = GL_DYNAMIC_READ);
		~ComputeBuffer();

		void bind(int loc);
		void resize(size_t newSize);
		void upload(void* src);
		void upload(void* src, size_t count);
		void download(void* dst);
		void download(void* dst, size_t count);

#if defined(__CUDA_RUNTIME_H__) || defined(USE_HIP)
		// These are provided as an alternative to CudaComputerBuffer for compatibility reasons
		void cudaWriteGL(void* ptr, size_t dataSize) {
			cudaGraphicsResource* cudaRes;
			void* cudaPtr;
			cudaGraphicsGLRegisterBuffer(&cudaRes, glHandle, cudaGraphicsRegisterFlagsNone);
			cudaGraphicsMapResources(1, &cudaRes);

			size_t tmp;
			cudaGraphicsResourceGetMappedPointer(&cudaPtr, &tmp, cudaRes);
			cudaMemcpy(cudaPtr, ptr, dataSize, cudaMemcpyDeviceToDevice);

			cudaGraphicsUnmapResources(1, &cudaRes, 0);
			cudaGraphicsUnregisterResource(cudaRes);
		}

		void cudaWriteGL(void* ptr) { cudaWriteGL(ptr, elementSize * size); }

		void cudaReadGL(void* ptr, size_t dataSize) {
			cudaGraphicsResource* cudaRes;
			void* cudaPtr;
			cudaGraphicsGLRegisterBuffer(&cudaRes, glHandle, cudaGraphicsRegisterFlagsNone);
			cudaGraphicsMapResources(1, &cudaRes);

			size_t tmp;
			cudaGraphicsResourceGetMappedPointer(&cudaPtr, &tmp, cudaRes);
			cudaMemcpy(ptr, cudaPtr, dataSize, cudaMemcpyDeviceToDevice);

			cudaGraphicsUnmapResources(1, &cudaRes, 0);
			cudaGraphicsUnregisterResource(cudaRes);
		}

		void cudaReadGL(void* ptr) { cudaReadGL(ptr, elementSize * size); }
#endif
	};

#if defined(__CUDA_RUNTIME_H__) || defined(USE_HIP)
	class CudaComputeBuffer : public ComputeBuffer {
	public:
		cudaGraphicsResource* cudaRes;
		void* cudaPtr;

		CudaComputeBuffer() = delete;
		CudaComputeBuffer(size_t elementSize, size_t count, GLenum usage = GL_DYNAMIC_READ) :
			ComputeBuffer(elementSize, count, usage) {
			cudaGraphicsGLRegisterBuffer(&cudaRes, glHandle, cudaGraphicsRegisterFlagsNone);
			cudaGraphicsMapResources(1, &cudaRes);

			size_t size;
			cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaRes);
		}

		~CudaComputeBuffer() {
			cudaGraphicsUnmapResources(1, &cudaRes, 0);
			cudaGraphicsUnregisterResource(cudaRes);
		}
	};
#endif

}