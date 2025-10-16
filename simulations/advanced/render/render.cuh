/*
 * render.cuh  Andrew Belles  Oct 4th, 2025 
 *
 * Defines the interface for the backend renderer
 *
 *
 */

#ifndef __RENDER_CUH 
#define __RENDER_CUH

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h> 
#include <cuda_gl_interop.h>

#include <cstdint> 

namespace render {

/*
 * Runtime configuration for renderer 
 */ 
struct Config {
  int agents; 
  float3 color; 
  float point_size; 
  int width = 1280, height = 720; 
};

/*
 * Single bound cudaGL buffer 
 */
struct Buffer {
  cudaGraphicsResource_t resource = nullptr; 
  GLuint vbo = 0;
  uint64_t byte_size = 0; 
};

/*
 * Context of window where render resides within
 */
struct Context {
  Config conf; 
  GLFWwindow* window = nullptr; 
  GLuint vao = 0, shader = 0; 
  Buffer positions; 
};

cudaError_t create(Context* context, const Config& conf);
void destroy(Context* context);
cudaError_t begin_frame(Context* context);
cudaError_t upload_positions(Context* context, const float3* d_positions,
                             int count);
cudaError_t end_frame(Context* context);

} // end namespace render 

#endif // __RENDER_CUH
