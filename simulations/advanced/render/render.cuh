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

struct Camera {
  float3 position = {0.0, 0.0, 5.0};
  float3 target = {0.0, 0.0, 0.0};
  float3 up = {0.0, 1.0, 0.0};
  float fov_y_degrees = 45.0;
  float near_plane = 0.1;
  float far_plane = 100.0;
};

/*
 * Runtime configuration for renderer 
 */ 
struct Config {
  int agents; 
  float3 color; 
  float point_size; 
  int width = 1280, height = 720; 
  Camera camera;
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
  GLint view_proj_loc = -1;
  GLint color_loc = -1;
  GLint point_size_loc = -1;
  int framebuffer_width = 0;
  int framebuffer_height = 0;
  int draw_count = 0;
  float view_proj[16] = {0.0};
};

cudaError_t create(Context* context, const Config& conf);
void destroy(Context* context);
cudaError_t begin_frame(Context* context);
cudaError_t upload_positions(Context* context, const float3* d_positions,
                             int count);
cudaError_t end_frame(Context* context);

} // end namespace render 

#endif // __RENDER_CUH
