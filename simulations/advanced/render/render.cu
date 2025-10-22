/*
 * render.cu  Andrew Belles  Oct 4th, 2025
 *
 * CUDA/OpenGL interop renderer implementation.
 * Credit ChatGPT-5 for writing this code. 
 * Reviewed by Andrew Belles
 */

#include "render.cuh"

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string_view>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace {

/************ hardcoded shaders & other static global constants *****/

bool g_glfw_initialized = false;
int g_glfw_contexts = 0;

constexpr float tol = 1e-3; 

constexpr std::string_view kWindowTitle = "Simulation";

constexpr std::string_view kVertexShaderSrc = R"(\
#version 330 core
layout(location = 0) in vec3 inPosition;
uniform mat4 uViewProj;
uniform float uPointSize;
void main() {
  gl_Position = uViewProj * vec4(inPosition, 1.0);
  gl_PointSize = uPointSize;
}
)";

constexpr std::string_view kFragmentShaderSrc = R"(\
#version 330 core
out vec4 fragColor;
uniform vec3 uColor;
void main() {
  fragColor = vec4(uColor, 1.0);
}
)";

void 
glfw_error_callback(int error, const char* description)
{
  std::cerr << "[RENDER] glfw error: " << error 
            << ((description) ? description : "") << '\n';
}

/************ compile_shader() **************************************/
/*
 * Compiles a shader given its type and the string literal to compile
 * 
 * Caller Provides; 
 *   Enum for type of shader and the shader literal 
 */
GLuint 
compile_shader(GLenum type, std::string_view source) 
{
  // compile shaders 
  GLuint shader = glCreateShader(type);
  const char* src = source.data();
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);

  // ensure shader was properly generated 
  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

  if ( !success ) {
    GLint log_len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);

    // log message 
    std::string log(log_len, '\0');
    glGetShaderInfoLog(shader, log_len, nullptr, log.data());
    std::cerr << "[RENDER] Shader compilation failed: " << log << '\n';
    glDeleteShader(shader);
    return 0;
  }

  return shader;
}

/************ build_program() ***************************************/
/*
 * Builds a shader program from fragment and vertex shaders 
 * 
 */
GLuint
build_program() 
{
  // compile both shaders 
  GLuint vs = compile_shader(GL_VERTEX_SHADER, kVertexShaderSrc);
  if ( !vs ) {
    return 0;
  }

  GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFragmentShaderSrc);
  if ( !fs ) {
    glDeleteShader(vs);
    return 0;
  }

  // attach to program 
  GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  glDetachShader(program, vs);
  glDetachShader(program, fs);
  glDeleteShader(vs);
  glDeleteShader(fs);

  // ensure program was correctly linked, bail if not 
  GLint success = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if ( !success ) {
    GLint log_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);

    std::string log(log_len, '\0');
    glGetProgramInfoLog(program, log_len, nullptr, log.data());
    std::cerr << "[RENDER] Program link failed: " << log << '\n';
    glDeleteProgram(program);
    return 0;
  }

  return program;
}

/************ ensure_gl_context() ***********************************/ 
/*
 * Wrapper function to ensure that context is valid  
 */
cudaError_t 
ensure_gl_context(render::Context* context) 
{
  if (!context || !context->window) {
    return cudaErrorInvalidValue;
  }
  glfwMakeContextCurrent(context->window);
  return cudaSuccess;
}

void 
update_view_projection(render::Context* context, int fb_width, int fb_height)
{
  if (!context) {
    return;
  }

  context->framebuffer_width = fb_width;
  context->framebuffer_height = fb_height;

  const float aspect =
      (fb_height > 0) ? static_cast<float>(fb_width) / static_cast<float>(fb_height) : 1.0f;

  const float near_plane = std::max(context->conf.camera.near_plane, tol);
  float far_plane = context->conf.camera.far_plane;
  if (far_plane <= near_plane + tol) {
    far_plane = near_plane + 1.0;
  }

  const glm::vec3 eye(context->conf.camera.position.x,
                      context->conf.camera.position.y,
                      context->conf.camera.position.z);
  const glm::vec3 target(context->conf.camera.target.x,
                         context->conf.camera.target.y,
                         context->conf.camera.target.z);
  glm::vec3 up(context->conf.camera.up.x,
               context->conf.camera.up.y,
               context->conf.camera.up.z);

  if (glm::dot(up, up) < tol) {
    up = glm::vec3(0.0, 1.0, 0.0);
  }

  const glm::mat4 view = glm::lookAt(eye, target, up);
  const glm::mat4 proj = glm::perspective(glm::radians(context->conf.camera.fov_y_degrees),
                                          aspect, near_plane, far_plane);
  const glm::mat4 vp = proj * view;
  std::memcpy(context->view_proj, glm::value_ptr(vp), sizeof(context->view_proj));
}

} // namespace

namespace render {

cudaError_t
create(Context* context, const Config& conf) 
{
  glfwSetErrorCallback(glfw_error_callback);
  if ( !context || conf.agents <= 0 || conf.point_size <= 0.0 ) {
    return cudaErrorInvalidValue;
  }

  *context = Context{};
  context->conf = conf;

  if ( !g_glfw_initialized ) {
    if (!glfwInit()) {
      glfwGetError(nullptr);
      return cudaErrorUnknown;
    } else {
      g_glfw_initialized = true;
    }
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);

  context->window = glfwCreateWindow(conf.width, conf.height,
                                     kWindowTitle.data(), nullptr, nullptr);
  glfwGetError(nullptr);
  if ( !context->window ) {
    destroy(context);
    return cudaErrorUnknown;
  }

  glfwMakeContextCurrent(context->window);
  glfwSwapInterval(0);

  glewExperimental = GL_TRUE;
  GLenum glew_status = glewInit();
  if ( glew_status != GLEW_OK ) {
    std::cerr << "[RENDER] Failed to initialize GLEW: "
              << reinterpret_cast<const char*>(glewGetErrorString(glew_status))
              << '\n';
    destroy(context);
    return cudaErrorUnknown;
  }
  glGetError();

  glEnable(GL_PROGRAM_POINT_SIZE);

  glGenVertexArrays(1, &context->vao);
  glBindVertexArray(context->vao);

  glGenBuffers(1, &context->positions.vbo);
  glBindBuffer(GL_ARRAY_BUFFER, context->positions.vbo);
  const size_t buffer_bytes = static_cast<size_t>(conf.agents) * sizeof(float3);
  glBufferData(GL_ARRAY_BUFFER, buffer_bytes, nullptr, GL_DYNAMIC_DRAW);
  context->positions.byte_size = buffer_bytes;

  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glClearDepth(1.0);

  cudaError_t status = cudaGraphicsGLRegisterBuffer(
    &context->positions.resource, context->positions.vbo,
    cudaGraphicsRegisterFlagsWriteDiscard
  );
  if ( status != cudaSuccess ) {
    destroy(context);
    return status;
  }

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), nullptr);
  glEnableVertexAttribArray(0);

  context->shader = build_program();
  if ( context->shader == 0 ) {
    destroy(context);
    return cudaErrorUnknown;
  }
  context->view_proj_loc = glGetUniformLocation(context->shader, "uViewProj");
  context->color_loc = glGetUniformLocation(context->shader, "uColor");
  context->point_size_loc = glGetUniformLocation(context->shader, "uPointSize");

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glfwShowWindow(context->window);

  g_glfw_contexts++;
  context->draw_count = conf.agents;
  std::cerr << "Context Count: " << g_glfw_contexts << '\n';
  return cudaSuccess;
}

void
destroy(Context* context)
{
  if (!context) {
    return;
  }

  if (context->window) {
    glfwMakeContextCurrent(context->window);
  }

  if (context->positions.resource) {
    cudaGraphicsUnregisterResource(context->positions.resource);
    context->positions.resource = nullptr;
  }

  if (context->positions.vbo) {
    glDeleteBuffers(1, &context->positions.vbo);
    context->positions.vbo = 0;
  }

  if (context->vao) {
    glDeleteVertexArrays(1, &context->vao);
    context->vao = 0;
  }

  if (context->shader) {
    glDeleteProgram(context->shader);
    context->shader = 0;
  }

  context->view_proj_loc = -1;
  context->color_loc = -1;
  context->point_size_loc = -1;
  context->framebuffer_width = 0;
  context->framebuffer_height = 0;
  context->draw_count = 0;
  std::fill(std::begin(context->view_proj), std::end(context->view_proj), 0.0f);

  if (context->window) {
    glfwDestroyWindow(context->window);
    context->window = nullptr;
    if (g_glfw_contexts > 0) {
      --g_glfw_contexts;
    }
  }

  if (g_glfw_initialized && g_glfw_contexts == 0) {
    glfwTerminate();
    g_glfw_initialized = false;
  }

  context->positions.byte_size = 0;
}

cudaError_t 
begin_frame(Context* context)
{
  cudaError_t status = ensure_gl_context(context);
  if ( status != cudaSuccess ) {
    return status;
  }

  glfwPollEvents();

  int fb_width = 0;
  int fb_height = 0;
  glfwGetFramebufferSize(context->window, &fb_width, &fb_height);
  if ( fb_width <= 0 || fb_height <= 0 ) {
    fb_width = context->conf.width;
    fb_height = context->conf.height;
  }

  update_view_projection(context, fb_width, fb_height);

  glViewport(0, 0, fb_width, fb_height);
  glClearColor(0.02, 0.02, 0.05, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  return cudaSuccess;
}

cudaError_t 
upload_positions(Context* context, const float3* d_positions, int count) 
{
  if (!d_positions || count < 0) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = ensure_gl_context(context);
  if (status != cudaSuccess) {
    return status;
  }

  const int expected = context->conf.agents;
  if (count > expected) {
    return cudaErrorInvalidValue;
  }

  status = cudaGraphicsMapResources(1, &context->positions.resource, 0);
  if (status != cudaSuccess) {
    return status;
  }

  void* mapped_ptr = nullptr;
  size_t mapped_size = 0;
  status = cudaGraphicsResourceGetMappedPointer(&mapped_ptr, &mapped_size,
                                                context->positions.resource);
  if (status != cudaSuccess) {
    cudaGraphicsUnmapResources(1, &context->positions.resource, 0);
    return status;
  }

  const size_t bytes = static_cast<size_t>(count) * sizeof(float3);
  if (bytes > mapped_size) {
    cudaGraphicsUnmapResources(1, &context->positions.resource, 0);
    return cudaErrorInvalidValue;
  }

  status = cudaMemcpy(mapped_ptr, d_positions, bytes, cudaMemcpyDeviceToDevice);

  cudaError_t unmap_status =
      cudaGraphicsUnmapResources(1, &context->positions.resource, 0);
  if (status == cudaSuccess && unmap_status != cudaSuccess) {
    status = unmap_status;
  } else if (status == cudaSuccess) {
    context->draw_count = count;
  }

  return status;
}

cudaError_t 
end_frame(Context* context) 
{
  cudaError_t status = ensure_gl_context(context);
  if (status != cudaSuccess) {
    return status;
  }

  glUseProgram(context->shader);
  const float3& color = context->conf.color;
  if (context->color_loc >= 0) {
    glUniform3f(context->color_loc, color.x, color.y, color.z);
  }

  if (context->point_size_loc >= 0) {
    glUniform1f(context->point_size_loc, context->conf.point_size);
  }

  if (context->view_proj_loc >= 0) {
    glUniformMatrix4fv(context->view_proj_loc, 1, GL_FALSE, context->view_proj);
  }

  glBindVertexArray(context->vao);
  const int vertices = (context->draw_count > 0) ? context->draw_count
                                                 : context->conf.agents;
  glDrawArrays(GL_POINTS, 0, vertices);
  glBindVertexArray(0);

  glUseProgram(0);
  glfwSwapBuffers(context->window);

  return cudaSuccess;
}

} // namespace render
