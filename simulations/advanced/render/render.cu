/*
 * render.cu  Andrew Belles  Oct 4th, 2025
 *
 * CUDA/OpenGL interop renderer implementation.
 * Credit ChatGPT-5 for writing this code. 
 * Reviewed by Andrew Belles
 */

#include "render.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <string_view>
#include <string>

namespace {

/************ hardcoded shaders & other static global constants *****/

bool g_glfw_initialized = false;
int g_glfw_contexts = 0;

constexpr std::string_view kWindowTitle = "Simulation";

constexpr std::string_view kVertexShaderSrc = R"(\
#version 330 core
layout(location = 0) in vec3 inPosition;
uniform float uPointSize;
void main() {
  gl_Position = vec4(inPosition, 1.0);
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

} // namespace

namespace render {

cudaError_t
create(Context* context, const Config& conf) 
{
  if ( !context || conf.agents <= 0 || conf.point_size <= 0.0 ) {
    return cudaErrorInvalidValue;
  }

  *context = Context{};
  context->conf = conf;

  if ( !g_glfw_initialized ) {
    if (!glfwInit()) {
      return cudaErrorUnknown;
    }
    g_glfw_initialized = true;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

  context->window = glfwCreateWindow(conf.width, conf.height,
                                     kWindowTitle.data(), nullptr, nullptr);
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

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glfwShowWindow(context->window);

  g_glfw_contexts++;
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

  glViewport(0, 0, fb_width, fb_height);
  glClearColor(0.02, 0.02, 0.05, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);

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
  GLint color_loc = glGetUniformLocation(context->shader, "uColor");
  if (color_loc >= 0) {
    glUniform3f(color_loc, color.x, color.y, color.z);
  }

  GLint point_size_loc = glGetUniformLocation(context->shader, "uPointSize");
  if (point_size_loc >= 0) {
    glUniform1f(point_size_loc, context->conf.point_size);
  }

  glBindVertexArray(context->vao);
  glDrawArrays(GL_POINTS, 0, context->conf.agents);
  glBindVertexArray(0);

  glUseProgram(0);
  glfwSwapBuffers(context->window);

  return cudaSuccess;
}

} // namespace render
