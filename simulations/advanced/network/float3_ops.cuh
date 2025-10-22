/*
 * float3_ops.cuh  Andrew Belles  Oct 4th, 2025
 *
 * Common inline overloads for CUDA float3 utility operations.
 *
 */

#ifndef NETWORK_FLOAT3_OPS_CUH
#define NETWORK_FLOAT3_OPS_CUH

#include <cuda_runtime.h>

__host__ __device__ inline float3
operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3
operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3&
operator+=(float3& a, const float3& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__host__ __device__ inline float3&
operator-=(float3& a, const float3& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

__host__ __device__ inline float3
operator*(float c, const float3 a)
{
  return make_float3(c * a.x, c * a.y, c * a.z);
}

__host__ __device__ inline float3
operator-(const float3 a, float c)
{
  return make_float3(a.x - c, a.y - c, a.z - c);
}

__host__ __device__ inline float3&
operator*=(float3& a, const float& c)
{
  a.x *= c;
  a.y *= c;
  a.z *= c;
  return a;
}

#endif // NETWORK_FLOAT3_OPS_CUH
