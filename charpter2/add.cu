#include <cuda_runtime_api.h>
#include <iostream>

#define CUDA_CHECK(x)               \
    do {                            \
        auto s_ = (x);              \
        if (s_ != cudaSuccess) {    \
            std::cout << __FILE__ << ":" <<  __LINE__   \
                      << " cuda error [" << s_ << "\n"; \
            exit(1);                \
        }                           \
    } while(0)

#define CUDA_LAUNCH_CHECK()             \
    do {                                \
        CUDA_CHECK(cudaGetLastError()); \
    } while(0)

template <typename T> 
__global__ void vec_add(int n, const T* a, const T* b, T* c)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    c[tid] = a[tid] + b[tid];
}

template <typename T>
__global__ void mat_add(int rows, int cols, const T* a, const T* b, T* c)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= rows || x >= cols) return;

    int off = y * cols + x;
    c[off] = a[off] + b[off]; 
}

template <typename T>
__global__ void vol_add(int rows, int cols, int depth,
                        const T* a, const T* b, T* c)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z >= depth || y >= rows || x >= cols) return;

    int off = (z * rows + y) * cols + x;
    c[off] = a[off] + b[off];
}

int main(int argc, char** argv)
{
    float *a = nullptr, *b = nullptr, *c = nullptr;

    size_t vec_length = 1024;
    CUDA_CHECK(cudaMalloc(&a, vec_length * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, vec_length * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, vec_length * sizeof(float)));

    // fill in

    vec_add<<<256, 4>>>(vec_length, a, b, c);
    CUDA_LAUNCH_CHECK();

    // copy out
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));

    size_t mat_rows = 512, mat_cols = 512;
    CUDA_CHECK(cudaMalloc(&a, mat_rows * mat_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, mat_rows * mat_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, mat_rows * mat_cols * sizeof(float)));

    // fill in

    dim3 grid_dim(16, 16);
    dim3 block_dim(32, 32);
    mat_add<<<grid_dim, block_dim>>>(mat_rows, mat_cols, a, b, c);
    CUDA_LAUNCH_CHECK();

    // copy out

    return 0;
}