## Kernels
Kernel函数，由$\_\_global\_\_$修饰，返回值为$void$。执行可以使用"<<<...>>>"语法的方式，也可以使用cudaLaunchKernel函数来完成调用。
```c++
__global__ void kernel(...)
{
  ...
}

int main()
{
    ...

    // kernel launch
    dim3 grid_dim;
    dim3 block_dim;

    kernel<<<grid_dim, block_dim, extern_shm_size, stream>>>(...);
    或
    if (cudaSuccess != cudaLaunchKernel(kernel, grid_dim, block_dim, .../*args*/, 0/*shared memory size*/, 0/*cudaStream_t*/)) {
      ...
    }

    ...
}
```
Block Cluster调用方式，可以编译期指定cluster dimension，也可以使用cudaLaunchKernelEx函数在运行时指定。

```c++
// cluster attribute assigned within compile time
__global__ void __cluster_dims__(2, 1, 1) cluseter_kernel_1(...)
{
  ...
}

// No compile time attribute attached to the kernel
__global__ void cluster_kernel_2(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    cluster_kernel_1<<<numBlocks, threadsPerBlock>>>();
    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel_2, input, output);
    }
}
```

## Thread Hierarchy
- Up to 3 dimension thread block, maximum threads number per block: 1024
- Up to 3 dimension block grid, block execute independently
- Thread/block id relate to idx in a straightforward way: $(Idx.z * Dim.y  + Idx.y) * Dim.x + Idx.x$
- Threads within a block can cooperate by sharing data through some shared memory and by synchronizing their execution to coordinate memory accesses.
  1. __syncthreads(), acts as a barrier at which all threads in the block must wait before every is allowed to proceed
  2. Cooperative Groups API
  3. Shared memory is expected to be a low-lantency memory(much like an L1 cache), and __syncthreads() is expected to be lightweight
  4. Compute Capability 9.0, block cluster, blocks in cluster are guaranteed to be co-scheduled on a streaming multiprocessor, blocks in cluster also are guaranteed to be co-scheduled on a GPU Processing Cluster(GPC) int the GPU, maximum blocks number per cluster: 8


## Memory Hierarchy
- Each thread has private local memory
- Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block
- Thread blocks in a thread block cluster can perform read, write and atomics operations on each other's shared memory
- All threads have access to the same global memory
- Read-only memory spaces accessible by all threads: the constant and texture memory spaces, texture memory also offers different addressing modes, as well as data filtering. they are persistent across kernel launches by the same application

## Heterogeneous Programming
- Both the host and device maintain their own seperate memory spaces in DRAM
- Porgram manages the global, constant, and texture memory space visible to kernels through calls to the CUDA runtime, includes device memory allocation and deallocation as well as data transfer between host and device memory

## Asynchronous SIMT Programming Model
- Starting with devices based on the NVIDIA Ampere GPU architecture, the CUDA programming model provides acceleration to memory operations via the asynchronous programming model
- The asynchronous programming model defines the behavior of asynchronous operations with respect to CUDA threads
- cuda::thread_scope::thread_scope_thread
- cuda::thread_scope::thread_scope_block
- cuda::thread_scope::thread_scope_device
- cuda::thread_scope::thread_scope_system

## Compute Capability
RTX 86
Orin 87
Xavier 72
