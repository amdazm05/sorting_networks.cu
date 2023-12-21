#ifndef _GPU_MEM_ALLOC
#define _GPU_MEM_ALLOC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Great article
// https://committhis.github.io/2020/10/06/cuda-abstractions.html
// https://en.cppreference.com/w/cpp/named_req/Allocator
namespace cuda_mem
{
    template<typename T>
    class pinned_memory_allocator
    {
        public:
            using size_type  = std::size_t;
            using value_type = T;
            using pointer  = T *;
            pinned_memory_allocator() noexcept = default;
            template <typename U>
            pinned_memory_allocator(pinned_memory_allocator<U> const&) noexcept {}
            [[nodiscard]] T * allocate(size_type n)
            {
                size_t max_malloc_size;
                cudaError_t err = cudaDeviceGetLimit(&max_malloc_size, cudaLimitMallocHeapSize);
                if (err != cudaSuccess)
                    throw std::runtime_error{ cudaGetErrorString(err) };
                T* pinnedMemory = nullptr;
                err = cudaMallocHost((void**)&pinnedMemory, n * sizeof(value_type));
                if(err!=cudaSuccess)
                    throw std::runtime_error { cudaGetErrorString(err) };                
                return pinnedMemory;
            }
            void deallocate(T * pinnedMemory,size_type n)
            {
                cudaError_t err = cudaFreeHost (pinnedMemory);
                if(err!=cudaSuccess)
                    throw std::runtime_error { cudaGetErrorString(err) };      
            }
    };
    template<class T, class U>
    bool operator==(const pinned_memory_allocator <T>&, const pinned_memory_allocator <U>&) { return true; }
    
    template<class T, class U>
    bool operator!=(const pinned_memory_allocator <T>&, const pinned_memory_allocator <U>&) { return false; }
}
#endif //_GPU_MEM_ALLOC
