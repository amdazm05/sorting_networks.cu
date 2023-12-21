#include <allocators/gpu_mem_allocator.tpp>
#include <vector>
#include <iostream>
int main()
{
    std::vector<int,cuda_mem::pinned_memory_allocator<int>> vec(32);
    std::cout<<vec.size()<<std::endl;
    return 0;
}