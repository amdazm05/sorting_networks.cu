#include <allocators/gpu_mem_allocator.tpp>
#include <vector>
#include <iostream>
#include <iterator>
#include <ostream>
int main()
{
    std::vector<int,cuda_mem::pinned_memory_allocator<int>> vec(32);
    std::cout<<vec.size()<<std::endl;
    vec[10] = 10;
    std::copy(std::begin(vec), std::end(vec),
        std::ostream_iterator<int>(std::cout, ", "));
    return 0;
}