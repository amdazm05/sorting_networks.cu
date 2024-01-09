#ifndef _BITONIC_SORT
#define _BITONIC_SORT

#include <bitonic_sort_wrappers.cuh>
// std includes 
#include <iostream>
#include <vector>
#include <exception>
#include <algorithm>

/// @brief 
/// @tparam T 
/// @tparam data_length  --> Window is specified from outside
template<typename T,size_t data_length,bool shared_mem_mode=false>
class BitonicSorter 
{   
    static_assert((data_length & (data_length - 1)) == 0, "Bitonic Sorter ::: data_length must be a power of 2.");
    private:
        std::ptrdiff_t length;
    public:
        BitonicSorter(){}
        /// @brief For Stl containers
        /// @param begin 
        /// @param end 
        void sort_gpu(typename std::vector<T>::iterator begin,typename std::vector<T>::iterator end);
        void sort_cpu(typename std::vector<T>::iterator begin,typename std::vector<T>::iterator end);
        void sort_gpu(T * begin,T * end);
        /// @brief 
        /// @param start 
        /// @param length 
        void sort_gpu(T * start);
};

template<typename T,size_t data_length,bool shared_mem_mode=false>
void BitonicSorter<T,data_length,shared_mem_mode>::sort_gpu(T * start)
{
    dim3 blockthreads = data_length;
    dim3 gridblocks = 1;
    #pragma unroll
    for(std::size_t current_bitonic_length = 2
        ;current_bitonic_length<=data_length
        ;current_bitonic_length=current_bitonic_length<<1)
    {
        std::size_t compare_dist = current_bitonic_length>>1;
        while(compare_dist>0)
        {
            if constexpr (shared_mem_mode)
            {
                bitonic_sort_wrap_shared_mem(start,(uint32_t)current_bitonic_length,
                    (uint32_t)compare_dist,std::pair<dim3,dim3>(gridblocks,blockthreads));
            }
            else
            {
                bitonic_sort_wrap<T>(start,(uint32_t)current_bitonic_length,
                    (uint32_t)compare_dist,std::pair<dim3,dim3>(gridblocks,blockthreads));
            }
            compare_dist=compare_dist>>1;
        }
    }
    return;
}

#endif // _BITONIC_SORT