#ifndef _BITONIC_SORT
#define _BITONIC_SORT
#include "../baseLib/cudaBaseLib.cuh"

// std includes 
#include <iostream>
#include <vector>
#include <exception>
#include <algorithm>

/// @brief  Implements a bitonic sort on window of fixed length
/// @tparam T 
/// @tparam window_size                     //This is the size of window that will be required
/// @param inputArray 
/// @param current_bitonic_length           // current bitonic window size ... [a1,a2,a3,a4...an] n = cbs
/// @param compare_dist                     // compare distance is equal to current_bitonic_length/2
/// @return 
template<typename T>
extern __global__ void bitonic_sort(T * inputArray,uint32_t current_bitonic_length,uint32_t compare_dist);
template<typename T>
extern void bitonic_sort_wrap(T * inputArray,uint32_t current_bitonic_length,uint32_t compare_dist,std::pair<dim3,dim3> &&dims);
/// @brief 
/// @tparam T 
/// @tparam data_length  --> Window is specified from outside
template<typename T,size_t data_length>
class BitonicSorter 
{   
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

template<typename T,size_t data_length>
void BitonicSorter<T,data_length>::sort_gpu(T * start)
{
    dim3 blockthreads = 32;
    dim3 gridblocks = 1;
    for(std::size_t current_bitonic_length = 2
        ;current_bitonic_length<=data_length
        ;current_bitonic_length=current_bitonic_length<<1)
    {
        std::size_t compare_dist = current_bitonic_length/2;
        while(compare_dist>0)
        {
            bitonic_sort_wrap<T>(start,current_bitonic_length,compare_dist,std::pair<dim3,dim3>(gridblocks,blockthreads));
            compare_dist=compare_dist>>1;
        }
    }
    return;
}

#endif // _BITONIC_SORT