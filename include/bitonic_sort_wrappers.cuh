
#ifndef _BITONIC_SORT_W
#define _BITONIC_SORT_W
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<typename T>
extern void bitonic_sort_wrap(T * inputArray,uint32_t current_bitonic_length,uint32_t compare_dist,std::pair<dim3,dim3> dims);
template<typename T>
extern void bitonic_sort_wrap_shared_mem(T * inputArray,uint32_t current_bitonic_length,uint32_t compare_dist,std::pair<dim3,dim3> dims);

#endif //_BITONIC_SORT_W