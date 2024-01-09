#include "bitonic_sort_wrappers.cuh"

///@breif 
// cbs = is the current bitonic sort 
// nbs = is the next bitonic sort element interval
// Something on XORs = A+B = (A^B) + 2*(A.B) which means (A+B%2) is A^B 
// provided A.B is in powers of 2
// https://stackoverflow.com/questions/21293278/mathematical-arithmetic-representation-of-xor#:~:text=When%20it%20comes%20to%20translating,%2Dc)...)

template<typename T>
__global__ void bitonic_sort(T * inputArray,uint32_t current_bitonic_length,uint32_t compare_dist)  
{
    int idx = threadIdx.x +blockDim.x * blockIdx.x;;
    // This evaulates to A^B = A+B - 2(A.B) by properties of XOR this will come out to be // somewhat like a modulus 
    // the next/ previous point that is seperated by compare_dist with idx
    // example 3 ^ 1 = 2 which is (3+1) - 2(1) = 2, 2 is seperated by compare_dist with 3
    // Basically what this does is that it computes the next index , with which to compare to , accounting for the compare distance
    int ij  = compare_dist ^ idx;
    // 3 ^ 1 = 2 , 2 ^ 1 = 3 , 2 and 3 are only swapped once 
    // to avoid unneccessary comparisons  we only make comparisons such that ij < idx
    if(ij>idx)
    {
        // dividing the array into two portions // basically half of bitonic length
        // example
        // 0b100 is my length so I need all elements compared in a specific manner to 0b001,0b010,0b011
        if((idx & current_bitonic_length)==0)
        {
            if(inputArray[idx]>inputArray[ij])
            {
                T temp = inputArray[idx];
                inputArray[idx] = inputArray[ij];
                inputArray[ij] = temp;
            }
        }
        else
        {
            if(inputArray[idx]<inputArray[ij])
            {
                T temp = inputArray[idx];
                inputArray[idx] = inputArray[ij];
                inputArray[ij] = temp;
            }
        }
    }
}

template<typename T>
__global__ void bitonic_sortV2(T * inputArray,uint32_t current_bitonic_length,uint32_t compare_dist)  
{
    __shared__ T sharedMemArray[8192];
    int idx = threadIdx.x +blockDim.x * blockIdx.x;
    int ij  = compare_dist ^ idx;
    sharedMemArray[ij] = inputArray[ij];
    sharedMemArray[idx] = inputArray[idx];
    
    __syncthreads();
    if(ij>idx)
    {
        if((idx & current_bitonic_length)==0)
        {
            if(sharedMemArray[idx]>sharedMemArray[ij])
            {
                T temp = sharedMemArray[idx];
                sharedMemArray[idx] = sharedMemArray[ij];
                sharedMemArray[ij] = temp;
            }
        }
        else
        {
            if(sharedMemArray[idx]<sharedMemArray[ij])
            {
                T temp = sharedMemArray[idx];
                sharedMemArray[idx] = sharedMemArray[ij];
                sharedMemArray[ij] = temp;
            }
        }
        inputArray[ij] = sharedMemArray[ij];
        inputArray[idx]  = sharedMemArray[idx] ;
    }
}

template<>
void bitonic_sort_wrap<float>(float* inputArray, uint32_t current_bitonic_length, uint32_t compare_dist, std::pair<dim3, dim3> dims)
{
    bitonic_sort<<<dims.second,dims.first>>>(inputArray, current_bitonic_length, compare_dist);
}

template<>
void bitonic_sort_wrap_shared_mem<float>(float* inputArray, uint32_t current_bitonic_length, uint32_t compare_dist, std::pair<dim3, dim3> dims)
{
    bitonic_sortV2<<<dims.second,dims.first>>>(inputArray, current_bitonic_length, compare_dist);
}