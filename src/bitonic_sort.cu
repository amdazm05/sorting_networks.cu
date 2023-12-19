#include "bitonic_sort.cuh"


///@breif 
// cbs = is the current bitonic sort 
// nbs = is the next bitonic sort element interval
// Something on XORs = A+B = (A^B) + 2*(A.B) which means (A+B%2) is A^B 
// provided A.B is in powers of 2
// https://stackoverflow.com/questions/21293278/mathematical-arithmetic-representation-of-xor#:~:text=When%20it%20comes%20to%20translating,%2Dc)...)
template<typename T>
void bitonic_sort(std::vector<T> & arr)
{
    if(arr.size()%2!=0) 
        throw std::runtime_error("Bitonic Sort : not possible");
    int n = arr.size();
    for(std::size_t cbs =2;cbs<=n ; cbs*=2)
    {
        std::size_t compare_dist = cbs/2;
        while(compare_dist>0)
        {
            for(std::size_t currInd = 0; currInd<n;currInd++)
            {
                std::size_t ij = currInd ^ compare_dist;
                if(ij>currInd)
                {
                    // this ensures halves are divided 
                    if ((currInd & cbs) == 0)
                    {
                        if (arr[currInd] > arr[ij])
                            std::swap(arr[currInd],arr[ij]);
                    } 
                    else
                    {
                        if (arr[currInd] < arr[ij])
                            std::swap(arr[currInd],arr[ij]);
                    }
                }
            } 
            compare_dist/=2;
        }
    }
    return;
}

template<typename T>
void bitonic_sort_wrap(T * inputArray,uint32_t current_bitonic_length,uint32_t compare_dist,std::pair<dim3,dim3> &&dims)
{
    bitonic_sort<T><<<dims.first,dims.second>>>(inputArray,current_bitonic_length,compare_dist);
}

template<typename T>
__global__ void bitonic_sort(T * inputArray,uint32_t current_bitonic_length,uint32_t compare_dist)  
{
    int idx = threadIdx.x;
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
                // * Evaluate if slower // @todo
                T temp = &inputArray[idx],inputArray[ij];
                &inputArray[ij],temp;
            }
        }
        else
        {
            if(inputArray[idx]<inputArray[ij])
            {
                T temp = inputArray[idx],inputArray[ij];
                &inputArray[ij],temp;
            }
        }
    }
}


/// Integers Specialisations
template<> void bitonic_sort<int>  (int * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<int>  (int * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<int>  (int * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<int>  (int * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<int>  (int * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<int>  (int * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<int>  (int * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<int>  (int * inputArray,uint32_t cbs,uint32_t compare_dist);
 
//  Floats
template<> void bitonic_sort<float> (float * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<float> (float * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<float> (float * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<float> (float * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<float> (float * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<float>(float * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<float>(float * inputArray,uint32_t cbs,uint32_t compare_dist);
template<> void bitonic_sort<float>(float * inputArray,uint32_t cbs,uint32_t compare_dist);
