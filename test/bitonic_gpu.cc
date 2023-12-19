#include <bitonic_sort.cuh>
#include <array>
#include <random>
int main()
{
    std::array<float,32> data;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-5.f,5.f);
    for(auto & d:data)
    {
        d = dist(mt);
        std::cout<<d<<" ";
    }
    std::cout<<std::endl;
    BitonicSorter<float,32> sorter;
    float * k_input;
    cudaMalloc((void **)&k_input,32*sizeof(float));
    cudaMemcpy(k_input,data.data(),32*sizeof(float),cudaMemcpyHostToDevice);
    sorter.sort_gpu(k_input);
    cudaMemcpy(data.data(),k_input,32*sizeof(float),cudaMemcpyDeviceToHost);
    for(auto & d:data)
    {
        std::cout<<d<<" ";
    }
    std::cout<<std::endl;
    cudaFree(k_input);
    return 0;
}