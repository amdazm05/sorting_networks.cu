#include <bitonic_sort.cuh>
#include <array>
#include <random>
#define WINDOW_SIZE 4096
int main()
{
    std::array<float,WINDOW_SIZE> data;
    std::array<float,WINDOW_SIZE> ver_data;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-5.f,5.f);
    for(auto & d:data)
    {
        d = dist(mt);
    }
    ver_data = data;
    std::sort(ver_data.begin(),ver_data.end());
    BitonicSorter<float,WINDOW_SIZE> sorter;
    float * k_input;
    cudaMalloc((void **)&k_input,WINDOW_SIZE*sizeof(float));
    cudaMemcpy(k_input,data.data(),WINDOW_SIZE*sizeof(float),cudaMemcpyHostToDevice);
    sorter.sort_gpu(k_input);
    cudaMemcpy(data.data(),k_input,WINDOW_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
    for(std::size_t i;i<data.size();i++)
    {
        if(ver_data[i] != data[i])
            throw std::runtime_error("Bitonic Sort has failed");
    }
    std::cout<<"TEST Passed"<<std::endl;
    cudaFree(k_input);
    return 0;
}