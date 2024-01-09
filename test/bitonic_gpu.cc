#include <bitonic_sort.tpp>
#include <array>
#include <random>
#include <chrono>
#define WINDOW_SIZE 8192
int main()
{
    std::array<float,WINDOW_SIZE> data;
    std::array<float,WINDOW_SIZE> ver_data;
    std::array<float,WINDOW_SIZE> data2;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-5.f,5.f);
    for(auto & d:data)
    {
        d = dist(mt);
    }
    ver_data = data;
    data2 = data;

    std::sort(ver_data.begin(),ver_data.end());
    BitonicSorter<float,WINDOW_SIZE> sorter;
    BitonicSorter<float,WINDOW_SIZE,true> sorter2;
    float * k_input;
    cudaMalloc((void **)&k_input,WINDOW_SIZE*sizeof(float));
    cudaMemcpy(k_input,data.data(),WINDOW_SIZE*sizeof(float),cudaMemcpyHostToDevice);

    float * k_input2;
    cudaMalloc((void **)&k_input2,WINDOW_SIZE*sizeof(float));
    cudaMemcpy(k_input2,data2.data(),WINDOW_SIZE*sizeof(float),cudaMemcpyHostToDevice);

    std::chrono::time_point t1 = std::chrono::high_resolution_clock::now();
    sorter.sort_gpu(k_input);
    std::chrono::time_point t2 = std::chrono::high_resolution_clock::now();
    auto diff = t2-t1;
    std::cout<<"Benchmark time V1::: "<<
        std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count()/256<<" ns"
        <<std::endl; 


    t1 = std::chrono::high_resolution_clock::now();
    sorter2.sort_gpu(k_input2);
    t2 = std::chrono::high_resolution_clock::now();
    diff = t2-t1;
    
    std::cout<<"Benchmark time V2 ::: "<<
        std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count()/256<<" ns"
        <<std::endl; 
    cudaMemcpy(data.data(),k_input,WINDOW_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(data2.data(),k_input2,WINDOW_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
    for(std::size_t i=0;i<data.size();i++)
    {
        if(ver_data[i] != data[i] || data2[i]!=ver_data[i])
            throw std::runtime_error("Bitonic Sort has failed");
    }
    std::cout<<"TESTs Passed"<<std::endl;
    cudaFree(k_input);
    return 0;
}