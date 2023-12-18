#include <iostream>
#include <vector>
#include <exception>
#include <algorithm>

///@breif 
// cbs = is the current bitonic sort 
// nbs = is the next bitonic sort element interval
// Something on XORs = A+B = (A^B) + 2*(A.B) which means (A+B%2) is A^B 
// provided A.B is in powers of 2
//https://stackoverflow.com/questions/21293278/mathematical-arithmetic-representation-of-xor#:~:text=When%20it%20comes%20to%20translating,%2Dc)...)
void bitonic_sort(std::vector<int> & arr)
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

int main()
{
    std::vector<int> vec= {2,4,1,3};
    bitonic_sort(vec);
    std::for_each(vec.begin(),vec.end(),[](int & a){std::cout<<a<<" ";});
    return 0;
}