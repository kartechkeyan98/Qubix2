// #define QB_DEBUG

#include<core/common.h>         // contains std includes
#include<core/storage.hpp>
using namespace qb::core;


int main(void){
    storage<float, 3> s;
    s[0] = 0, s[1] = 1, s[2] = 3;
    std::cout<<"Printing s...\n";
    for(int i=0;i<3;i++){
        try{std::cout<<s[i]<<" ";}
        catch(const std::runtime_error& e){
            std::cout<<"NO_ACCESS["<<i<<"] ";
        }
    }
    std::cout<<std::endl;

    // this should actually result in a compile time error....
    // try{
    //     std::cout<<"Trying to resize s...\n";
    //     s.resize(6);
    //     std::cout<<"Successfully resized! New size = "<<s.size()<<std::endl;
    // }catch(const std::runtime_error& e){
    //     std::cout<<"Cannot resize the storage s!\n";
    // }

    int n;
    std::cout<<"Provide me a number! ";
    std::cin>>n;
    storage<int,Dynamic> q(n);
    for(int i=0;i<n;i++)q[i]=2*i;
    std::cout<<"Printing s...\n";

#ifdef _openmp_included
    omp_set_num_threads(4);
    #pragma omp parallel for
#endif
    for(int i=0;i<n;i++){
        try{std::cout<<q[i]<<" ";}
        catch(const std::runtime_error& e){
            std::cout<<"NO_ACCESS["<<i<<"] ";
        }
    }
    std::cout<<std::endl;

    try{
        std::cout<<"Trying to resize q...\n";
        q.resize(7);
        std::cout<<"Successfully resized! New size = "<<q.size()<<std::endl;
    }catch(const std::runtime_error& e){
        std::cout<<"Could resize the storage q!\n";
    }

    return 0;
}