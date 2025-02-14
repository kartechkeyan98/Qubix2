#include<iostream>

#include<qbshape.h>

namespace qb{

template<typename T>
class tensor{
private:
    T *_data=nullptr;           // array of elements [8B]
    qb::shape _shape;           // shape of our tensor [16B]

public:
/**
 * Constructors:
 * 1. Default Constructor: Does nothing
 * 2. Empty Tensor of a given size
 */
    tensor(){}

    template<typename... Args,
    typename = std::enable_if_t<(std::is_convertible_v<Args, size_t> && ...)> >
    tensor(Args... args)
    _shape(qb::shape(args...)){
        
    }

/**
 * Important functions:
 * 1. Toggle requires grad
 * 2. Reshape tensor inplace
 */

    

};

}