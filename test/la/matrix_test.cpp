#include<core/common.h>
#include<core/matrix.hpp>
using namespace qb::core;

#include<cstdio>


int main(void){
    // create matrices

    // static matrix
    matVi<4,5> m1;
    m1<<1,2,3,4,5,
        6,7,8,9,10,
        11,12,13,14,15,
        16,17,18,19,20;
    print_info(m1);

    // dynamic matrices
    matXf m2(2,3);
    m2<<1,2,3,
        4,5,6;
    print_info(m2);

    // partially static (vectors...)
    vecXd v1(3);
    v1<<1.6,2.5,3.8;
    print_info(v1);

    // fully static vectors
    vecVf<4> v2;
    v2<<2.5,3.6,9.7,0.5;
    print_info(v2);

    // test the copy semantics
    vecVf<3> v3(v1);
    print_info(v3);

    rvecXcf v4(3);
    using cf = std::complex<float>;
    v4<<cf(3,4),cf(4,5),cf(5,6);
    print_info(v4);

    rvecXci v5 = v4;
    print_info(v5);
}