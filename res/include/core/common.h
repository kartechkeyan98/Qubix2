#pragma once

// std
#include<iostream>
#include<type_traits>
#include<new>
#include<stdexcept>
#include<memory>
#include<functional>
#include<iomanip>
#include<complex>


// openmp
#ifdef _OPENMP
#define _openmp_included
#include<omp.h>
#endif