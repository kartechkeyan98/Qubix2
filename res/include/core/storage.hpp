#pragma once

// qb
#include<core/common.h>
#include<core/constants.hpp>

namespace qb{
namespace core{

template<typename T, int64_t crows, int64_t ccols>   
class matrix;

// define storage class
template<typename T, int64_t sizeatcompiletime, 
bool is_dynamic=(sizeatcompiletime == Dynamic)>
class storage{
private:
    T * m_data = nullptr;                 // will always be allocated @ runtime
    uint64_t m_size = sizeatcompiletime;  // m_size is @ runtime
    size_t m_nref  = 1;                   // no.of variables referencing this storage
    // when created, obv only one guy referencing it now....

    
     
public:
// some public members:
    

/**
 * Constructors
 */
    // runtime constructor (meaning you have to provide a size, bitch!)
    template<typename U = void, 
    std::enable_if_t<is_dynamic, U>* = nullptr>
    storage(uint64_t n)
    :m_size(n){
        #ifdef QB_DEBUG
        std::cout<<"Using Runtime Constructor!\n";
        #endif

        if(n>0){
            this->m_data = new T[n];
            if(this->m_data==nullptr){
                throw std::bad_alloc();
            }
        }
    }
    // compile time constructor (size already provided by you)
    template<typename U = void, 
    std::enable_if_t<!is_dynamic, U>* = nullptr>
    storage(){
        #ifdef QB_DEBUG
        std::cout<<"Using Compile Constructor!\n";
        #endif
        
        this->m_data = new T[this->m_size];
        if(this->m_data==nullptr){
            throw std::bad_alloc();
        }
    }

/**
 * Destructors
 */
    ~storage(){
        if(this->m_data!=nullptr)delete[] this->m_data;
        this->m_data = nullptr;
    }

/**
 * Copy, Assignment and move semantics
 */
    // copy constructor
    /**
     * Two cases: this,other static & this,other dynamic
     * Below logic works in both cases
     */
    template<typename U,int64_t csize>
    storage(const storage<U,csize>& st)
    :m_size(st.m_size)
    {   
        if constexpr(sizeatcompiletime!=Dynamic){
            if(st.m_size!=sizeatcompiletime){   // deals with when st is static and dynamic
                throw std::runtime_error("Size mismatch for both storages!");
            }
        }
        this->m_data= new T[st.m_size];

    #ifdef _openmp_included
        // parallel copying
        uint64_t optimal_threads = std::min((uint64_t)omp_get_num_procs(), 
        (m_size) / 10000);
        omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
        #pragma omp parallel for
    #endif
        for(uint64_t i=0;i<this->m_size;i++){
            this->m_data[i] = T(st[i]);
        }
    }
    

    // deep copy during assignments
    template<typename U,int64_t csize>
    storage<T,sizeatcompiletime>& operator=(
        const storage<U,csize>& st
    ){
        if(this == &st)return *this;    // prevent self assignment

        if constexpr(sizeatcompiletime!=Dynamic){
            if(st.m_size!=sizeatcompiletime){
                throw std::runtime_error("Size mismatch for both storages!");
            }
        }

        delete[] this->m_data;          // clear the old stuff
        this->m_data = new T[st.size()];
        this->m_size = st.size();  
    
    #ifdef _openmp_included
        // parallel copying
        uint64_t optimal_threads = std::min((uint64_t)omp_get_num_procs(), 
        (m_size) / 10000);
        omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
        #pragma omp parallel for
    #endif
        for(uint64_t i=0;i<this->m_size;i++){
            this->m_data[i] = T(st[i]);
        } 
        return *this;
    }

    // move constructor (only between same types, crossmoving not possible)
    // storage(storage<T,sizeatcompiletime>&& other)noexcept=default;
    // storage<T,sizeatcompiletime>& operator=(storage<T,sizeatcompiletime>&& other)noexcept=default;
    
    
/**
 * Accessors and Modifiers
 */

    template<typename V, 
    typename=std::enable_if_t<std::is_integral_v<V> > >
    T operator[](V i)const{
        if(i < 0) i+=this->m_size;
        if(i<0||i>this->m_size) throw std::runtime_error("Index out of range!");
        return this->m_data[i];
    }
    template<typename V, 
    typename=std::enable_if_t<std::is_integral_v<V> > > 
    T& operator[](V i){
        if(i < 0) i+=this->m_size;
        if(i<0||i>this->m_size) throw std::runtime_error("Index out of range!");
        return this->m_data[i];
    }
    size_t size()const{
        return this->m_size;
    }

    

/**
 * Friendships
 */

    // these have to be friends because well, m_nref has to be
    // accessed somehow
    template<typename U, int64_t r, int64_t c>
    friend class matrix;

    template<typename U,int64_t r, bool isdy>
    friend class storage;

/**
 * Some more operations, for example storage copying
 */

    // does the same thing as copy constructor... but we are explicit here
    template<typename U,int64_t s>
    storage<U,s> copy(){
        return storage<U,s>(*this); // c++17: rvo
    }

    // enabled only if we operate on dynamic storages!
    template<typename U=void>
    void resize(uint64_t N, std::enable_if_t<is_dynamic,U>* = nullptr){
        // deallocate the current data
        if(this->m_data!=nullptr)delete[] this->m_data;

        // reallocate your stuff
        this->m_data = new T[N];
        if(this->m_data==nullptr){
            throw std::bad_alloc();
        }
        this->m_size = N;
        return;
    }


};





}
}
