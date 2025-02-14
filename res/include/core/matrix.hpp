#pragma once

//qb 
#include<core/common.h>
#include<core/constants.hpp>
#include<core/storage.hpp>


namespace qb{
namespace core{
/**
 * This datatype is going to be a hybrid storage type
 * Depending on how the user sees fit, it can be static
 * or dynamic
 */
template<typename T, int64_t crows, int64_t ccols>
class matrix{
private:
/**
 * Things we do for type safety!
 */
    // some things you might wanna have for bookkeeping
    static constexpr bool dynamic_rows = (crows == Dynamic);
    static constexpr bool dynamic_cols = (ccols == Dynamic);
    static constexpr bool dynamic_strg = (dynamic_rows||dynamic_cols);

    // what is the storage size, decide at compile time!
    static constexpr int64_t strg = std::conditional_t<
        dynamic_strg,
        std::integral_constant<int64_t, Dynamic>,
        std::integral_constant<int64_t,crows*ccols>
    >::value;

    // private attribs
    storage<T,strg>* m_storage = nullptr; // init to a nullptr
    uint64_t rows = crows;                // rows at runtime
    uint64_t cols = ccols;                // cols at runtime
      
public:
/**
 * Constructors:
 * default: nothing happens!
 */

    // dynamic rows and cols
    template<typename U=void, 
    bool drows = dynamic_rows, bool dcols = dynamic_cols, 
    std::enable_if_t<drows&&dcols, U>* = nullptr >
    matrix(int m,int n)
    :rows(m),cols(n){
        // every matrix comes with it's own storage
        this->m_storage = new storage<T, Dynamic>(m*n);
        
    }
    // only rows dynamic here!
    template<typename U=void, 
    bool drows = dynamic_rows, bool dcols = dynamic_cols, 
    std::enable_if_t<drows&&!dcols, U>* = nullptr >
    matrix(int m)
    :rows(m){
        // every matrix comes with it's own storage
        this->m_storage = new storage<T, Dynamic>(m*ccols);
    }
    // only cols dynamic here!
    template<typename U=void, 
    bool drows = dynamic_rows, bool dcols = dynamic_cols, 
    std::enable_if_t<!drows&&dcols, U>* = nullptr >
    matrix(int n)
    :cols(n){
        // every matrix comes with it's own storage
        this->m_storage = new storage<T, Dynamic>(crows*n);
    }
    // everything static!
    template<typename U=void, 
    bool drows = dynamic_rows, bool dcols = dynamic_cols, 
    std::enable_if_t<!drows&&!dcols, U>* = nullptr >
    matrix(){
        // every matrix comes with it's own storage
        this->m_storage = new storage<T,strg>();
    }

/**
 * Destructor
 */

    ~matrix(){
        // understand first thing that the storage of this
        // matrix may be shared with other objects, so think 
        // carefully before you go deleting it!
        if(this->m_storage){
            // more than one thing is referencing the storage, don't delete!
            if(this->m_storage->m_nref > 1){
                this->m_storage->m_nref--;
            }
            // go ahead!
            else{
                delete this->m_storage;
            } 
            this->m_storage=nullptr;
        }
    }

/**
 * Comma Initializer! Nice Feature of this shit!
 */

    class commaInit{
    private:
        matrix<T,crows,ccols>& mat;
        int index = 0;  // tracks position in matrix
    public: 
        commaInit(matrix<T,crows,ccols>& m, const T& val)
        :mat(m){
            mat[0]=val;
            index = 1;
        }
        commaInit& operator,(const T& value){
            if(index>=mat.size()){
                throw std::runtime_error("Size of comma-initializer list exceeds matrix size!");
            }
            mat[index]=value;
            ++index;
            return *this;
        }
    };
    commaInit operator<<(T value){
        return commaInit(*this,value);
    }

/**
 * Copy Constructor and Assignment Operator!
 * This library's philosophy is to deep copy by defualt
 * Only shallow copy during views and slices!
 */
    template<typename U, int64_t r, int64_t c>
    matrix(const matrix<U,r,c>& mat)
    :rows(mat.nrows()),cols(mat.ncols())
    {
        if constexpr(crows!=Dynamic||ccols!=Dynamic){
            // if both dims static, make sure shape matches
            if((crows!=Dynamic&&crows!=mat.rows)
            ||(ccols!=Dynamic&&ccols!=mat.cols)){
                throw std::runtime_error("Shape mismatch error!");
            }
        }
        this->m_storage= new storage<T,strg>(*mat.m_storage);
    }

    template<typename U, int64_t r, int64_t c>
    matrix<T,crows,ccols>& operator=(const matrix<U,r,c>& mat)
    {
        if constexpr(crows!=Dynamic||ccols!=Dynamic){
            // if both dims static, make sure shape matches
            if((crows!=Dynamic&&crows!=mat.rows)
            ||(ccols!=Dynamic&&ccols!=mat.cols)){
                throw std::runtime_error("Shape mismatch error!");
            }
        }
        
        if(this==&mat)return *this;
        if(this->m_storage){
            if(this->m_storage->m_nref>1){
                this->m_storage->m_nref--;
            }
            else{
                delete[] this->m_storage;
            }
        }

        this->m_storage = new storage<T,strg>(*mat.m_storage);
        return *this;
    }

    


/**
 * Accessing operators
 * Cut some slack for the python users by enabling reverse indexing
 * So all we need to overload is ()
 */

    // usual indexing 2D
    template<typename V, 
    typename=std::enable_if_t<std::is_integral_v<V> > >
    T operator()(V i, V j)const{
        if(i>=this->rows||j>=this->cols){
            throw std::runtime_error("Index out of range!");
        }
        i = (i<0)? i+this->rows: i;
        j = (j<0)? j+this->cols: j;
        if(i<0||j<0){
            throw std::runtime_error("Index out of range!");
        }

        return (*m_storage)[i*this->cols + j];
    }
    template<typename V, 
    typename=std::enable_if_t<std::is_integral_v<V> > >
    T& operator()(V i, V j){
        if(i>=this->rows||j>=this->cols){
            throw std::runtime_error("Index out of range!");
        }
        i = (i<0)? i+this->rows: i;
        j = (j<0)? j+this->cols: j;
        if(i<0||j<0){
            throw std::runtime_error("Index out of range!");
        }

        return (*m_storage)[i*this->cols + j];
    }
    // flat index
    template<typename V, 
    typename=std::enable_if_t<std::is_integral_v<V> > >
    T operator[](V i)const{
        uint64_t N = rows*cols;
        if(i>=N){
            throw std::runtime_error("Index out of range!");
        }
        i = (i<0)? i+N: i;
        if(i<0){
            throw std::runtime_error("Index out of range!");
        }
        return (*m_storage)[i];
    }
    template<typename V, 
    typename=std::enable_if_t<std::is_integral_v<V> > >
    T& operator[](V i){
        uint64_t N = rows*cols;
        if(i>=N){
            throw std::runtime_error("Index out of range!");
        }
        i = (i<0)? i+N: i;
        if(i<0){
            throw std::runtime_error("Index out of range!");
        }
        return this->m_storage->m_data[i];
    }

    const storage<T,strg>* data()const{
        return this->m_storage;
    }

    int64_t size()const{
        return this->m_storage->m_size;
    }
    int64_t nrows()const{
        return this->rows;
    }
    int64_t ncols()const{
        return this->cols;
    }
     

/**
 * Unary ops(+,-)
 */

matrix<T,crows,ccols> operator-()const{
    matrix<T,crows,ccols> res = *this; // calls copy constructor

#ifdef _openmp_included
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        uint64_t(rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for
#endif
    for(size_t i=0;i<rows*cols;i++){
        res[i]=-res[i];
    }
    return res; // rvo mandatory after c++17
}
matrix<T,crows, ccols> operator+()const{
    return *this;   // copy constructor
}

/**
 * Arithmetic Operations, matrix-matrix
 */

// addition
template<typename U,int64_t r,int64_t c, 
typename R = decltype(std::declval<T>()+std::declval<U>())>
matrix<R,Dynamic,Dynamic> operator+(const matrix<U,r,c>& a){
    // size check: keeping it fully runtime only ow
    // the code will be huge and tedious to debug
    if(rows!=a.nrows()||cols!=a.ncols()){
        throw std::runtime_error("Size mismatch! Cannot perform operation");
    }

    // create matrix from *this
    matrix<R,Dynamic,Dynamic> res(rows,cols);

#ifdef _openmp_included
    // parallelized addition
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        uint64_t(rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for
#endif
    for(size_t i=0;i<a.size();++i){
        res[i]=(*m_storage)[i]+a[i];
    }
    return res; // rvo: c++17
}
// subtraction
template<typename U,int64_t r,int64_t c, 
typename R = decltype(std::declval<T>()-std::declval<U>())>
matrix<R,Dynamic,Dynamic> operator-(const matrix<U,r,c>& a){
    // size check: keeping it fully runtime only ow
    // the code will be huge and tedious to debug
    if(rows!=a.nrows()||cols!=a.ncols()){
        throw std::runtime_error("Size mismatch! Cannot perform operation");
    }

    // create matrix from *this
    matrix<R,Dynamic,Dynamic> res(rows,cols);

#ifdef _openmp_included
    // parallelized addition
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        uint64_t(rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for
#endif
    for(size_t i=0;i<a.size();++i){
        res[i]=(*m_storage)[i]-a[i];
    }
    return res; // rvo: c++17
}
// gemm 
template<typename U,int64_t r,int64_t c, 
typename R = decltype(std::declval<T>()*std::declval<U>())>
matrix<R,Dynamic,Dynamic> operator*(const matrix<U,r,c>& a){
    // size check: keeping it fully runtime only ow
    // the code will be huge and tedious to debug
    if(cols!=a.nrows()){
        throw std::runtime_error("Dimension Mismtach! Cannot matrix-multiply these");
    }

    // create matrix from *this
    matrix<R,Dynamic,Dynamic> res(rows,a.ncols());
    
    // parallelized addition
#ifdef _openmp_included
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        (rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for collapse(2)
#endif
    for(size_t i=0;i<rows;++i){
        for(size_t j=0;j<a.ncols();++j){

            res(i,j)=static_cast<R>(0);
            for(size_t k=0;k<cols;++k){
                res(i,j)=res(i,j)+(*m_storage)[i*cols+k]*a(k,j);
            }

        }
    }

    

    return res; // rvo: c++17
}


/**
 * Arithmetic operations, with scalars [matrix-scalar]
 */

template<typename U, 
typename R = decltype(std::declval<T>()+std::declval<U>())>
matrix<R,Dynamic,Dynamic> operator+(const U& k)const{
    matrix<R,Dynamic,Dynamic> res(rows,cols);

#ifdef _openmp_included
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        (rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for
#endif
    for(size_t i=0;i<rows*cols;i++){
        res[i] = (*m_storage)[i] + k;
    }
    return res;
}
template<typename U, 
typename R = decltype(std::declval<T>()-std::declval<U>())>
matrix<R,Dynamic,Dynamic> operator-(const U& k)const{
    matrix<R,Dynamic,Dynamic> res(rows,cols);

#ifdef _openmp_included
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        (rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for
#endif
    for(size_t i=0;i<rows*cols;i++){
        res[i] = (*m_storage)[i] - k;
    }
    return res;
}
template<typename U, 
typename R = decltype(std::declval<T>()*std::declval<U>())>
matrix<R,Dynamic,Dynamic> operator*(const U& k)const{
    matrix<R,Dynamic,Dynamic> res(rows,cols);

#ifdef _openmp_included
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        (rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for
#endif
    for(size_t i=0;i<rows*cols;i++){
        res[i] = (*m_storage)[i] * k;
    }
    return res;
}
template<typename U, 
typename R = decltype(std::declval<T>()/std::declval<U>())>
matrix<R,Dynamic,Dynamic> operator/(const U& k)const{
    matrix<R,Dynamic,Dynamic> res(rows,cols);

#ifdef _openmp_included
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        (rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for
#endif
    for(size_t i=0;i<rows*cols;i++){
        res[i] = (*m_storage)[i] / k;
    }
    return res;
}

/**
 * Inplace operators (matrix-matrix)
 */

// +=
template<typename U, int64_t r, int64_t c,
typename = std::enable_if_t<std::is_convertible_v<U,T> > >
matrix<T,crows,ccols>& operator+=(const matrix<U,r,c>& a){
    if(rows!=a.nrows()||cols!=a.ncols()){
        throw std::runtime_error("Cannot add two matrices of different sizes!");
    }

#ifdef _openmp_included
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        (rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for
#endif
    for(size_t i=0;i<a.size();i++){
        (*m_storage)[i] = (*m_storage)[i] + a[i];
    }
    return *this;
}
// -=
template<typename U, int64_t r, int64_t c,
typename = std::enable_if_t<std::is_convertible_v<U,T> > >
matrix<T,crows,ccols>& operator-=(const matrix<U,r,c>& a){
    if(rows!=a.nrows()||cols!=a.ncols()){
        throw std::runtime_error("Cannot subtract two matrices of different sizes!");
    }

#ifdef _openmp_included
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        (rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for
#endif
    for(size_t i=0;i<a.size();i++){
        (*m_storage)[i] = (*m_storage)[i] - a[i];
    }
    return *this;
}
// *=
template<typename U, int64_t r, int64_t c,
bool drows=dynamic_rows, bool dcols=dynamic_cols,
typename = std::enable_if_t<std::is_convertible_v<U,T>&&dcols> >
matrix<T,crows,ccols>& operator+=(const matrix<U,r,c>& a){
    if(cols!=a.nrows()){
        throw std::runtime_error("Dimension mismatch! Cannot matmul these!");
    }

    matrix<T,rows,Dynamic> res(a.ncols());

#ifdef _openmp_included
    uint64_t optimal_threads = std::min(
        (uint64_t)omp_get_num_procs(), 
        (rows*cols) / 10000
    );
    omp_set_num_threads(std::max(uint64_t(1), optimal_threads));
    #pragma omp parallel for collapse(2)
#endif
    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            res(i,j)=static_cast<T>(0);
            for(size_t k=0;k<cols;++k){
                res(i,j)=res(i,j)+(*m_storage)[i*cols+k]*a(k,j);
            }
        }
    }
    this->m_storage = std::move(res.m_storage);
    return *this;
}

/**
 * Inplace operations (matrix-scalar)
 */
    

/**
 * Transpose operation
 */


/**
 * Shape and Resize operations
 */
    
    template<typename U=void, bool drows = dynamic_rows, bool dcols = dynamic_cols, 
    std::enable_if_t<drows&&dcols,U>* =nullptr>
    void reshape(uint64_t r, uint64_t c){
        if(r*c != rows*cols){
            throw std::runtime_error("Size mismatch, cannot reshape!");
        }
        this->rows=r, this->cols=c;
        return;
    }

    // resize
    // when both dim is dynamic
    template<typename U=void, bool drows=dynamic_rows, bool dcols=dynamic_cols,
    std::enable_if_t<drows&&dcols, U>* = nullptr>
    void resize(uint64_t r, uint64_t c){
        this->m_storage->resize(r*c);
        this->rows=r, this->cols=c;
    }
    // when one dim is dynamic
    template<typename U=void, bool drows=dynamic_rows, bool dcols=dynamic_cols,
    std::enable_if_t<drows&&!dcols, U>* = nullptr>
    void resize(uint64_t r){
        this->m_storage->resize(r*cols);
        this->rows=r;
    }
    // when one dim is dynamic
    template<typename U=void, bool dcols=dynamic_cols, bool drows=dynamic_rows,
    std::enable_if_t<!drows&&dcols, U>* = nullptr>
    void resize(uint64_t c){
        this->m_storage->resize(rows*c);
        this->cols=c;
    }

/**
 * Friends
 */

    template<typename U,int64_t r,int64_t c>
    friend class matrix;

};

/**
 * Arithmetic operations, scalar-matrix
 */
template<typename T,typename U, int64_t r,int64_t c, 
typename R = decltype(std::declval<T>()+std::declval<U>()) >
matrix<R,Dynamic,Dynamic> operator+(const matrix<T,r,c>& a,const U& k){
    return a+k;
}
template<typename T,typename U, int64_t r,int64_t c, 
typename R = decltype(std::declval<T>()-std::declval<U>()) >
matrix<R,Dynamic,Dynamic> operator-(const matrix<T,r,c>& a,const U& k){
    return -a+k;
}
template<typename T,typename U, int64_t r,int64_t c, 
typename R = decltype(std::declval<T>()*std::declval<U>()) >
matrix<R,Dynamic,Dynamic> operator*(const matrix<T,r,c>& a,const U& k){
    return a*k;
}

/**
 * Misc Operators
 */

template<typename U,int64_t r, int64_t c>
std::ostream& operator<<(std::ostream& out, const matrix<U,r,c>& mat){
    
    int64_t m=mat.nrows(), n = mat.ncols();
    for(int64_t i=0;i<m;i++){
        for(int64_t j=0;j<n;j++){
            out<<std::left<<std::setw(5)<<mat(i,j);
            out<<" ";
        }
        out<<"\n";
    }
    return out;
}

template<typename U,int64_t r,int64_t c>
void print_info(const matrix<U,r,c>& mat){
    for(int i=0;i<30;i++)std::cout<<"-";
    std::cout<<std::endl;

    std::cout<<"Object Address: "<<&mat<<std::endl;
    std::cout<<"Data   Address: "<<mat.data()<<std::endl;
    std::cout<<"Rows: "<<mat.nrows()<<std::endl;
    std::cout<<"Cols: "<<mat.ncols()<<std::endl;
    std::cout<<"Size: "<<mat.size()<<std::endl;

    std::cout<<std::endl<<mat;
    for(int i=0;i<30;i++)std::cout<<"-";
    std::cout<<std::endl<<std::endl;
}


/**
 * usings
 */

// dynamic matrix, usage: matXf m1(5,6);
using matXf = matrix<float, Dynamic, Dynamic>;
using matXd = matrix<double, Dynamic, Dynamic>;
using matXi = matrix<int, Dynamic, Dynamic>;
using matXl = matrix<long, Dynamic, Dynamic>;
using matXll= matrix<int64_t, Dynamic, Dynamic>;
using matXcf= matrix<std::complex<float>, Dynamic,Dynamic>;
using matXcd = matrix<std::complex<double>, Dynamic, Dynamic>;
using matXci = matrix<std::complex<int>, Dynamic, Dynamic>;
using matXcl = matrix<std::complex<long>, Dynamic, Dynamic>;
using matXcll= matrix<std::complex<int64_t>, Dynamic, Dynamic>;

// static matrices, usage: matvd<4,6> m1;
template<uint64_t i, uint64_t j>
using matVf = matrix<float,i,j>;
template<uint64_t i, uint64_t j>
using matVd = matrix<double,i,j>;
template<uint64_t i, uint64_t j>
using matVi = matrix<int,i,j>;
template<uint64_t i, uint64_t j>
using matVl = matrix<long,i,j>;
template<uint64_t i, uint64_t j>
using matVll = matrix<int64_t,i,j>;
template<uint64_t i, uint64_t j>
using matVcf = matrix<std::complex<float>,i,j>;
template<uint64_t i, uint64_t j>
using matVcd = matrix<std::complex<double>,i,j>;
template<uint64_t i, uint64_t j>
using matVci = matrix<std::complex<int>,i,j>;
template<uint64_t i, uint64_t j>
using matVcl = matrix<std::complex<long>,i,j>;
template<uint64_t i, uint64_t j>
using matVcll = matrix<std::complex<int64_t>,i,j>;

// dynamic column vector
using vecXd = matrix<double, Dynamic, 1>;
using vecXf = matrix<float, Dynamic, 1>;
using vecXi = matrix<int, Dynamic, 1>;
using vecXl = matrix<long, Dynamic, 1>;
using vecXll= matrix<int64_t, Dynamic, 1>;
using vecXcf= matrix<std::complex<float>, Dynamic,1>;
using vecXcd = matrix<std::complex<double>, Dynamic, 1>;
using vecXci = matrix<std::complex<int>, Dynamic, 1>;
using vecXcl = matrix<std::complex<long>, Dynamic, 1>;
using vecXcll= matrix<std::complex<int64_t>, Dynamic, 1>;

// fixed size column vector
template<uint64_t i>
using vecVf = matrix<float,i,1>;
template<uint64_t i>
using vecVd = matrix<double,i,1>;
template<uint64_t i>
using vecVi = matrix<int,i,1>;
template<uint64_t i>
using vecVl = matrix<long,i,1>;
template<uint64_t i>
using vecVll = matrix<uint64_t,i,1>;
template<uint64_t i>
using vecVcf = matrix<std::complex<float>,i,1>;
template<uint64_t i>
using vecVcd = matrix<std::complex<double>,i,1>;
template<uint64_t i>
using vecVci = matrix<std::complex<int>,i,1>;
template<uint64_t i>
using vecVcl = matrix<std::complex<long>,i,1>;
template<uint64_t i>
using vecVcll = matrix<std::complex<int64_t>,i,1>;

// dynamic size row vectors
using rvecXd = matrix<double,1,Dynamic>;
using rvecXf = matrix<float, 1,Dynamic>;
using rvecXi = matrix<int, 1,Dynamic>;
using rvecXl = matrix<long, 1,Dynamic>;
using rvecXll= matrix<int64_t, 1,Dynamic>;
using rvecXcf= matrix<std::complex<float>, 1,Dynamic>;
using rvecXcd = matrix<std::complex<double>, 1,Dynamic>;
using rvecXci = matrix<std::complex<int>, 1,Dynamic>;
using rvecXcl = matrix<std::complex<long>, 1,Dynamic>;
using rvecXcll= matrix<std::complex<int64_t>, 1,Dynamic>;

// fixed size row vectors
template<uint64_t i>
using rvecVf = matrix<float,1,i>;
template<uint64_t i>
using rvecVd = matrix<double,1,i>;
template<uint64_t i>
using rvecVi = matrix<int,1,i>;
template<uint64_t i>
using rvecVl = matrix<long,1,i>;
template<uint64_t i>
using rvecVll = matrix<uint64_t,1,i>;
template<uint64_t i>
using rvecVcf = matrix<std::complex<float>,1,i>;
template<uint64_t i>
using rvecVcd = matrix<std::complex<double>,1,i>;
template<uint64_t i>
using rvecVci = matrix<std::complex<int>,1,i>;
template<uint64_t i>
using rvecVcl = matrix<std::complex<long>,1,i>;
template<uint64_t i>
using rvecVcll = matrix<std::complex<int64_t>,1,i>;

}
}