#pragma once

#include<core/common.h>

namespace qb{
namespace core{

class shape{
private:
    int64_t* m_data=nullptr;
    int64_t  m_size=0;

public:
    template<typename... sizes, 
    typename = std::enable_if_t<(std::is_convertible_v<sizes, int64_t> && ...)> >
    shape(sizes... szs)
    :m_size(sizeof...(sizes)){
        this->m_data = new int64_t[this->m_size];
        int64_t i=0;
        ((this->m_data[i++]=szs),...);
    }
};

}
}