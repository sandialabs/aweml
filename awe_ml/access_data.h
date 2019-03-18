

#ifndef ACCESS_DATA
#define ACCESS_DATA
#include <iostream>

//#define DEBUG

#include <vector>

//********** Define a class that allows for specific values in a data structure to be accessed using the [] operator
template<class T>
class AccessData
{
private:
    std::vector<T> & data;
    long & starting_index;

public:
    AccessData (std::vector<T> & data,long & starting_index ): data{data}, starting_index{starting_index}
    {
    }
    inline T & operator[](long index)
    {
//        #ifdef DEBUG
//           if ( (starting_index+index) >=(signed) data.size() )
//           {
//                std::cerr << "index "<<starting_index+index<<" larger than vector size of "<<data.size()<<"\n";
//                std::cout<<"access data error"<<starting_index<<" + "<<index<<" should be less than "<<data.size()<<"\n";
//                throw "Index out of Range";
//           };
////           std::cout<<"access data "<<starting_index<<" + "<<index<<" should be less than "<<data.size()<<"\n";
//        #endif
        return data[starting_index+index];
    }

    inline T * ptr(long index)
    {
//        #ifdef DEBUG
//           if ( (starting_index+index) >=(signed) data.size() )
//           {
//                std::cerr << "index "<<starting_index+index<<" larger than vector size of "<<data.size()<<"\n";
//                std::cout<<"access data error"<<starting_index<<" + "<<index<<" should be less than "<<data.size()<<"\n";
//                throw "Index out of Range";
//           };
//        #endif
        return & data[starting_index+index];
    };

    inline T * ptr()
    {
//        #ifdef DEBUG
//           if ( (starting_index+index) >=(signed) data.size() )
//           {
//                std::cerr << "index "<<starting_index+index<<" larger than vector size of "<<data.size()<<"\n";
//                std::cout<<"access data error"<<starting_index<<" + "<<index<<" should be less than "<<data.size()<<"\n";
//                throw "Index out of Range";
//           };
//        #endif
        return & data[starting_index];
    };

};


#endif
