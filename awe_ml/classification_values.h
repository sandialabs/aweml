/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#ifndef CLASSIFICATION_VALUES_H
#define CLASSIFICATION_VALUES_H



#include <vector>
#include <iostream>
#include <utility>
#include <array>
#include "access_data.h"
//using namespace std;




//********* Define object to be used during classification  of a particular instance
template<class T> //use a template type to allow user to specify the type of data stored
class ClassificationValues
{

private:
    //create a single object that can be coerced to python to hold all data to allow state to be exported
    //create references to particular values in state_data object
    std::pair<std::array<long,8>,std::vector<T>> state_data;

    long & n_nodes{state_data.first[0]};
    long & n_classes{state_data.first[1]};

    long & local_tree_index{state_data.first[2]};
    long & row_index{state_data.first[3]};
    long & estimated_p_start{state_data.first[4]};
    long & p_sum_start{state_data.first[5]};
    long & weight_sum_start{state_data.first[6]};
    long & max_child_noise_ind{state_data.first[7]};


public:

    std::vector<T> & data{state_data.second} ; //all data in classification values


    //Create access functions
    AccessData<T> estimated_p {data,estimated_p_start};
    AccessData<T> p_sum {data,p_sum_start};
    AccessData<T> weight_sum {data,weight_sum_start};
    AccessData<T> max_child_noise {data,max_child_noise_ind};
    
    //constructor
    ClassificationValues(long n_nodes=0, long n_classes=0)
    {
        state_data.first.fill(0);
        initialize(n_nodes, n_classes);
    }
   
    void initialize(long n_nodes, long n_classes)
    {
        //store values
        this->n_nodes=n_nodes;
        this->n_classes=n_classes;

        if (n_nodes >0 && n_classes>0)
        {
            data.clear();
            data.resize((n_nodes*(3*n_classes+1)), 0 );
            set_node(0);
        };
    }

    
    void set_node(long local_tree_index)
    {
        //:param local_tree_index: index of where in local tree we are
        //:return: nothing

        this->local_tree_index=local_tree_index;

        if (local_tree_index>=n_nodes || n_classes<=0)
        {
            std::cerr<<"local_tree_index="<<local_tree_index<<" , n_nodes="<<n_nodes<<" , n_classes="<<n_classes<<"\n";
            std::cerr << "Tree not initialized or local_tree_index too large\n";
            throw "Tree not initialized or local_tree_index too large";
        };
        
        //create indicies for each value
        row_index = (3*n_classes+1) * local_tree_index;
        estimated_p_start = row_index+0;                //stores the estimated probability
        p_sum_start = row_index+n_classes;              //stores the sum of p*weight for children
        weight_sum_start = row_index+2*n_classes;       //stores the total weight of the children
        max_child_noise_ind = row_index + 3*n_classes;  // stores the max noise value of children
    };

    //outputs all values needed to save the state of an object, convert from array to vector for python coercion
    std::pair<std::vector<long>,std::vector<T>> getstate()
    {
        return std::pair<std::vector<long>,std::vector<T>> {{state_data.first.begin(),state_data.first.end()},state_data.second};
    };

    //takes in values defining the state and sets the current state (values come from getstate() )
    void setstate(const std::pair<std::vector<long>,std::vector<T> > & in_data)
    {
        //use for loop to assign values
        for (unsigned long ind=0; ind<in_data.first.size(); ++ind)
        {
            state_data.first[ind]=in_data.first[ind];
        }
        state_data.second = in_data.second; //copy assignment
    };


    //override copy constructor to account for internal references
    ClassificationValues(const ClassificationValues &object_to_copy)
    {
        state_data=object_to_copy.state_data;
    }


    // override assignment operator to handle copying of access data objects
    ClassificationValues& operator= (const ClassificationValues & object_to_copy)
    {

        state_data=object_to_copy.state_data;
        return *this;
    }

};

#endif