/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#ifndef FULL_TREE_DATA_H
#define FULL_TREE_DATA_H



#include <vector>
#include <iostream>
#include <utility>
#include <array>
#include "access_data.h"
//using namespace std;




//********* Define object to be used during classification  of a particular instance
template<class Tfloat, class Tint> //use a template type to allow user to specify the type of data stored
class FullTreeData
{

private:
    //create a single object that can be coerced to python to hold all data to allow state to be exported
    //create references to particular values in state_data object
    std::pair<  std::array<long,11>,std::pair< std::vector<Tfloat>,std::vector<Tint> >  >  state_data;
        //use two vectors one of double and one of long to store the different types of data

    //user specified properties
    long & n_nodes_{state_data.first[0]};
    long & n_classes{state_data.first[1]};
    long & level {state_data.first[2]};
    long & full_tree_index{state_data.first[3]};

    //derived indicies
    long & label_index{state_data.first[4]};
    long & counts_index{state_data.first[5]};
    long & parent_index{state_data.first[6]};

    long & p0_index{state_data.first[7]};
    long & noise_weight_index{state_data.first[8]};

    long & n_floats_per_node{state_data.first[9]};
    long & n_ints_per_node{state_data.first[10]};



    std::vector<Tfloat> & float_data{state_data.second.first} ;
    std::vector<Tint> & int_data{state_data.second.second} ;

public:
    const long & n_nodes=n_nodes_;  //make readonly version publicly accessible

    //Create data access functions
    AccessData<Tint> label {int_data,label_index};//level values per node
    AccessData<Tint> counts {int_data,counts_index}; //n_classes values per node
    AccessData<Tint> parent_indicies {int_data,parent_index}; //level values per node

    AccessData<Tfloat> p0 {float_data,p0_index}; //n_classes values per node
    AccessData<Tfloat> noise_weight {float_data,noise_weight_index}; //single value per node


    //constructor
    FullTreeData(long n_nodes=0, long n_classes=0, long level=0)
    {
        state_data.first.fill(0);
        initialize(n_nodes, n_classes, level);
    }

    void initialize(long n_nodes, long n_classes, long level)
    {
        //store values
        this->n_nodes_=n_nodes;
        this->n_classes=n_classes;
        this->level=level;


        if (n_nodes >0 && n_classes>0)
        {
            int_data.clear();
            float_data.clear();

            //set array sizes based on stored data
            n_ints_per_node   = level + //label
                                n_classes+ //counts
                                level; //parent_label

            n_floats_per_node = n_classes +  //p0
                                   1;      //noise_weight

            float_data.resize((n_nodes*n_floats_per_node), -1 ); //initialize to -1
            int_data.resize((n_nodes*n_ints_per_node), -1 ); //initialize to -1
            set_node(0);
        };
    }


//            label_array = np.empty( (n_features_combos,level), order = 'C', dtype=np.int32)
//            counts_array = np.empty( (n_features_combos,self.n_classes), order = 'C', dtype=np.int32)
//            parent_array = np.full( (n_features_combos,level),-1, order = 'C', dtype=np.int32) # null values are assigned to -1

//            p0_array = np.empty_like(counts_array, dtype=NP_FLOAT)
//            noise_weight_array =np.empty(n_features_combos, dtype=NP_FLOAT)


    void set_node(long full_tree_index)
    {
        //:param full_tree_index: index of where in full tree we are
        //:return: nothing

        this->full_tree_index=full_tree_index;

        if (full_tree_index>=n_nodes_ || n_classes<=0)
        {
            std::cerr<<"full_tree_index="<<full_tree_index<<" , n_nodes="<<n_nodes_<<" , n_classes="<<n_classes<<"\n";
            std::cerr << "Tree not initialized or full_tree_index too large\n";
            throw "Tree not initialized or full_tree_index too large";
        };

        //create indicies for each value
        label_index = n_ints_per_node * full_tree_index;
        counts_index = label_index+level;
        parent_index = counts_index+n_classes;

        p0_index = n_floats_per_node *full_tree_index;
        noise_weight_index = p0_index+n_classes;
    };

    //outputs all values needed to save the state of an object, convert from array to vector for python coercion
    std::pair< std::vector<long>,std::pair< std::vector<Tfloat>,std::vector<Tint> >  > getstate()
    {
        return std::pair< std::vector<long>,std::pair< std::vector<Tfloat>,std::vector<Tint> >  > {{state_data.first.begin(),state_data.first.end()},state_data.second};
    };

    //takes in values defining the state and sets the current state (values come from getstate() )
    void setstate(const std::pair< std::vector<long>,std::pair< std::vector<Tfloat>,std::vector<Tint> >  > & in_data)
    {
        //use for loop to assign values
        for (unsigned long ind=0; ind<in_data.first.size(); ++ind)
        {
            state_data.first[ind]=in_data.first[ind];
        }
        state_data.second = in_data.second; //copy assignment
    };


    //override copy constructor to account for internal references
    FullTreeData(const FullTreeData &object_to_copy)
    {
        state_data=object_to_copy.state_data;
    }


    // override assignment operator to handle copying of access data objects
    FullTreeData& operator= (const FullTreeData & object_to_copy)
    {

        state_data=object_to_copy.state_data;
        return *this;
    }

};

#endif