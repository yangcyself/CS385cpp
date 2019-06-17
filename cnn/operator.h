#ifndef OPERATOR_H_
#define OPERATOR_H_

#include <cnn/tensor.h>
#include <Eigen/Dense>
#include <Eigen/Core>


namespace convnn{

/**
 * The Operator class, the parent class of all the network structures
 * Contains the pointer of upstream operators and thus forms a computation graph
 * Every time in forward propagation, call the forward propagation of all upstream operators
 *      and return the result
 * Every time in backward propagation, calculate the gradient an recursively pass to upstream
 * (deparcated) Every time call update, calculate the gradient an recursively and update a step
 * 
 * TODO: add a dirty flag and cache operator's output to optimize
*/

class Operator{
public:
    using vector = Eigen::VectorXd; 
    using matrix = Eigen::Matrix<double,  Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>;

private:
    
public:
    Operator(){}
    virtual ~Operator(){}
    virtual Tensor forward() =0;
    virtual void backward(const Tensor& in) =0; 
    // virtual void update(double step) =0;
};
} /* namespace logistic */

#endif //OPERATOR_H_