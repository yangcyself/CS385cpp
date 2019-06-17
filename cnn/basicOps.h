#ifndef BASICOPS_H_
#define BASICOPS_H_


#include "cnn/operator.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <memory> //std::shared_ptr

namespace convnn{

/**
 * basic operator contains two basic endpoint classes:
 *      placeholder
 *      variable
 * some simple operators 
 *      add 
 *      mul 
 *      conv
*/

void null_deleter(Operator *)
{}

/**
 * Placeholder has nothing to do with back propagation
 *  it just pass its tensor out when call by forward.
 */
class Placeholder:public Operator{

private:
    Tensor data;

public:
    Placeholder(){}
    ~Placeholder(){}
    void feedData(const Tensor& d){data = d;}
    Tensor forward(){return data;}
    void backward(const Tensor& in){}
    // void update(double step){}
};

/**
 * Variable class, contains the variable, and update according to the input gradient tensor.
 */
class Variable:public Operator{
private:
    Tensor data;
public:
    Variable(){}
    ~Variable(){}
    Tensor forward(){return data;}
    void initData(const Tensor d){data = d;}  // here should not pass by ref, because the data is going to change
    void backward(const Tensor& in){data.eleadd(in);}
};

class Add:public Operator{
private:
    std::shared_ptr<Operator> a;
    std::shared_ptr<Operator> b;
public:
    Add(Operator& aa,Operator& bb):a(&aa,&null_deleter),b(&bb,&null_deleter){}
    ~Add(){}
    Tensor forward(){return a->forward().eleadd(b->forward());}
    void backward(const Tensor& in){a->backward(in); b->backward(in);}
};

} /* namespace convnn */

#endif //BASICOPS_H_