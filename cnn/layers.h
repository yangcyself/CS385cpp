#ifndef LAYERS_H_
#define LAYERS_H_


#include <cnn/basicOps.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <memory> //std::shared_ptr

namespace convnn{

/**
 * layers are also operators, but with more complex functions
 * implemented layers are:
 *  conv layer
 *  fc layer
 *  relu layer
 *  max pooling layer
 *  
*/

/**
 * convlayer contains a variable as kernel, 
 *  it is just a wraper of Conv, together with its own variable
 *  the forward and backward operations all works on kernel
 */
class ConvLayer:public Operator{

private:
    std::shared_ptr<Operator> a;
    Variable kernel;
    Conv convolution;
public:

    /**
     * arguments 
     *  aa: input operator
     *  inC: input channel number
     *  outC: output channel number
     *  K: the kernel size K*K
     *  pad: padding
     *  stride: stride
     *  initRange: the variance to initialize the variable
     */
    ConvLayer(Operator& aa, int inC, int outC, int K, int pad, int stride, double initRange)
        :a(&aa,&null_deleter),kernel(),convolution(aa,kernel,pad,stride)
        {kernel.initData(Tensor(matrix::Random(outC,inC*K*K )*initRange, K,K ));}
    ~ConvLayer(){}
    Tensor forward(){return convolution.forward();}
    void backward(const Tensor& in){convolution.backward(in);}
    void train(){convolution.train();}
    void test(){convolution.test();}
};


/**
 * FcLayer is a special case of conv layer where the kernel size equals to the input size
 *  in most case, their size are all one.
 */
class FcLayer:public Operator{

private:
    std::shared_ptr<Operator> a;
    Variable kernel;
    Conv convolution;
public:

    FcLayer(Operator& aa, int inC, int outC, double initRange)
        :a(&aa,&null_deleter),kernel(),convolution(aa,kernel,0,1)
        {kernel.initData(Tensor(matrix::Random(outC,inC*1*1 )*initRange, 1,1 ));}
    ~FcLayer(){}
    Tensor forward(){return convolution.forward();}
    void backward(const Tensor& in){convolution.backward(in);}
    void train(){convolution.train();}
    void test(){convolution.test();}
};


/**
 * relulayer ...
 */
class ReluLayer:public Operator{

private:
    std::shared_ptr<Operator> a;
    Tensor ca;
    Tensor reluMask(const Tensor& in)const;
public:

    ReluLayer(Operator& aa):a(&aa,&null_deleter){}
    ~ReluLayer(){}
    Tensor forward();
    void backward(const Tensor& in);
    void train(){a->train();}
    void test(){a->test();}
};

/**
 * maxpooling layer ...
 */
class MaxpoolLayer:public Operator{

private:
    std::shared_ptr<Operator> a;
    Eigen::VectorXi indevx; // the indexs for pooling
    Eigen::VectorXi indevy; // the indexs for pooling
    int K;
    int inputh, inputw;
    int inH, inW;
    void indexVector(Tensor& in);

public:

    MaxpoolLayer(Operator& aa, int k):a(&aa,&null_deleter),K(k){}
    ~MaxpoolLayer(){}
    Tensor forward();
    void backward(const Tensor& in);
    void train(){a->train();}
    void test(){a->test();}
};



} /* namespace convnn */

#endif //LAYERS_H_