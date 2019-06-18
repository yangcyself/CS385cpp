/**
 * g++ -I/home/yangcy/programs/eigen -I. test/layertest.cpp cnn/tensor.cpp cnn/basicOps.cpp cnn/layers.cpp -o  test/layertest -std=c++11 -Wall
 */
#include <iostream>
#include "cnn/layers.h"
// #include "cnn/basicOps.h"
#include <Eigen/Dense>
#include <vector>

using namespace convnn;
using namespace std;
//初始话是真的重要,要不然在sigmoid一个很平的位置
int main(int argc, char* argv[] )
{
    // Variable feature;
    // // MaxpoolLayer layer(feature,2);
    // ReluLayer layer(feature);
    // Tensor::matrix aa(2,32);
    // aa<< -1,2,3,-4, 2,-3,4,1, 4,-3,2,1, 2,-1,3,4,  -2,4,1,-2, 4,-4,-4,4, 2,3,-1,2, 2,-1,3,4,
    //      4,-2,3,1, -3,2,1,4, -2,1,4,-3, -2,4,1,-3,  3,-2,-1,4, 2,2,3,1, 1,-1,2,2, 4,-3,2,-1;
    // Tensor a(aa,4,4);
    // feature.initData(a);
    // layer.forward().print();
    // // Tensor::matrix gg = Tensor::matrix::Ones(2,8);
    // // layer.backward(Tensor(gg,2,2));
    // Tensor::matrix gg = Tensor::matrix::Ones(2,32);
    // layer.backward(Tensor(gg,4,4));
    // feature.forward().print();
    Placeholder input; //2 * 3*4*4
    Tensor::matrix aa(2,48);
    aa << Tensor::matrix::Constant(1,48,-1),
            Tensor::matrix::Constant(1,48,-2);
    input.feedData(Tensor(aa,4,4));
    ConvLayer c1(input,3,8,3,1,1,0.1); //2 * 8*4*4
    // c1.ker().forward().print();
    c1.forward();
    // c1.backward(Tensor(Tensor::matrix::Constant(2,8*4*4,1),4,4));
    // c1.ker().forward().print();
    // cout<<"c1 forwarded"<<endl;
    ReluLayer r1(c1);
    // r1.forward();
    // cout<<"r1 forwarded"<<endl;
    MaxpoolLayer p1(r1,2);//2 * 8*2*2
    // p1.forward().print();
    // cout<<"p1 forwarded"<<endl;
    ConvLayer c2(p1,8,8,3,1,1,0.1);//2 * 8*2*2
    // c2.forward();
    // cout<<"c2 forwarded"<<endl;
    ReluLayer r2(c2);
    // r2.forward();
    // cout<<"r2 forwarded"<<endl;
    MaxpoolLayer p2(r2,2);//2 * 8*1*1
    FcLayer f1(p2,8,1,0.1);
    SigmoidLayer o(f1);

    // o.forward().print();
    
    o.train();
    // c1.ker().forward().print();
    Tensor::matrix gg(2,1);
    gg << 1,-10;
    o.forward().print();
    f1.forward().print();
    o.backward(gg);
    // c1.ker().forward().print();
    f1.forward().print();
    o.forward().print();

    return 0;
}