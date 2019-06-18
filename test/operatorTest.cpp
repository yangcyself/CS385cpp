/**
 * g++ -I/home/yangcy/programs/eigen -I. test/operatorTest.cpp cnn/tensor.cpp cnn/basicOps.cpp -o  test/operatorTest -std=c++11 -Wall
 */
#include <iostream>
// #include "cnn/tensor.h"
#include "cnn/basicOps.h"
#include <Eigen/Dense>
#include <vector>

using namespace convnn;
using namespace std;

int main(int argc, char* argv[] )
{
    Variable feature;
    Variable kernel;
    Conv c (feature,kernel,1);
    c.train();
    Tensor::matrix aa(2,8);
    aa <<   1,6,3,2 , 1,4,7,2,
            1,2,3,4 , 1,2,3,4;
            
    Tensor faa(aa,2,2);

    Tensor::matrix bb(2,18); //treated as kernel
    bb <<   1,1,   1,2,   1,3,
            -2,4,  -2,5,  -2,6,    
            1,7,   1,8,   1,9 ,

            1,0,   1,0,   1,0,
            -2,0,  -2,0,  -2,0,    
            1,0,   1,0,   1,0;

    Tensor kbb(bb,3,3);

    feature.initData(faa);
    kernel.initData(kbb);

    Tensor res = c.forward();
    res.print();

    

    Tensor::matrix gg(2,8);
    gg <<   1,0,0,0, 0,0,0,0,
            0,0,0,0, 0,0,0,0;
    Tensor grad(gg,2,2);
    c.backward(grad);

    feature.forward().print();
    kernel.forward().print();

    return 0;
}