/**
 * g++ -I/home/yangcy/programs/eigen -I. test/operatorTest.cpp cnn/tensor.cpp cnn/basicOps.cpp -o  test/operatorTest -std=c++11 -Wall
 */
#include <iostream>
#include "cnn/tensor.h"
#include "cnn/basicOps.h"
#include <Eigen/Dense>
#include <vector>

using namespace convnn;
using namespace std;

int main(int argc, char* argv[] )
{
    Placeholder a;
    Variable b;
    Add c(a,b);
    Tensor::matrix aa(2,2);
    aa << 1,2,3,4;
    Tensor::matrix bb(2,2);
    bb << 5,6,7,8;

    Tensor ta(aa,1,1);
    Tensor tb(bb,1,1);


    a.feedData(ta);
    b.initData(tb);

    Tensor res = c.forward();
    res.print();
    
    Tensor::matrix dd(2,2);
    dd << 1,1,1,1;
    Tensor td(dd,1,1);
    c.backward(td);
    b.forward().print();
    return 0;
}