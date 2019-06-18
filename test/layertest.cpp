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

int main(int argc, char* argv[] )
{
    Variable feature;
    // MaxpoolLayer layer(feature,2);
    ReluLayer layer(feature);
    Tensor::matrix aa(2,32);
    aa<< -1,2,3,-4, 2,-3,4,1, 4,-3,2,1, 2,-1,3,4,  -2,4,1,-2, 4,-4,-4,4, 2,3,-1,2, 2,-1,3,4,
         4,-2,3,1, -3,2,1,4, -2,1,4,-3, -2,4,1,-3,  3,-2,-1,4, 2,2,3,1, 1,-1,2,2, 4,-3,2,-1;
    Tensor a(aa,4,4);
    feature.initData(a);
    layer.forward().print();
    // Tensor::matrix gg = Tensor::matrix::Ones(2,8);
    // layer.backward(Tensor(gg,2,2));
    Tensor::matrix gg = Tensor::matrix::Ones(2,32);
    layer.backward(Tensor(gg,4,4));
    feature.forward().print();
    return 0;
}