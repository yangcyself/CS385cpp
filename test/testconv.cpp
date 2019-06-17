/**
 * g++ -I/home/yangcy/programs/eigen -I. test/testconv.cpp cnn/tensor.cpp -o  test/testconv -std=c++11
 */
#include <iostream>
#include "cnn/tensor.h"
#include <Eigen/Dense>
#include <vector>

using namespace convnn;
using namespace std;

int main()
{

//   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> a(3,4);
  Eigen::MatrixXd aa(3,8);
  aa <<  1,2,3,4,1,2,3,4,
        1,2,4,5,3,6,9,10,
        5,6,7,8,2,5,3,7;
//   a <<aa; // this means the stroage order is transparent to the << operator 

  Tensor a(aa,2,2); // a is a 3 * 2*2*2 tensor
  Eigen::MatrixXd bb(1,18); //b is a 1 * 3*3*2 tensor
  bb << 1,0,   1,0,   1,0,
        -2,0,  -2,0,  -2,0,    
        1,0,   1,0,   1,0   ;
  Tensor b(bb,3,3);

//   cout << a.expand(1,2,4)<<endl;
  Tensor c = a.conv(b);
  c.print();


//   Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> > M2(b.data(), 2,2);
//   cout << "M2:" << endl << M2 << endl;
}