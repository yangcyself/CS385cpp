/**
 * g++ -I/home/yangcy/programs/eigen -I. test/testconv.cpp cnn/tensor.cpp -o  test/testconv -std=c++11 -Wall
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
  Eigen::MatrixXd aa(3,18);
  aa <<  1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9,
         1,2,4,5,3,6,9,1,8, 4,6,4,5,3,6,9,10,3,
         5,6,7,8,2,5,3,7,0, 2,6,3,7,3,1,7,3,4;
//   a <<aa; // this means the stroage order is transparent to the << operator 

  Tensor a(aa,3,3); // a is a 3 * 2*3*3 tensor
  Eigen::MatrixXd bb(1,18); //b is a 1 * 3*3*2 tensor
  bb << 1,0,   1,0,   1,0,
        -2,0,  -2,0,  -2,0,    
        1,0,   1,0,   1,0   ;
  Tensor b(bb,3,3);

//   cout << a.expand(1,2,4)<<endl;
  Tensor c = a.conv(b,-1,0,1);
  c.print();

cout<<"#####1######"<<endl;

  bb << 0,1,     0,2,   0,3,
        0,4,     0,5,   0,6, 
        0,7,     0,8,   0,9  ;
  b = Tensor(bb,3,3);

//   cout << a.expand(1,2,4)<<endl;
  c = a.conv(b,-1,0,1);
  c.print();

cout<<"#####2######"<<endl;
  bb << 1,1,   1,2,   1,3,
        -2,4,  -2,5,  -2,6,    
        1,7,   1,8,   1,9 ;
  b = Tensor(bb,3,3);
  c = a.conv(b,-1,0,1);
  c.print();

cout<<"#####3######"<<endl;
  bb << 1,1,   1,2,   1,3,
        -2,4,  -2,5,  -2,6,    
        1,7,   1,8,   1,9 ;
  b = Tensor(bb,3,3);
  c = a.conv(b,-1,0,2);
  c.print();

cout<<"#####4######"<<endl;
  c = a.eleadd(a);
  c.print();

cout<<"#####5######"<<endl;
  c = a.elemul(a);
  c.print();


  bb =Eigen::MatrixXd(2,18);
  bb <<   1,1,   1,2,   1,3,
        -2,4,  -2,5,  -2,6,    
        1,7,   1,8,   1,9 ,

        1,0,   1,0,   1,0,
        -2,0,  -2,0,  -2,0,    
        1,0,   1,0,   1,0;
  b = Tensor(bb,3,3);
cout<<"#####6######"<<endl; // kernel flip
  c = b.kernelFlip();
  c.print();

cout<<"#####7######"<<endl; // kernelize
  c = a.kernelize();
  c.print();

cout<<"#####8######"<<endl; // transposetoNHWC
  c = a.transposetoNHWC();
  c.print();

cout<<"#####9######"<<endl; // transposefromNHWC
  c = b.transposefromNHWC();
  c.print();

cout<<"#####10######"<<endl; // transposefromNHWC and to ...
  c = a.transposetoNHWC().transposefromNHWC();
  c.print();

cout<<"#####11######"<<endl; // kerneltranspose
  c = b.kernelTranspose();
  c.print();

return 0;
}