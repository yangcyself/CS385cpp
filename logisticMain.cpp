#include <logistic/logistic.h>
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace logistic;

int main(int argc, char const *argv[])
{
    Logistictor a(3);
    Eigen::Matrix3d X;
    Eigen::Vector3d Y;
    X << 1,0,1,
         -2,2,-1,
         0,1,2;
    Y << 1,0,1;
    cout<<a.forward(X)<<endl;
    a.train(Y,X,100,0.1,0.01);
    cout<<a.forward(X)<<endl;
    return 0;
}
