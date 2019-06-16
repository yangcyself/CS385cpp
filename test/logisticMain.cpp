/**
 * g++ $(pkg-config --cflags eigen3) -I. test/logisticMain.cpp logistic/logistic.cpp -o test/logisticMain -std=c++11
 * g++ -I/home/yangcy/programs/eigen -I. test/logisticMain.cpp logistic/logistic.cpp -o test/logisticMain -std=c++11
 * g++ -I/home/yangcy/programs/eigen -I. test/logisticMain.cpp  -o test/logisticMain -std=c++11
 */

#include <logistic/logistic.h>
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace logistic;
// using namespace Eigen;
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
    // MatrixXd a(2,2);
    // MatrixXi b(2,2);
    // a << 1,2,
    //     3,4;
    // cout << "(a > 0).all()   = " << (a.array() > 2).all() << endl;
    // b = (a.array() >0).cast <int> () ;
    // cout<< b<<endl;
    // // cout << "(a > 0).any()   = " << (a > 0).any() << endl;
    // // cout << "(a > 0).count() = " << (a > 0).count() << endl;
    // // cout << endl;
    // // cout << "(a > 2).all()   = " << (a > 2).all() << endl;
    // // cout << "(a > 2).any()   = " << (a > 2).any() << endl;
    // // cout << "(a > 2).count() = " << (a > 2).count() << endl;
}
