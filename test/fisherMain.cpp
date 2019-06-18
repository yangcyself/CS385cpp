/**
 * 
 * usage
 * g++ -I/home/yangcy/programs/eigen -I. test/fisherMain.cpp fisher/fisher.cpp -o test/fisherMain -std=c++11
 * ./fisherMain
 */

#include <fisher/fisher.h>
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace fisher;

int main(int argc, char const *argv[])
{
    // Eigen::VectorXd a(1);
    // a << 0.5;
    // double b = a(0);
    // cout<<b<<endl;
    fisheror a(2);
    Eigen::MatrixXd X(3,2);
    Eigen::Vector3d Y;
    X << 0,3,
         -1,-1,
         2,0;
    Y << 1,0,1;
    a.train(Y,X);
    // cout<<"X:/n"<<X<<"Y:/n"<<Y<<endl;
    cout<<"forward:\n"<<a.forward(X)<<endl;

    return 0;
}
