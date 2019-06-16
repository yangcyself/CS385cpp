/**
 * 
 * usage
 * g++ -I/home/yangcy/programs/eigen -I. fisherMain.cpp fisher/fisher.cpp -o fisherMain -std=c++11
 * ./fisherMain
 */

#include <fisher/fisher.h>
#include <iostream>
#include <Eigen/Dense>
#include <dataset/matrix_dataset.h>

using namespace std;
using namespace fisher;
using namespace dataset;

int main(int argc, char const *argv[])
{
    MatrixDataset TrainDataset("./out/hog.ptbf",false);
    MatrixDataset TestDataset("./out/hog.ptbf",true);

    MatrixDataset::matrix trainX = TrainDataset.X();
    MatrixDataset::vector trainY = TrainDataset.Y();
    MatrixDataset::matrix testX = TestDataset.X();
    MatrixDataset::vector testY = TestDataset.Y();

    fisheror a(2);
    a.train(trainY,trainX);
    // cout<<"X:/n"<<X<<"Y:/n"<<Y<<endl;
    MatrixDataset::vector trainY_ = a.forward(trainX);
    
    MatrixDataset::vector testY_ = a.forward(testX);
    
    cout<<"forward:\n"<<a.forward(X)<<endl;

    return 0;
}
