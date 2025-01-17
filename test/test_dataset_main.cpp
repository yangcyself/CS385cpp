/**
 * usage:
 * g++ -I/home/yangcy/programs/eigen -I. test/test_dataset_main.cpp dataset/matrix_dataset.cpp protobuf/dataset.hog.pb.cc -o test/test_dataset_main $(pkg-config --cflags --libs protobuf) -std=c++11
 * test the functionality of dataset_matrix and dataset images
 */


#include <dataset/matrix_dataset.h>
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace dataset;

int main(int argc, char const *argv[])
{
    MatrixDataset TrainDataset("./out/hog.ptbf",false);
    MatrixDataset TestDataset("./out/hog.ptbf",true);

    MatrixDataset::matrix trainX = TrainDataset.X();
    MatrixDataset::vector trainY = TrainDataset.Y();
    MatrixDataset::matrix testX = TestDataset.X();
    MatrixDataset::vector testY = TestDataset.Y();

    cout<<"Trainx corner1: \n"<<trainX.block(0,0,20,20)<<" * "<<endl;
    cout<<"Trainx corner2: \n"<<trainX.block(100,870,20,20)<<" * "<<endl;
    
    cout<<"Trainx: "<<trainX.rows()<<" * "<<trainX.cols()<<"mean: "<< trainX.mean() <<endl;
    cout<<"Trainy: "<<trainY.rows()<<" * "<<trainY.cols()<<"mean: "<< trainY.mean() <<endl;
    cout<<"Testx: "<<testX.rows()<<" * "<<testX.cols()<<"mean: "<< testX.mean() <<endl;
    cout<<"Testy: "<<testY.rows()<<" * "<<testY.cols()<<"mean: "<< testY.mean() <<endl;
    
    return 0;
}
