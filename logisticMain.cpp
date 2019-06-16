/**
 * usage
 * 
 * g++ -I/home/yangcy/programs/eigen -I. logisticMain.cpp logistic/logistic.cpp dataset/matrix_dataset.cpp -o logisticMain $(pkg-config --cflags --libs protobuf) -std=c++11
 * ./logisticMain
 */

#include <logistic/logistic.h>
#include <iostream>
#include <Eigen/Dense>
#include <dataset/matrix_dataset.h>
using namespace std;
using namespace logistic;
using namespace dataset;

// double accuracy(MatrixDataset::vector y, MatrixDataset::vector y_)
// {
    
//     // MatrixDataset::vector y_pos = (y_.array() >= 0.5);

//     Eigen::VectorXi y_neg = (y_.array() < 0.5).cast <int> ();

//     // cout<<"y_pos.count() "<<y_pos.count()<<endl;
//     // cout<<"y_neg.count() "<<y_neg.count()<<endl;
//     // int true_positive = (y * y_pos).sum();
//     // int true_negtive = (y * y_neg).sum();
//     // int false_positive = ((1.0-y) * y_pos).sum();
//     // int false_negtive = ((1.0-y) * y_neg).sum();
//     // cout<<"true_positive"<<true_positive<<endl;
//     // cout<<"true_negtive"<<true_negtive<<endl;
//     // cout<<"false_positive"<<false_positive<<endl;
//     // cout<<"false_negtive"<<false_negtive<<endl;

//     // return (double) (true_positive + false_negtive) / y.rows();
//     return 0.5;
// }

int main(int argc, char const *argv[])
{
    MatrixDataset TrainDataset("./out/hog.ptbf",false);
    MatrixDataset TestDataset("./out/hog.ptbf",true);

    MatrixDataset::matrix trainX = TrainDataset.X();
    MatrixDataset::vector trainY = TrainDataset.Y();
    MatrixDataset::matrix testX = TestDataset.X();
    MatrixDataset::vector testY = TestDataset.Y();


    // Logistictor a(3);
    // a.train(trainY,trainX,100,0.1,0.001);
    // // cout<<"X:/n"<<X<<"Y:/n"<<Y<<endl;
    // MatrixDataset::vector trainY_ = a.forward(trainX);
    // cout<<"train accuracy: "<<accuracy(trainY, trainY_);

    // MatrixDataset::vector testY_ = a.forward(testX);
    
    // cout<<"forward:\n"<<a.forward(X)<<endl;

    // a.train(Y,X,100,0.1,0.001);
    // cout<<a.forward(X)<<endl;

    return 0;
}
