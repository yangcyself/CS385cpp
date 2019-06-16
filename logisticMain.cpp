/**
 * usage
 * 
 * g++ -I/home/yangcy/programs/eigen -I. logisticMain.cpp logistic/logistic.cpp dataset/matrix_dataset.cpp  protobuf/dataset.hog.pb.cc -o logisticMain $(pkg-config --cflags --libs protobuf) -std=c++11
 * ./logisticMain
 */

#include <logistic/logistic.h>
#include <iostream>
#include <Eigen/Dense>
#include <dataset/matrix_dataset.h>
using namespace std;
using namespace logistic;
using namespace dataset;

double accuracy(Eigen::VectorXd y, Eigen::VectorXd y_)
{
    
    Eigen::VectorXi y_true = y.cast <int> ();
    Eigen::VectorXi y_false = (1-y.array()).cast <int> ();
    Eigen::VectorXi y_pos = (y_.array() >= 0.5).cast <int> ();
    Eigen::VectorXi y_neg = (y_.array() < 0.5).cast <int> ();
    // cout<<"y_.head(100): \n"<<y_.head(100)<<endl;
    // cout<<"y_pos.shape: "<< y_pos.rows()<<endl;
    // cout<<"y_pos.count() "<<y_pos.sum()<<endl;
    // cout<<"y_neg.count() "<<y_neg.sum()<<endl;
    int true_positive = y_true.transpose() * y_pos;
    int true_negtive = y_true.transpose() * y_neg;
    int false_positive = y_false.transpose() * y_pos;
    int false_negtive = y_false.transpose() * y_neg;
    // cout<<"true_positive"<<true_positive<<endl;
    // cout<<"true_negtive"<<true_negtive<<endl;
    // cout<<"false_positive"<<false_positive<<endl;
    // cout<<"false_negtive"<<false_negtive<<endl;

    return (double) (true_positive + false_negtive) / y.rows();
    // return 0.5;
}

int main(int argc, char const *argv[])
{
    MatrixDataset TrainDataset("./out/hog.ptbf",false);
    MatrixDataset TestDataset("./out/hog.ptbf",true);

    // MatrixDataset::matrix trainX = TrainDataset.X();
    // MatrixDataset::vector trainY = TrainDataset.Y();
    MatrixDataset::matrix testX = TestDataset.X();
    MatrixDataset::vector testY = TestDataset.Y();
    Eigen::VectorXd testY_;
    int p = TrainDataset.P();
    Logistictor a(p);
    cout<<"testX.mean \n"<<testX.mean()<<endl;
    cout<<"testY.mean \n"<<testY.mean()<<endl;
    // a.train(trainY,trainX,1,0.1,0.001);

    testY_ = a.forward(testX);
    cout<<"train accuracy: "<<accuracy(testY, testY_)<<endl;

    a.train(testY,testX,1,0.01,0);

    testY_ = a.forward(testX);
    cout<<"train accuracy: "<<accuracy(testY, testY_)<<endl;

    // MatrixDataset::vector testY_ = a.forward(testX);
    
    // cout<<"forward:\n"<<a.forward(X)<<endl;

    // a.train(Y,X,100,0.1,0.001);
    // cout<<a.forward(X)<<endl;

    return 0;
}
