/**
 * 
 * usage
 * g++ -I/home/yangcy/programs/eigen -I. fisherMain.cpp fisher/fisher.cpp dataset/matrix_dataset.cpp  protobuf/dataset.hog.pb.cc -o fisherMain  $(pkg-config --cflags --libs protobuf) -std=c++11
 * ./fisherMain
 */

#include <fisher/fisher.h>
#include <iostream>
#include <Eigen/Dense>
#include <dataset/matrix_dataset.h>

using namespace std;
using namespace fisher;
using namespace dataset;

/**
 * the accuracy needs normalize
 */
double accuracy(Eigen::VectorXd y, Eigen::VectorXd y_)
{
    y_.normalize();
    cout<<"y_.mean, max, min"<<y_.mean() << " " << y_.maxCoeff()<< " " << y_.minCoeff() <<endl;
    double thresh = y_.maxCoeff()*0.2 + y_.minCoeff()*0.8;
    Eigen::VectorXi y_true = y.cast <int> ();
    Eigen::VectorXi y_false = (1-y.array()).cast <int> ();
    Eigen::VectorXi y_pos = (y_.array() >= thresh).cast <int> ();
    Eigen::VectorXi y_neg = (y_.array() < thresh).cast <int> ();
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

    MatrixDataset::matrix trainX = TrainDataset.X();
    MatrixDataset::vector trainY = TrainDataset.Y();
    MatrixDataset::matrix testX = TestDataset.X();
    MatrixDataset::vector testY = TestDataset.Y();

    fisheror a(2);
    a.train(trainY,trainX);
    // cout<<"X:/n"<<X<<"Y:/n"<<Y<<endl;

    MatrixDataset::vector trainY_ = a.forward(trainX);
    cout<<"train accuracy: "<<accuracy(trainY, trainY_)<<endl;

    MatrixDataset::vector testY_ = a.forward(testX);
    cout<<"train accuracy: "<<accuracy(testY, testY_)<<endl;


    return 0;
}
