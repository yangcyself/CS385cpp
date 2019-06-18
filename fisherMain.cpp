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
double accuracy(Eigen::VectorXd y, Eigen::VectorXd y_, double thresh)
{
    cout<<"y_.mean, max, min"<<y_.mean() << " " << y_.maxCoeff()<< " " << y_.minCoeff() <<endl;
    
    Eigen::VectorXi y_true = y.cast <int> ();
    Eigen::VectorXi y_false = (1-y.array()).cast <int> ();
    Eigen::VectorXi y_pos = (y_.array() >= thresh).cast <int> ();
    Eigen::VectorXi y_neg = (y_.array() < thresh).cast <int> ();
    // cout<<"y_.head(100): \n"<<y_.head(100)<<endl;
    // cout<<"y_pos.shape: "<< y_pos.rows()<<endl;
    cout<<"y_pos.count() "<<y_pos.sum()<<endl;
    cout<<"y_neg.count() "<<y_neg.sum()<<endl;
    int true_positive = y_true.transpose() * y_pos;
    int true_negtive = y_true.transpose() * y_neg;
    int false_positive = y_false.transpose() * y_pos;
    int false_negtive = y_false.transpose() * y_neg;
    cout<<"true_positive"<<true_positive<<endl;
    cout<<"true_negtive"<<true_negtive<<endl;
    cout<<"false_positive"<<false_positive<<endl;
    cout<<"false_negtive"<<false_negtive<<endl;

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
    int p = TestDataset.P();
    fisheror a(p);
    a.train(trainY,trainX);
    // a.train(testY,testX);

    // cout<<"X:/n"<<X<<"Y:/n"<<Y<<endl;
    double thresh = a.threshold();
    MatrixDataset::vector trainY_ = a.forward(trainX);
    cout<<"train accuracy: "<<accuracy(trainY, trainY_,thresh)<<endl;

    MatrixDataset::vector testY_ = a.forward(testX);
    cout<<"test accuracy: "<<accuracy(testY, testY_,thresh)<<endl;


    return 0;
}
