#ifndef MATRIX_DATASET_H
#define MATRIX_DATASET_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>

namespace dataset{
/**
 * The MatrixDataset class, reads the data saved in protobuf and returns two matrixs X Y
 * used for learning methods like fisher and logistic regression
*/
class MatrixDataset{
public:
    using vector = Eigen::VectorXd;
    using matrix = Eigen::MatrixXd;
private:
    matrix x;
    vector y;
    int n;
    int p;
public:
    /**
     * constructor:
     * load the protoPath's data, and forms X and Y
     */
    MatrixDataset(std::string protoPath, bool train = true);

    ~MatrixDataset();

    /**
     * return the related objects in this dataset
     */
    vector Y();
    matrix X();
    int N();
    int P();
};


} /* namespace dataset */

#endif //MATRIX_DATASET_H