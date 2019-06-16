#ifndef LOGISTIC_H_
#define LOGISTIC_H_

#include <Eigen/Dense>
#include <Eigen/Core>
#include <util/eigenmvn.h>
namespace logistic{
/**
 * The logistictor class, hosting a model and its variables, can train and inference.
 * contains a vector beta as the variable
 * 
*/
class Logistictor{
public:
    using vector = Eigen::VectorXd;
    using matrix = Eigen::MatrixXd;
private:
    vector beta;

    /**
     * augment the matrix X n*p to n*(p+1)
     */
    matrix augment(const matrix& X) const;
public:
    /**
     * constructor:
     * p: int, number of features  
     * will generate a p+1 dim vector as beta
     */
    Logistictor(int p);

    ~Logistictor();

    /**
     * give a matrix n*p, multiply with the beta, (add bias) return a vector n
     */
    vector forward(const matrix& X) const;


    /**
     * train the beta for number of steps with learning rate lr
     * lgv: the variance of the Langevin randomness
     */
    void train(const vector& Y, const matrix& X, const int steps, const double lr, const double lgv = 0);
};


} /* namespace logistic */

#endif //LOGISTIC_H_