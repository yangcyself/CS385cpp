#ifndef LOGISTIC_H_
#define LOGISTIC_H_

#include <Eigen/Dense>
#include <Eigen/Core>
#include <util/eigenmvn.h>
namespace fisher{
/**
 * The fisheror class, hosting a model and its variables, can train and inference.
 * contains a vector beta as the variable
 * 
*/
class fisheror{
public:
    using vector = Eigen::VectorXd;
    using matrix = Eigen::MatrixXd;
private:
    vector beta;

    /**
     * calculate the covariance matrix given matrix n*p 
     */
    matrix cov(const matrix& X)const;
    vector miu(const matrix& X)const;
public:

    /**
     * constructor:
     * p: int, number of features  
     * will generate a p dim vector as beta
     */
    fisheror(int p);

    ~fisheror();

    /**
     * give a matrix n*p, multiply with the beta, (add bias) return a vector n
     */
    vector forward(const matrix& X) const;


    /**
     * calculat the optimal projection vector beta
     */
    void train(const vector& Y, const matrix& X);

};


} /* namespace logistic */

#endif //LOGISTIC_H_