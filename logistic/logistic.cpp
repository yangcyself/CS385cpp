#include <logistic/logistic.h>

// #include <stdio.h>
#include <iostream>
namespace logistic{

/**
 * The sigmoid function given Xbeta as x
 */
double sigmoid(double x) // the functor we want to apply
{
    return 1/( 1 + std::exp(-x) );
}


Logistictor::Logistictor(int p):beta(p+1)
{
    // beta << 1,0,2; //just for debug
    beta<<vector::Random(p+1);
}


Logistictor::~Logistictor()
{}

Logistictor::matrix
Logistictor::augment(const Logistictor::matrix& X )const
{
    matrix aX(X.rows(), X.cols()+1); // the larger X concated with one vector
    aX << X, matrix::Constant(X.rows(), 1, 1.0);
    return aX;
}

Logistictor::vector
Logistictor::forward(const Logistictor::matrix& X)const
{
    matrix aX = augment(X);
    vector logit = aX*beta;
    return logit.unaryExpr(std::ptr_fun(sigmoid));
    /**
     * IMPORTANT: CANNOT USE THIS LINE IF 
     * -I/home/yangcy/programs/eigen instead of $(pkg-config --cflags eigen3) in compile
     * the error message is "error: base type ‘double (*)(double)’ fails to be a struct or class type" 
     *  if use the following line
     * XIT!
     */
    // return logit.unaryExpr(&(sigmoid)); 

}

void 
Logistictor::train(const Logistictor::vector& Y, const Logistictor::matrix& X, 
                        const int steps, const double lr, double lgv)
{
    matrix aX = augment(X);
    for(int i = 0;i<steps;i++){
        vector p = forward(X); // probability
        // vector d = (X.rowwise().transpose()*(Y-p)).sum(); //derivative
        vector dyp = Y-p;
        vector grad = dyp.array() * (1-p.array())*p.array(); // the gradient of sigmoid
        vector d = grad.transpose()*aX; //derivative n
        // vector d = dm.transpose().colwise().sum();
        // std::cout<<d<<std::endl;
        beta += lr*d;

        //Langevin:
        if(lgv>0.000001){
            vector mean = vector::Zero(beta.rows());
            matrix cov = matrix::Identity(beta.rows(),beta.rows());
            cov = cov * lgv;
            Eigen::EigenMultivariateNormal<double> normx(mean,cov);
            beta += normx.samples(1);
        }
    }
}


} // name space logistic