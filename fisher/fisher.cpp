#include <fisher/fisher.h>

// #include <stdio.h>
#include <iostream>
#include <vector>
namespace fisher{

fisheror::vector
fisheror::miu(const fisheror::matrix& X)const
{

    int n = X.rows();
    vector u = X.colwise().sum()/n;
    return u;
}

/**
 * The idea of calculating cov matrix is by cov = E[(x-Ex)*(y-Ey)]
 * thus, first calculate xi-Ex by substracting Ex from the matrix
 * Then the cov_xy is the average of vector [xi-Ex] * [yi - Ey]
 */

fisheror::matrix 
fisheror::cov(const fisheror::matrix& X)const
{

    int n = X.rows();
    int p = X.cols();

    vector u = miu(X);
    matrix delta(n,p);
    delta << X;
    delta.rowwise() -=  u.transpose();   //| x1-Ex  y1-Ey|
                                         //| x2-Ex  y2-Ey|
    matrix res =  delta.transpose() * delta;
    res = res/n;
    return res;
}


fisheror::fisheror(int p):beta(p)
{
    beta<<vector::Random(p);
}


fisheror::~fisheror()
{}

fisheror::vector
fisheror::forward(const fisheror::matrix& X)const
{
    return X*beta;
}

void 
fisheror::train(const fisheror::vector& Y, const fisheror::matrix& X)
{
    // divide X to X_neg and X_pos
    std::vector<int> ipos,ineg; // the index of all pos/neg rows
    int npos = Y.sum();
    int nneg = Y.rows() - npos;
    int n = X.rows();
    int p = X.cols();

    for (int i = 0;i<n;i++){
        if(Y(i))
            ipos.push_back(i);
        else
            ineg.push_back(i);
    }
    matrix Xpos = X(ipos,Eigen::all);
    matrix Xneg = X(ineg,Eigen::all);
    matrix Cpos = cov(Xpos); //covariance
    matrix Cneg = cov(Xneg);
    matrix Cwithin = npos*Cpos + nneg * Cneg;

    // std::cout<<"\n Cpos:\n"<<Cpos<<std::endl;
    // std::cout<<"\n Cneg:\n"<<Cneg<<std::endl;

    vector upos = miu(Xpos);
    vector uneg = miu(Xneg);
    // to make sure that the rank of Cov_within is full, add a very small Identity matrix
    Cwithin += 1e-6 * matrix::Identity(p,p);
    beta = Cwithin.colPivHouseholderQr().solve(upos - uneg);
    beta.normalize();
}


} // name space fisher