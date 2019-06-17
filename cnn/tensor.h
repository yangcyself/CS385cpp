#ifndef TENSOR_H_
#define TENSOR_H_

#include <Eigen/Dense>
#include <Eigen/Core>

namespace convnn{
/**
 * The Tensor class (NCHW), contains an inner matrix (N * CHW)
 * The matrix is stored in row-major order
 * all the computation is carried out `for i in N`
 * all the convnn calculation can be represented by the tensors operation
*/
class Tensor{
public:
    using vector = Eigen::VectorXd; // this should be tested as to compatible with col major matrix
    using matrix = Eigen::Matrix<double,  Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>;

private:
    matrix data;
    int W,H;

public:
    friend Tensor::matrix augment(Tensor a, int kh, int kw);
    /**
     * constructor:
     */
    Tensor(matrix d, int h=1, int w=1);

    ~Tensor(){}

    /**
     * choose one row n and expand it as w*h
     *  note the w and h here is the w and h of the matrix, not of the tensor
     * if n is -1, just return the data field in tensor
     */
    matrix expand(int n, int h, int w)const ;

    /**
     * kernel tensors are represented OutC * (Hk * Wk * InC)
     * this means the kernel's one line is like C1X1Y1 C2X1Y1 C3X1Y1 ... CnX1Y1 C1X1Y2 ...
     * in conv the data are arranged in the shape (H*W) * (Hk * Wk * C)
     * So the kernel has to be transpose into the (Hk* Wk* C) * OutC [NOTE: C is the inner loop]
     * **BASIC IDEA**
     * transfer one row of the input data into a matrix, 
     * augment the matrix in order to correspond each value to the kernel
     * Then do the matrix multiplication
     * **ARGUMENTS**
     * kernel: the kernel
     * pad: the padding around the input feature map(can only pad around)
     *       when pad = -1, pad (k-1)/2 around the kernel
     * padv: the value of the const padding
     * stride: the stride
     */
    // 
    Tensor conv(const Tensor& kernel, int pad = -1, double padv = 0 ,int stride = 1)const;


    /**
     * add with another tensor element wise, with the function of broadcast
     */
    Tensor eleadd(const Tensor& other);

    /**
     * multiply with another tensor element wise, with the function of broadcast
     */
    Tensor elemul(const Tensor& other);
    
    void print();


};


} /* namespace logistic */

#endif //TENSOR_H_