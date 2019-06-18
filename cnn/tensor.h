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
 * 
 * basically, there are two types of tensor: 
 *  one is for feature map N * CHW 
 *  another is kernel outC * H*W*inC 
 *  data field is the matrix, and H and W are H and W in the above lines
 * 
 * TODO: optimize ColomnBlockIndex and other column arrangement function, they have the same structure.
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
    Tensor(){W = H = -1;} // to indicate the tensor is not initialized
    ~Tensor(){}

    /**
     * return whether the tensor is an empty one,
     *  can be used to check the validaty 
     */
    bool is_empty(){return H<=0&&W<=0;}
    void set_empty(){H = W = -1;}

    int height()const {return H;}
    int width()const {return W;}
    matrix mat()const {return data;}

    /**
     * choose one row n and expand it as w*h
     *  note the w and h here is the w and h of the matrix, not of the tensor
     * if n is -1, just return the data field in tensor
     */
    matrix expand(int n, int h, int w)const ;

    /**
     * if the tensor is a conv kernel, i.e. outC * H*W*inC
     *  flip the kernel to get k_(hi,wj) = k(H-hi, W-wj)
     */
    Tensor kernelFlip()const; 

    /**
     * transpose a tensor into kernel's representation
     *  N * inC*H*W -> inC * H*W*N
     * this can be done by simply mapping from row storage to col storage.
     */
    Tensor kernelize()const;

    /**
     * transpose a kernel tensor outC * H*W*inC into inC * H*W*outC
     *  is transposefromNHWC  and then kernelize
     */
    Tensor kernelTranspose()const;

    /**
     * transpose the representation between "feature map"  and "open cv"
     *  i.e. NCHW -> NHWC
     */
    Tensor transposetoNHWC()const;

    /**
     * transpose the representation between "feature map"  and "open cv"
     *  i.e. NHWC -> NCHW
     */
    Tensor transposefromNHWC()const;

    /**
     * kernel tensors are represented OutC * (Hk * Wk * InC)
     * this means the kernel's one line is like C1X1Y1 C2X1Y1 C3X1Y1 ... CnX1Y1 C1X1Y2 ...
     * in conv the data are arranged in the shape (H*W) * (Hk * Wk * C)
     * So the kernel has to be (Hk* Wk* C) * OutC [NOTE: C is the inner loop]
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
    Tensor eleadd(const Tensor& other) const;

    /**
     * multiply with another tensor element wise, with the function of broadcast
     */
    Tensor elemul(const Tensor& other) const;
    
    void print();


};


} /* namespace logistic */

#endif //TENSOR_H_