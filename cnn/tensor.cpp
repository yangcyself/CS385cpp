#include <cnn/tensor.h>

// #include <stdio.h>
#include <iostream>
namespace convnn
{

/**
 * for a matrix that arranged in a row W*H, find the indexs of a block of this matrix
 * 
 */
Eigen::VectorXi ColomnBlockIndex(int H, int W, int h, int w)
{
    Eigen::VectorXi irow(w);
    for (int i = 0;i<w;i++)
        irow(i) = i;
    Eigen::VectorXi ind(w*h);
    for (int i = 0;i<h;i++)
        ind.segment(i*w,w) = irow.array() + i*W;
    // std::cout<<"H,W,h,w,ind"<<H<<" "<<W<<" "<<h<<" "<<w<<" \n"<<ind <<std::endl;
    return ind;
}

/**
 * stack the matrix C*WH's in the first axis
 * get the C*kh*kw * (H-kh+1)*(W-kw+1)
 * **ARGUMENTS**
 * a: The tensor to be operated on, have the shape C * H*W,
 *      its H and W is just the H and W
 * 
 */
Tensor::matrix augment(Tensor a, int kh, int kw)
{
    int C = a.data.rows();
    int H = a.H;
    int W = a.W;
    assert(H*W == a.data.cols());
    int h = H-kh+1; // the output h and w
    int w = W-kw+1;

    /**
     * build the vector like 0,1,2...w-1, W, W+1, W+2...
     */
    Eigen::VectorXi ind = ColomnBlockIndex(H,W,h,w);

    /**
     * build the concatenate matrix
     */
    Tensor::matrix out(C*kh*kw,h*w);
    for(int i = 0;i<kh;i++)
        for (int j = 0;j<kw;j++)
            out.block((i*kw+j)*C,0,C,w*h) = a.data(Eigen::all,ind.array()+j+i*W);

    return out;
}

Tensor::Tensor(matrix d, int h, int w)
{
    data =  matrix(d.rows(),d.cols());
    data << d; // copy from matrix d
    W = w;
    H = h;
}

Tensor::matrix
Tensor::expand(int n, int h, int w) const
{
    vector t = data.row(n);
    Eigen::Map < matrix > out (t.data(), h,w);
    return out;
}

Tensor
Tensor::conv(const Tensor& kernel, int pad, double padv,int stride)const
{
    int n = data.rows();
    int inc = data.cols()/H/W;
    matrix kermat = kernel.data;
    int outc = kermat.rows();
    matrix out(n,H*W*outc);
    // std::cout<<"#0"<<std::endl;
    for(int i = 0; i < n ;i++){
        matrix tmp = expand(i,inc,H*W); //inc * H*W

        /**
         * pad the matrix tmp
         */
        // std::cout<<"#1"<<std::endl;
        int padh = pad; int padw = pad;
        if(pad==-1){
            padh = (kernel.H-1)/2;
            padw = (kernel.W-1)/2;
        }
        int padedh = (H+2*padh); int padedw = (W+2*padw);
        matrix padded = matrix::Constant (tmp.rows(), padedh * padedw , padv);
        Eigen::VectorXi ind = ColomnBlockIndex( padedh , padedw , H , W);
        // std::cout << ind <<std::endl;
        padded(Eigen::all, ind.array() + padw + padh * padedw ) = tmp;
        // std::cout<<"#2"<<std::endl;
        /**
         * augment the matrix
         * get C*kh*kw * (H-kh+1)*(W-kw+1)
         */
        // std::cout<<"padded: \n"<< padded<<std::endl;
        matrix augmented = augment(Tensor(padded,padedh,padedw),kernel.H,kernel.W);
        // std::cout<<"augmented: \n"<< augmented<<std::endl;
        // std::cout<<"kermat: \n"<< kermat<<std::endl;
        matrix tmpout =  kermat * augmented; // (outc) * (H*W)
        // std::cout<<"#3"<<std::endl;
        Eigen::Map<Eigen::RowVectorXd> v(tmpout.data(), tmpout.size());
        // std::cout<<"#4"<<std::endl;
        out.row(i) = v;
        // std::cout<<"#5"<<std::endl;
    }
    return Tensor(out,H,W);
}

void
Tensor::print()
{
    
    std::cout << "DATA: \n" << data<< std::endl;
    int inc = data.cols()/H/W;
    for(int i = 0;i<data.rows();i++){
        std::cout << "\n" << expand(i,inc,H*W )<< std::endl;
    }
}



} // namespace convnn
