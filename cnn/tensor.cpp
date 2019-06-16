#include <cnn/tensor.h>

// #include <stdio.h>
#include <iostream>
namespace convnn
{


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
Tensor::conv(const Tensor& kernel)const
{
    int n = data.rows();
    int inc = data.cols()/H/W;
    matrix kermat = kernel.data;
    int outc = kermat.rows();
    matrix out(n,H*W*outc);
    for(int i = 0; i < n ;i++){
        matrix tmp = expand(i,inc,H*W);
        matrix tmpout =  kermat * tmp; // (outc) * (H*W) 
        // std::cout<<"#1"<<std::endl;
        Eigen::Map<Eigen::RowVectorXd> v(tmpout.data(), tmpout.size());
        // std::cout<<"#2"<<std::endl;
        out.row(i) = v;
        // std::cout<<"#3"<<std::endl;
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
