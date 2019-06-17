#include <cnn/tensor.h>

// #include <stdio.h>
#include <iostream>
namespace convnn
{

/**
 * for a matrix that arranged in a row W*H, find the indexs of a submatrix 0-h 0-w
 * s is the stride
 */
Eigen::VectorXi ColomnBlockIndex(int H, int W, int h, int w, int s = 1)
{
    w = (w+s-1)/s;
    h = (h+s-1)/s;
    Eigen::VectorXi irow(w);
    for (int i = 0;i<w;i++){
        irow(i) = i*s;
    }
    Eigen::VectorXi ind(w*h);
    for (int i = 0;i<h;i++)
        ind.segment(i*w,w) = irow.array() + i*W*s;
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
Tensor::kernelFlip()const
{
    int outC = data.rows();
    int inC = data.cols()/H/W;
    int ns = H*W;
    /**
     * The  3*3 0-8's flip is simply 8-0
     */
    Eigen::VectorXi irow(inC);
    for (int i = 0;i<inC;i++){
        irow(i) = i;
    }

    Eigen::VectorXi ind(inC*ns);
    for (int i = 0;i<ns;i++){
        int ii = ns - i -1; 
        ind.segment(i*inC,inC) = irow.array() + ii*inC;
    }
    matrix res =  data(Eigen::all, ind);
    return Tensor(res,H,W);
}


Tensor
Tensor::conv(const Tensor& kernel, int pad, double padv,int stride)const
{
    int n = data.rows();
    int inc = data.cols()/H/W;
    matrix kermat = kernel.data;
    int outc = kermat.rows();
    int kh = kernel.H; int kw = kernel.W;

    int padh = pad; int padw = pad;
    if(pad==-1){
        padh = (kh-1)/2;
        padw = (kw-1)/2;
    }
    int padedh = (H+2*padh); int padedw = (W+2*padw);

    int augh = (padedh-kh+1); int augw = padedw-kw+1;

    int outH = (augh + stride -1 )/stride;
    int outW = (augw + stride -1 )/stride;
    matrix out(n,outH*outW*outc);
    // std::cout<<"#0"<<std::endl;
    for(int i = 0; i < n ;i++){
        matrix tmp = expand(i,inc,H*W); //inc * H*W

        /**
         * pad the matrix tmp
         */
        // std::cout<<"#1"<<std::endl;
        
        matrix padded = matrix::Constant (tmp.rows(), padedh * padedw , padv);
        Eigen::VectorXi ind = ColomnBlockIndex( padedh , padedw , H , W);
        // std::cout << ind <<std::endl;
        padded(Eigen::all, ind.array() + padw + padh * padedw ) = tmp;
        // std::cout<<"#2"<<std::endl;

        /**
         * augment the matrix
         * get C*kh*kw * (H-kh+1)*(W-kw+1)
         */

        matrix augmented = augment(Tensor(padded,padedh,padedw),kh,kw);
        // std::cout<<"padded: \n"<< padded<<std::endl;
        // std::cout<<"augmented: \n"<< augmented<<std::endl;
        // std::cout<<"kermat: \n"<< kermat<<std::endl;

        /**
         * stride the matrix
         */
        matrix stridded;
        if(stride == 1)
            stridded = augmented;
        else{
            stridded = augmented(Eigen::all,ColomnBlockIndex(augh,augw,augh,augw,stride) );
        }
        matrix tmpout =  kermat * stridded; // (outc) * (H*W)
        // std::cout<<"#3"<<std::endl;
        Eigen::Map<Eigen::RowVectorXd> v(tmpout.data(), tmpout.size());
        // std::cout<<"#4"<<std::endl;
        out.row(i) = v;
        // std::cout<<"#5"<<std::endl;
    }
    return Tensor(out,outH,outW);
}

void
Tensor::print()
{
    
    std::cout << "DATA: \n" << data<< std::endl;
    std::cout <<"N C H W: "<< data.rows()<<" " << data.cols()/H/W<<" " << H<<" " << W <<std::endl; 
    // int inc = data.cols()/H/W;
    // for(int i = 0;i<data.rows();i++){
    //     std::cout << "\n" << expand(i,inc,H*W )<< std::endl;
    // }
}

Tensor
Tensor::eleadd(const Tensor& other)
{
    matrix out = data.array() + other.data.array();
    return Tensor(out, H,W);
}
Tensor 
Tensor::elemul(const Tensor& other)
{
    matrix out = data.array() * other.data.array();
    return Tensor(out, H,W);
}

} // namespace convnn
