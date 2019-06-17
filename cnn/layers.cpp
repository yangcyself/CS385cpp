
#include <cnn/layers.h>
// #include <stdio.h>
#include <iostream>

namespace convnn
{

Tensor 
ReluLayer::reluMask(const Tensor& in)const
{
    matrix pos = (in.mat() > 0).cast <double> ();
    return Tensor(pos);
}

Tensor
ReluLayer::forward()
{

    ca = a->forward();
    return ca.elemul(reluMask(ca));
}


void
ReluLayer::backward(const Tensor& in)
{
    return in.elemul(reluMask(ca));
}

int indInCol(int H, int W, int c, int h, int w)
{
    return c*H*W + x*W + w;
}

void
MaxpoolLayer::indexVector (Tensor& in)
{
    int N = in.mat().rows();
    int H = in.height();
    int W = in.width();
    int C = in.mat().cols()/H/W;

    int w = (W+K-1)/K;
    int h = (H+K-1)/K;

    int elenum = N*C*W*H;
    indevx = Eigen::VectorXi(elenum);
    indevy = Eigen::VectorXi(elenum);

    for(int n = 0;n<N;n++){
        for(int c = 0;c<C;c++){
            for(int i=0;i<h;i++){
                for(int j =0; j<w;j++){
                    int indexPos = j + i*w + c*w*h + n*w*h*C;
                    int hh = i*K; // the upper left corner of the pooling area
                    int ww = j*K;
                    int maxPos = 0;
                    double maxValue = -1;
                    for(int x = 0;x<K;x++){
                        for (int y = 0;y<K;y++){
                            int xx = hh + x;
                            int yy = ww + y;
                            int ind = indInCol(H,W,c,xx,yy);
                            if(ind < in.mat().cols() && in.mat(n,ind) > maxValue)
                                maxPos = ind;
                        }
                    }
                    indevx(indexPos) = n;
                    indevy(indexPos) = maxPos
                }
            }
        }
    }
}

Tensor
MaxpoolLayer::forward()
{
    Tensor res = a->forward();
    indexVector(res);
    inH = res.height();
    inW = res.width();
    inputh = res.mat().rows();
    inputw = res.mat().cols();
    return res(indevx,indevy);

}

void
MaxpoolLayer::backward(const Tensor& in)
{
    matrix grad = matrix::Zero(inputh,inputw);
    grad(indevx,indevy) = in.mat;

    return a->backward(Tensor(grad, inH, inW));
}



}//namespace convnn