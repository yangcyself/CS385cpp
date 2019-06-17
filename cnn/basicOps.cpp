// #include <cnn/tensor.h>
#include <cnn/basicOps.h>
// #include <stdio.h>
#include <iostream>
namespace convnn
{



Conv::Conv(Operator& aa,Operator& bb, int ipad, int istride = 1)
    :a(&aa,&null_deleter),b(&bb,&null_deleter)
{
    pad = ipad;
    stride = istride;
}

/**
 * The forward function for Conv operators
 *  if the operator is in train mode, it caches the a and b 
 *  to make it convinence to calculate the gradient.
 *  ** Thus, the backward of one loss should be called right after the loss is derived **
 */
Tensor 
Conv::forward()
{
    if(trainmod){
        ca = a->forward();
        cb = b->forward();
        return ca.conv(cb);
    }else{
        return a->forward().conv( b->forward());
    }
}

/**
 * dout/da can be calculate as another conv 
 *  out conv flip(b) pad_: kh-1-pad(h) , kw-1-pad(w)
 *  kernel has to transpose inC and outC
 *  use dilation to deal with forward stride(change it back to the shape before stride)
 * 
 * dout/db can also be calculate as conv similar to above
 *  however, the kernels to gradient should be flip (Not flip the input feature)
 *  The output tensor should be transpose into outC * N*H*W 
 *      and the input tensor should be transpose into inC * H*W*N
 */
void
Conv::backward(const Tensor& in)
{
    assert(trainmod);
    assert(!ca.is_empty() && !cb.is_empty());
    int dpad = cb.height()-1-pad;

    Tensor tpkernel = cb.kernelTranspose(); //transposed kernel
    Tensor fpkernel = tpkernel.kernelFlip(); //fliped kernel
    assert(cb.height()==cb.width()); //for the time being, only support padding the same of H and W
    Tensor da = in.conv(fpkernel,dapad);
    a->backward(da);

    Tensor kout = in.kernelize();
    Tensor tpout = kout.transposefromNHWC();

    Tensor kin = ca.kernelize();
    Tensor tdb = tpout.conv(kin, dpad);
    Tensor db = tdb.kernelFlip();
    b->backward(db);
}

void 
Conv::train(){
    trainmod = true;
    a->train();b->train();
}

void 
Conv::test(){
    trainmod = false;
    a->test();b->test();
    ca.set_empty(); cb.set_empty();
}

}//namespace convnn