// #include <cnn/tensor.h>
#include <cnn/basicOps.h>
// #include <stdio.h>
#include <iostream>
namespace convnn
{

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
 *  use dilation to deal with forward stride(change it back to the shape before stride)
 * 
 * dout/db can also be calculate as conv similar to above
 *  however, the kernels to gradient should be flip (Not flip the input feature)
 */
void
Conv::backward(const Tensor& in)
{
    assert(trainmod);
    assert(!ca.is_empty() && !cb.is_empty());

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