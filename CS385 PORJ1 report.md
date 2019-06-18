# CS385 PORJ1 report

杨晨宇 516030910537

I have to say sorry before beginning this report for this project is really a poorly completed one. I devoted most of my time into implementation of the framework in C++, and resulted in an unfinished work that can be seen on [github](https://github.com/yangcyself/CS385cpp.git). The C++ work depends on three packages: 1) **Google's protocol buffer** for storing and communicating the data,  2） **Eigen** for matrix calculation and manipulation, 3）**opencv** for basic picture processing. I challenged myself in building a simple machine learning system on my own, and now stop at the stage that basic network structures are **tested and can work** (see the [test programs](<https://github.com/yangcyself/CS385cpp/tree/master/test>)). Now the time does not allow me to further finish this course work on C++. Thus I have to restart this project on python, and the following sections introduce the finished parts (three in c++ and three in python) for this project. 

## Raw Picture Processing

### cutting resizing and dividing datasets

The raw picture processing source file is [processImage.cpp](./processImage.cpp).  In this program, the raw faces are cut, resized, and putted into corresponding folder. To generate a full dataset, use the bash script [generageDataset.sh](./generageDataset.sh) to call the picture processing program 9 times, each time indicating the input folder, whether the output is train dataset or test dataset, and whether to generate negative samples.

`bash ./generageDataset,sh`

### calculating and visualizing Hog features

I use the `opencv::HOGDescriptor` to calculate the Hog features, forms a 900d descript vector for each picture , as in [hogCalculate.cpp](./hogCalculate.cpp) . I save the calculated the Hog feature into binary file using Protocol buffer, formatted as in the file [dataset.hog.proto](protobuf\dataset.hog.proto)

To visualize the Hog features, use the program complied from [hogOutVisualize.cpp](./hogOutVisualize.cpp), which reads in the protobuf binary file and visualize the hog feature saved in it. Here I visualize the hog vectors on the original graph for a better visualization effect. Some example visualization results are as following.

![visualize](./reportPics/1.jpg) ![visualize](./reportPics/2.jpg) ![visualize](./reportPics/3.jpg) ![visualize](./reportPics/4.jpg) ![visualize](./reportPics/5.jpg) ![visualize](./reportPics/6.jpg)



## Logistic Model

The C++ Logistic implementation is in [logistic](./logistic), and using the [logisticMain.cpp](logisticMain.cpp) to train and test the model.  With learning rate 0.1 and Langevin 0.001

```bash
./logisticMain 0.01 0.001
```

After training for 1 epoch, the train accuracy and test accuracy can all reach 0.8. But that's because the model learned to guess 0 for all inputs.

After training for 10 epoch, the situation still remains the same,  and so are the other epochs

With learning rate 0.0005 and Langevin 0.00001

```bash
./logisticMain 0.0005 0.00001
```

The accuracies of the initial epochs are:

> train test
>
> 0.81 0.82
>
> 0.95 0.96
>
> 0.95 0.96
>
> and on ...

And at epoch 100, the model can reach

0.971 for train accuracy and 0.975 for test accuracy



## Fisher Model

The fisher model is implemented in the folder [fisher](./fisher), and using [fisherMain.cpp](fisherMain.cpp) to train and test the model.

When the model do classifying, it uses the middle point of the embedded mean value of positive samples and negative samples as the threshold.  The final accuracy is **0.980 for training and 0.977 for testing**.

The embedded training set has the following inter and intra variance:

> intra variance for positive samples: 0.00929 
>
> intra variance for negative samples: 0.00245
>
> inter variance for whole samples: 0.0199

We can find that the inter variance is much larger than inter variance.



## SVMs

The code for SVM can be seen in [SVM.py](./python/SVM.py) In which four kinds of kernels are tested, namely: "linear", "poly", "rbf", and "sigmoid". The scripts uses the SVC class provided by `sklearn`. 

The testing result can be shown following:



### linear

ACC:  0.9803959254276379
support number:  1261

![](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_1208.jpg) ![supportimg_linear_1723](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_1723.jpg) ![supportimg_linear_2124](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_2124.jpg) ![supportimg_linear_2145](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_2145.jpg) ![supportimg_linear_3441](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_3441.jpg) ![supportimg_linear_3914](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_3914.jpg) ![supportimg_linear_5084](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_5084.jpg) ![supportimg_linear_6291](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_6291.jpg) ![supportimg_linear_8653](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_8653.jpg) ![supportimg_linear_8677](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_8677.jpg) ![supportimg_linear_10096](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_10096.jpg) ![supportimg_linear_10097](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_10097.jpg) ![supportimg_linear_12446](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_12446.jpg) ![supportimg_linear_13053](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_13053.jpg) ![supportimg_linear_14446](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_14446.jpg) ![supportimg_linear_17309](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_17309.jpg) ![supportimg_linear_18221](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_18221.jpg) ![supportimg_linear_18543](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_18543.jpg) ![supportimg_linear_19260](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_19260.jpg) ![supportimg_linear_19762](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_19762.jpg) ![supportimg_linear_20286](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_linear_20286.jpg) 



### poly

ACC:  0.8010763021333845
support number:  8272

![](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_265.jpg) ![supportimg_poly_1902](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_1902.jpg) ![supportimg_poly_2633](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_2633.jpg) ![supportimg_poly_3002](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_3002.jpg) ![supportimg_poly_3527](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_3527.jpg) ![supportimg_poly_3933](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_3933.jpg) ![supportimg_poly_5914](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_5914.jpg) ![supportimg_poly_6132](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_6132.jpg) ![supportimg_poly_7978](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_7978.jpg) ![supportimg_poly_8550](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_8550.jpg) ![supportimg_poly_10984](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_10984.jpg) ![supportimg_poly_11420](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_11420.jpg) ![supportimg_poly_11756](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_11756.jpg) ![supportimg_poly_14865](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_14865.jpg) ![supportimg_poly_15159](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_15159.jpg) ![supportimg_poly_15440](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_15440.jpg) ![supportimg_poly_15447](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_15447.jpg) ![supportimg_poly_15818](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_15818.jpg) ![supportimg_poly_16781](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_16781.jpg) ![supportimg_poly_18162](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_18162.jpg) ![supportimg_poly_18543](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_poly_18543.jpg) 



### rbf

ACC:  0.9667499519507976
support number:  3660

![](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_11.jpg) ![supportimg_rbf_173](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_173.jpg) ![supportimg_rbf_301](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_301.jpg) ![supportimg_rbf_590](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_590.jpg) ![supportimg_rbf_684](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_684.jpg) ![supportimg_rbf_987](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_987.jpg) ![supportimg_rbf_1444](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_1444.jpg) ![supportimg_rbf_1562](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_1562.jpg) ![supportimg_rbf_1697](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_1697.jpg) ![supportimg_rbf_1702](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_1702.jpg) ![supportimg_rbf_1890](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_1890.jpg) ![supportimg_rbf_2502](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_2502.jpg) ![supportimg_rbf_2559](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_2559.jpg) ![supportimg_rbf_2932](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_2932.jpg) ![supportimg_rbf_3826](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_3826.jpg) ![supportimg_rbf_6138](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_6138.jpg) ![supportimg_rbf_7940](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_7940.jpg) ![supportimg_rbf_9419](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_9419.jpg) ![supportimg_rbf_10803](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_10803.jpg) ![supportimg_rbf_11759](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_11759.jpg) ![supportimg_rbf_17428](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_rbf_17428.jpg) 



### sigmoid

ACC:  0.9621372285220066
support number:  4597

![](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_73.jpg) ![supportimg_sigmoid_75](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_75.jpg) ![supportimg_sigmoid_529](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_529.jpg) ![supportimg_sigmoid_531](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_531.jpg) ![supportimg_sigmoid_650](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_650.jpg) ![supportimg_sigmoid_1320](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_1320.jpg) ![supportimg_sigmoid_1645](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_1645.jpg) ![supportimg_sigmoid_2304](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_2304.jpg) ![supportimg_sigmoid_2394](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_2394.jpg) ![supportimg_sigmoid_2459](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_2459.jpg) ![supportimg_sigmoid_2603](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_2603.jpg) ![supportimg_sigmoid_3097](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_3097.jpg) ![supportimg_sigmoid_3495](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_3495.jpg) ![supportimg_sigmoid_3692](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_3692.jpg) ![supportimg_sigmoid_4079](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_4079.jpg) ![supportimg_sigmoid_8548](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_8548.jpg) ![supportimg_sigmoid_10512](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_10512.jpg) ![supportimg_sigmoid_14191](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_14191.jpg) ![supportimg_sigmoid_15010](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_15010.jpg) ![supportimg_sigmoid_17143](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_17143.jpg) ![supportimg_sigmoid_19062](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\reportPics\supportimg_sigmoid_19062.jpg) 



### Observation

We can find an interesting phenomenon that the more complex the kernel (that is, more non-linearity), the poorer the testing performance. For example, the poly model uses most support vectors (means the hyper plane is complex), but the testing accuracy is little greater than guess all zero.  From this observation, we can say that the most important issue for SVM in this task is overfitting. Thus, the higher expressiveness in model leads to higher dangers in overfitting.



## CNN

### CNN with C++

The C++ implementation of CNN can be seen on [github](<https://github.com/yangcyself/CS385cpp/tree/master/cnn>). However, as mentioned at the beginning, I do not have the time to test on a real classification task (due to the amount of work in implementing a minibatch data loader and the potential difficulties). I have checked the correctness of each modules by comparing the result on toy data with the result calculated by hand. 

To construct a CNN frame work by hand, two key issues have to be solved. The first one is the convolution calculation, and the second one is to design a abstraction that allows forward and backward propagation. 



### CNN with off the shelf frameworks

I use Keras as the frame work for it is super easy. I built a four-layer network with feature map channel numbers `64, 256, 256, 256` 

With learning rate = 0.001, weight decay= 1e-6 and momentum=0.9, after trained for 20 epochs, finally the training and testing accuracy reaches 

> train accuracy: 0.9815  	test accuracy: 0.9822

## face detection

The face detection file can be seen at [python/faceDection.py](./python/faceDection.py) , in which a window is cut from all kinds of positions and scales, and run the classifier to get the result. The result can be seen as following. (the result has been choose to be representative)

### Linear SVM

![](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\linearSVM_4.jpg)

![linearSVM_7](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\linearSVM_7.jpg)

![linearSVM_8](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\linearSVM_8.jpg)

![linearSVM_9](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\linearSVM_9.jpg)

![linearSVM_12](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\linearSVM_12.jpg)

![linearSVM_14](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\linearSVM_14.jpg)

![linearSVM_18](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\linearSVM_18.jpg)

#### polySVM

![](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\polySVM_9.jpg)

The Poly SVM can hardly find any positive results

#### rbfSVM

![](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\rbfSVM_7.jpg)

![rbfSVM_14](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\rbfSVM_14.jpg)

![rbfSVM_17](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\rbfSVM_17.jpg)

![sigmoidSVM_2](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\sigmoidSVM_2.jpg)

![sigmoidSVM_3](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\sigmoidSVM_3.jpg)

![sigmoidSVM_4](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\sigmoidSVM_4.jpg)

#### CNN 

![](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\CNN_0.jpg)

![CNN_7](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\CNN_7.jpg)

![CNN_13](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\CNN_13.jpg)

#### LOGISTIC

![](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\LOGISTIC_2.jpg)

![LOGISTIC_3](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\LOGISTIC_3.jpg)

![LOGISTIC_4](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\LOGISTIC_4.jpg)

![LOGISTIC_6](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\LOGISTIC_6.jpg)

![LOGISTIC_7](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\LOGISTIC_7.jpg)

![LOGISTIC_12](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\LOGISTIC_12.jpg)

![LOGISTIC_19](D:\yangcy\UNVjunior\CS385\PROJ1\CS385cpp\python\out\LOGISTIC_19.jpg)

we can find that if the boxes can be clustered together, the result will be much better.

