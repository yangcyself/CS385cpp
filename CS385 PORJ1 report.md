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



## CNN

The c++ implementation of CNN can be seen on [github](<https://github.com/yangcyself/CS385cpp/tree/master/cnn>). However, as mentioned at the beginning, I do not have the time to test on a real classification task (due to the amount of work in implementing a minibatch data loader and the potential difficulties). I have checked the correctness of each modules by comparing the result on toy data with the result calculated by hand. 

## face detection

## feature distribution

# Stories (met problems)

