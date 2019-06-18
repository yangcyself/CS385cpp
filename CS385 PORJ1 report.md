# CS385 PORJ1 report

杨晨宇 516030910537

I have to say sorry before beginning this report for this project is really a poorly completed one. I devoted most of my time into implementation of the framework in C++, and resulted in an unfinished work that can be seen on [github](https://github.com/yangcyself/CS385cpp.git). The C++ work depends on three packages: 1) **Google's protocol buffer** for storing and communicating the data,  2） **Eigen** for matrix calculation and manipulation, 3）**opencv** for basic picture processing. I challenged myself in building a simple machine learning system on my own, and now stop at the stage that basic network structures are **tested and can work** (see the [test programs](<https://github.com/yangcyself/CS385cpp/tree/master/test>)). Now the time does not allow me to further finish this course work on C++. Thus I have to restart this project on python, and the following sections introduce the finished parts (three in c++ and three in python) for this project. 

## Raw Picture Processing

### cutting resizing and dividing datasets

The raw picture processing source file is [processImage.cpp](./processImage.cpp).  In this program, the raw faces are cut, resized, and putted into corresponding folder. To generate a full dataset, use the bash script [generageDataset.sh](./generageDataset.sh) to call the picture processing program 9 times, each time indicating the input folder, whether the output is train dataset or test dataset, and whether to generate negative samples.

`bash ./generageDataset,sh`

### calculating and visualizing Hog features

I use the `opencv::HOGDescriptor` to calculate the Hog features, as in [hogCalculate.cpp](./hogCalculate.cpp) . I save the calculated the Hog feature into binary file using Protocol buffer, formatted as in the file [dataset.hog.proto](protobuf\dataset.hog.proto)

To visualize the Hog features, use the program complied from [hogOutVisualize.cpp](./hogOutVisualize.cpp), which reads in the protobuf binary file and visualize the hog feature saved in it. Here I visualize the hog vectors on the original graph for a better visualization effect. Some example visualization results are as following.