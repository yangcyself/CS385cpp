/**
 * calculate the hog features and save them into a protobuf
 * ./hogCalculate 
 * now the default process dir is ./out outputfile is ./out/hog.ptbf
*/
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include "protobuf/dataset.hog.pb.h"
#include <iostream>
#include <sys/types.h>
#include <fstream>
#include <cstdio>
#include <vector>
#include <dirent.h>
#include <string>
#include <errno.h>

using namespace cv;
using namespace std;

const string HogFile = "";
const string ImgFolder = "./out";


/**
 * read an image from the ImgFolder/[datatype]/[classtype]/filename, calculates its hog feature
 * write all the information into the dataset dscrp
 * Arguments:
 * dscrp:, the protobuf image descriptor pointer
 * d: use d.compute( img, descriptorsValues, Size(0,0), Size(0,0), locations) to calculate hog
 * datatype test/train
 * classtype neg/pos
 * filename ....
 */
void addAnImage(dataset::ImageDescrip * dscrp, HOGDescriptor& d,
                 string datatype, string classtype, string filename)
{
    string imgpath = ImgFolder + "/" + datatype + "/" + classtype + "/" + filename;
    vector<Point> locations;
    vector<float> descriptorsValues;
    Mat img = imread(imgpath, 0); // load as color image
    d.compute( img, descriptorsValues, Size(0,0), Size(0,0), locations);
    dscrp -> set_imgpath(imgpath);
    dscrp -> set_classtype(classtype=="pos"? dataset::POS : dataset::NEG );
    dscrp -> set_datatype(datatype=="train"? dataset::TRAIN : dataset::TEST);
    for (int i = 0;i< descriptorsValues.size();i++){
        dscrp -> add_imghog(descriptorsValues[i]);
    }
}

/**
 * To read all image names in a dir and call addAnImage
 */
void processAllImg(dataset::hog* imagehogs, HOGDescriptor& d,string datatype, string classtype)
{
    imagehogs -> set_classtype(classtype=="pos"? dataset::POS : dataset::NEG );
    imagehogs -> set_datatype(datatype=="train"? dataset::TRAIN : dataset::TEST);
    // cout<<"[@]1"<<endl;
    string dir = ImgFolder + "/" + datatype + "/" + classtype;
    // cout<<"[@]2"<< dir <<endl;
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return;
    }
    while ((dirp = readdir(dp)) != NULL) {
        if(string(dirp->d_name).size() < 5)
            continue; //dirp can be "."
        // cout<<"[@]3 "<<string(dirp->d_name)<<endl;
        addAnImage(imagehogs->add_image(),d,datatype,classtype,dirp->d_name);
        // cout<<"[@]2"<<endl;
    }
    closedir(dp);
}

int main( int argc, char** argv )
{
    HOGDescriptor d(
        Size(96,96), //winSize
        Size(32,32), //blocksize
        Size(16,16), //blockStride,
        Size(16,16), //cellSize,
        9 //nbins,
        // 1, //derivAper,
        // -1, //winSigma,
        // 0, //histogramNormType,
        // 0.2, //L2HysThresh,
        // false //gamma correction,
        // //nlevels=64
    );
    cout<<"[*]1"<<endl;
    // dataset::hog imagehogs; 
    dataset::hogdataset imagehogs; 

    cout<<"[*]2"<<endl;
    processAllImg(imagehogs.add_data(),d,"train","pos");
    cout<<"[*]3"<<endl;
    processAllImg(imagehogs.add_data(),d,"train","neg");
    processAllImg(imagehogs.add_data(),d,"test","pos");
    processAllImg(imagehogs.add_data(),d,"test","neg");
    cout<<"[*]4"<<endl;
    fstream output("./out/hog.ptbf", ios::out | ios::trunc | ios::binary); 

    if (!imagehogs.SerializeToOstream(&output)) { 
        cerr << "Failed to write msg." << endl; 
        return -1; 
    }         
    return 0;
}