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
// using std::vector;
// using std::cin;
// using std::cout;
// using std::endl;
// using std::string;

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
    dscrp -> set_classtype(classtype=="pos"? dataset::ImageDescrip::POS : dataset::ImageDescrip::NEG );
    dscrp -> set_datatype(datatype=="train"? dataset::ImageDescrip::TRAIN : dataset::ImageDescrip::TEST);
    for (int i = 0;i< descriptorsValues.size();i++){
        dscrp -> add_imghog(descriptorsValues[i]);
    }
}

/**
 * To read all image names in a dir and call addAnImage
 */
void processAllImg(dataset::hog& imagehogs, HOGDescriptor d,string datatype, string classtype)
{
    string dir = ImgFolder + "/" + datatype + "/" + classtype;
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return;
    }
    while ((dirp = readdir(dp)) != NULL) {
        // cout<<string(dirp->d_name)<<endl;
        addAnImage(imagehogs.add_image(),d,datatype,classtype,dirp->d_name);
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
    dataset::hog imagehogs; 
    processAllImg(imagehogs,d,"train","pos");
    processAllImg(imagehogs,d,"train","neg");
    processAllImg(imagehogs,d,"test","pos");
    processAllImg(imagehogs,d,"test","neg");
    fstream output("./out/hog.ptbf", ios::out | ios::trunc | ios::binary); 

    if (!imagehogs.SerializeToOstream(&output)) { 
        cerr << "Failed to write msg." << endl; 
        return -1; 
    }         
    return 0;
}