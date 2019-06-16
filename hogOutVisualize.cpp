/**
 * Visualize the hog feature saved in protobuf file
 * ./hogOutVisualize <visualize index>
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

#if 1 //comment out the get_hog... function

Mat get_hogdescriptor_visu(const Mat& color_origImg, 
            vector<float>& descriptorValues, const Size & size )
{
    const int DIMX = size.width;
    const int DIMY = size.height;
    float zoomFac = 3;

    Mat visu;
    resize(color_origImg, visu, Size( (int)(color_origImg.cols*zoomFac), 
                                        (int)(color_origImg.rows*zoomFac) ) );
 
    int cellSize        = 16;
    int gradientBinSize = 9;
    float radRangeForOneBin = (float)(CV_PI/(float)gradientBinSize); 
            // dividing 180 into 9 bins, how large (in rad) is one bin?
 
    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = DIMX / cellSize;
    int cells_in_y_dir = DIMY / cellSize;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1; // - step size
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
 
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                cellx = blockx;
                celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
 
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
 
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
 
                } // for (all bins)
 
 
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
 
            } // for (all cells)
 
 
        } // for (all block x pos)
    } // for (all block y pos)
 
 
    // compute average gradient strengths
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {
 
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
 
    // draw cells
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;
 
            int mx = drawX + cellSize/2;
            int my = drawY + cellSize/2;
 
            rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX+cellSize)*zoomFac), (int)((drawY+cellSize)*zoomFac)), Scalar(100,100,100), 1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = (float)(cellSize/2.f);
                float scale = 2.5; // just a visualization scale, to see the lines better
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient visualization
                line(visu, Point((int)(x1*zoomFac),(int)(y1*zoomFac)), Point((int)(x2*zoomFac),(int)(y2*zoomFac)), Scalar(0,255,0), 1);
 
            } // for (all bins)
 
        } // for (cellx)
    } // for (celly)
 
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
        for (int x=0; x<cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
 
    return visu;
} 
#endif


const string OUTFOLDER="./tmp/";

int main( int argc, char** argv )
{
    cout<<"start"<<endl;
    dataset::hog imagehogs; 
    { 
        fstream input("./out/hog.ptbf", ios::in | ios::binary); 
        if (!imagehogs.ParseFromIstream(&input)) { 
            cerr << "Failed to parse address book." << endl; 
            return -1; 
        } 
    } 
    if(argc < 1){
        cout <<"usage: ./hogOutVisualize <visualize index>" <<endl;
        return -2;
    }
    for (int i = 1; i< argc;i++){
        int ind = atoi(argv[i]);
        cout << "Loading the index "<< ind<< " from protobuf file"<< endl;
        const dataset::ImageDescrip& tmp = imagehogs.image(ind);
        cout<< "num: "<<i<<"\t";
        cout<<"imgPath: "<<tmp.imgpath()<<"\n";
        cout<<"Classtype: "<<tmp.classtype()<<"\t";
        cout<<"Datatype: "<<tmp.datatype()<<"\t";


        vector<float> descriptorsValues;
        for (int j = 0;j<tmp.imghog_size();j++){
            descriptorsValues.push_back(tmp.imghog(j));
        }
        Mat img = imread(tmp.imgpath(), 0);
        cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << endl;
        cout << "Found " << descriptorsValues.size() << " descriptor values" << endl;
        Mat visu = get_hogdescriptor_visu(img,descriptorsValues,Size(96,96));

        /**
         * Now the following three lines with namedWindow and waitkey cannot be compiled together with
         * protobuf.
         * Otherwise an fatal error will be raised:
         * [libprotobuf FATAL ../../src/google/protobuf/stubs/common.cc:86] This program was compiled
         *  against version 2.6.1 of the Protocol Buffer runtime library, which is not compatible with 
         *  the installed version (3.7.1).  Contact the program author for an update.  If you compiled 
         *  the program yourself, make sure that your headers are from the same version of Protocol 
         *  Buffers as your link-time library.  
         * Shit
         */
        // namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
        // imshow( "Display window", visu );                // Show our image inside it.
        // waitKey(0); // Wait for a keystroke in the window

        imwrite(OUTFOLDER+to_string(i)+".jpg",visu);

    }
    return 0;
}