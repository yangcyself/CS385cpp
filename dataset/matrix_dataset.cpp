#include "dataset/matrix_dataset.h"
#include "protobuf/dataset.hog.pb.h"
#include "fstream"

#include <stdio.h>
#include <iostream>
#include <vector>

namespace dataset{

MatrixDataset::MatrixDataset(std::string protoPath, bool test)
{
    hogdataset imagehogsdataset; 
    { 
        std::fstream input(protoPath, std::ios::in | std::ios::binary); 
        if (!imagehogsdataset.ParseFromIstream(&input)) { 
            std::cerr << "Failed to parse input file." << std::endl; 
        } 
    }
    p = 1;
    n = 0;
    matrix inx(1,p); // use the size 1 1 to indicate the matrix is empty
    vector iny(1);
    for (int dtset = 0;dtset<imagehogsdataset.data_size();++dtset){
        const hog& imagehogs = imagehogsdataset.data(dtset);
        if(imagehogs.datatype() != test)
            continue; 
        p = imagehogs.image(0).imghog_size();
        matrix tmpx = matrix::Zero (imagehogs.image_size(),p);
        vector tmpy = vector::Zero (imagehogs.image_size())  ;
        for (int i = 0;i <imagehogs.image_size();i++){
            const ImageDescrip& tmp = imagehogs.image(i);

            for (int j = 0;j<tmp.imghog_size();j++){
                // cout << tmp.imghog(j)<<" ";
                if(tmp.imghog(j) != tmp.imghog(j))
                {
                    std::cout<<"nan is at: "<< i<<" , "<<j<< " " <<tmp.imghog(j)<<std::endl;
                }
                // assert(tmp.imghog(j) == tmp.imghog(j)); //check nan
                tmpx(i,j) = tmp.imghog(j);
            }
            tmpy(i) = tmp.classtype();
        }
        // concatenate inx
        // std::cout<< "tmpx MEAN " << tmpx.mean() << std::endl;
        if(inx.cols()==1){ //inx is the empty matrix
            
            inx = tmpx;
            iny = tmpy;
        }else{
            matrix tx(inx.rows() + tmpx.rows() , p);
            tx << inx, tmpx;
            vector ty(iny.rows() + tmpy.rows());
            ty << iny, tmpy;
            inx = tx;
            iny = ty;
        }
        n += tmpx.rows();
    }
    x = inx;
    y = iny;
    std::cout<< "LOADED MATRIX " << n <<" * "<< p << std::endl;
}

}// namespace dataset