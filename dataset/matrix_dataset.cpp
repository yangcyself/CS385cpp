#include "dataset/matrix_dataset.h"
#include "protobuf/dataset.hog.pb.h"
#include "fstream"

#include <stdio.h>
#include <iostream>
#include <vector>

namespace dataset{

MatrixDataset::MatrixDataset(std::string protoPath, bool train)
{
    hog imagehogs; 
    { 
        std::fstream input(protoPath, std::ios::in | std::ios::binary); 
        if (!imagehogs.ParseFromIstream(&input)) { 
            std::cerr << "Failed to parse input file." << std::endl; 
        } 
    }

    for (int i = 0;i <imagehogs.image_size();i++){
        const ImageDescrip& tmp = imagehogs.image(i);

        cout<< "num: "<<i<<"\t";
        cout<<"imgPath: "<<tmp.imgpath()<<"\n";
        cout<<"Classtype: "<<tmp.classtype()<<"\t";
        cout<<"Datatype: "<<tmp.datatype()<<"\t";
        cout<<"img hog: ";
        for (int j = 0;j<tmp.imghog_size();j++){
            cout << tmp.imghog(j)<<" ";
        }
        cout<<endl;
    } 
    return 0;
}

}// namespace dataset