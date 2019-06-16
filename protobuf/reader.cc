/**
 * the reader file used to experiment the protocol buf functionalities
 * g++ reader.cc dataset.hog.pb.cc -o reader $(pkg-config --cflags --libs protobuf) -std=c++11
 * 
 */
#include "dataset.hog.pb.h"
#include "iostream"
#include <stdio.h>
// #include <iostream.h>
#include "fstream"

using namespace std;


int main(int argc, char* argv[]) { 

    dataset::hog imagehogs; 

    { 
        fstream input("./log", ios::in | ios::binary); 
        if (!imagehogs.ParseFromIstream(&input)) { 
            cerr << "Failed to parse address book." << endl; 
            return -1; 
        } 
    } 

    for (int i = 0;i <imagehogs.image_size();i++){
        const dataset::ImageDescrip& tmp = imagehogs.image(i);
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