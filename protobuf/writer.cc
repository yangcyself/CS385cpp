/**
 * the writer file used to experiment the protocol buf functionalities
 * g++ writer.cc dataset.hog.pb.cc -o writer $(pkg-config --cflags --libs protobuf) -std=c++11
 */
#include "dataset.hog.pb.h"
#include "iostream"
#include <stdio.h>
// #include <iostream.h>
#include "fstream"
#include <vector>
using namespace std;

void addAnImage(dataset::ImageDescrip * dscrp)
{
  vector<float> descriptorsValues = {1,1,0,1,2,8,1,9,9,8,8,7,2,3};
  dscrp -> set_imgpath("/yangcy/CS385/codes/out/train/pos/01_22.jpg");
  dscrp -> set_classtype(dataset::ImageDescrip::POS);
  dscrp -> set_datatype(dataset::ImageDescrip::TRAIN);
  for (int i = 0;i< descriptorsValues.size();i++){
    dscrp -> add_imghog(descriptorsValues[i]);
  }
  // dscrp -> set_imghog(0, descriptorsValues[0]);
}

int main(void) 
 { 
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  dataset::hog imagehogs; 
  addAnImage(imagehogs.add_image());
  // Write the new address book back to disk. 
//   ofstream
  fstream output("./log", ios::out | ios::trunc | ios::binary); 

  if (!imagehogs.SerializeToOstream(&output)) { 
      cerr << "Failed to write msg." << endl; 
      return -1; 
  }         
  // Optional:  Delete all global objects allocated by libprotobuf.
  // google::protobuf::ShutdownProtobufLibrary();
  return 0; 
 }