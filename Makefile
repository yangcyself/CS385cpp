CC=g++
CFLAGS=-I.

CXX_FLAGS = -std=c++11  
CXX_DEFINES = 
CXX_INCLUDES = -pthread -I/usr/local/include -isystem /usr/local/include/opencv4  -I/home/yangcy/programs/eigen -I.

OPENCV_LIBS = -rdynamic /usr/local/lib/libopencv_dnn.so.4.0.1 /usr/local/lib/libopencv_ml.so.4.0.1 /usr/local/lib/libopencv_photo.so.4.0.1 /usr/local/lib/libopencv_gapi.so.4.0.1 /usr/local/lib/libopencv_stitching.so.4.0.1 /usr/local/lib/libopencv_video.so.4.0.1 /usr/local/lib/libopencv_objdetect.so.4.0.1 /usr/local/lib/libopencv_calib3d.so.4.0.1 /usr/local/lib/libopencv_features2d.so.4.0.1 /usr/local/lib/libopencv_flann.so.4.0.1 /usr/local/lib/libopencv_highgui.so.4.0.1 /usr/local/lib/libopencv_videoio.so.4.0.1 /usr/local/lib/libopencv_imgcodecs.so.4.0.1 /usr/local/lib/libopencv_imgproc.so.4.0.1 /usr/local/lib/libopencv_core.so.4.0.1 -Wl,-rpath,/usr/local/lib 
PROTBF_LIBS = $(shell pkg-config --libs protobuf)


all: processImage hogVisualize  writer reader hogCalculate hogOutVisualize  test_dataset_main logisticMain fisherMain

processImage: processImage.cpp
	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS)  -o processImage processImage.cpp $(OPENCV_LIBS)


writer: protobuf/writer.cc protobuf/dataset.hog.pb.cc  protobuf/dataset.hog.pb.h
	$(CC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o protobuf/writer   protobuf/writer.cc protobuf/dataset.hog.pb.cc $(PROTBF_LIBS)

reader: protobuf/reader.cc protobuf/dataset.hog.pb.cc  protobuf/dataset.hog.pb.h
	$(CC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o protobuf/reader   protobuf/reader.cc protobuf/dataset.hog.pb.cc $(PROTBF_LIBS)

hogVisualize: hogVisualize.cpp
	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS)  -o hogVisualize  hogVisualize.cpp $(OPENCV_LIBS)


hogCalculate: hogCalculate.cpp protobuf/dataset.hog.pb.h protobuf/dataset.hog.pb.cc
	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o hogCalculate hogCalculate.cpp  protobuf/dataset.hog.pb.cc $(PROTBF_LIBS) $(OPENCV_LIBS)

hogOutVisualize: hogOutVisualize.cpp protobuf/dataset.hog.pb.h protobuf/dataset.hog.pb.cc
	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o hogOutVisualize hogOutVisualize.cpp  protobuf/dataset.hog.pb.cc $(PROTBF_LIBS) $(OPENCV_LIBS)


logisticMain: logisticMain.cpp
	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o logisticMain logisticMain.cpp logistic/logistic.cpp dataset/matrix_dataset.cpp protobuf/dataset.hog.pb.cc $(PROTBF_LIBS) $(OPENCV_LIBS)

fisherMain: fisherMain.cpp
	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o fisherMain fisherMain.cpp fisher/fisher.cpp dataset/matrix_dataset.cpp protobuf/dataset.hog.pb.cc $(PROTBF_LIBS) $(OPENCV_LIBS)

test_dataset_main: test/test_dataset_main.cpp dataset/matrix_dataset.cpp protobuf/dataset.hog.pb.cc dataset/matrix_dataset.h protobuf/dataset.hog.pb.h
	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) test/test_dataset_main.cpp dataset/matrix_dataset.cpp protobuf/dataset.hog.pb.cc -o test/test_dataset_main $(PROTBF_LIBS) 
