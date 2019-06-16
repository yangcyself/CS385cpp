CC=g++
CFLAGS=-I.

CXX_FLAGS = -std=c++11  
CXX_DEFINES = 
CXX_INCLUDES = -pthread -I/usr/local/include -isystem /usr/local/include/opencv4 

OPENCV_LIBS = -rdynamic /usr/local/lib/libopencv_dnn.so.4.0.1 /usr/local/lib/libopencv_ml.so.4.0.1 /usr/local/lib/libopencv_photo.so.4.0.1 /usr/local/lib/libopencv_gapi.so.4.0.1 /usr/local/lib/libopencv_stitching.so.4.0.1 /usr/local/lib/libopencv_video.so.4.0.1 /usr/local/lib/libopencv_objdetect.so.4.0.1 /usr/local/lib/libopencv_calib3d.so.4.0.1 /usr/local/lib/libopencv_features2d.so.4.0.1 /usr/local/lib/libopencv_flann.so.4.0.1 /usr/local/lib/libopencv_highgui.so.4.0.1 /usr/local/lib/libopencv_videoio.so.4.0.1 /usr/local/lib/libopencv_imgcodecs.so.4.0.1 /usr/local/lib/libopencv_imgproc.so.4.0.1 /usr/local/lib/libopencv_core.so.4.0.1 -Wl,-rpath,/usr/local/lib 

all: processImage hogVisualize # writer #hogCalculate

processImage: 
	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS)  -o processImage processImage.cpp $(OPENCV_LIBS)

# processImage.cpp.o: processImage.cpp
# 	$(CC)   $(CXX_FLAGS)   build/processImage.cpp.o  -o processImage $(OPENCV_LIBS)


# writer: writer.cc.o
# 	$(CC) $(CXX_FLAGS) build/writer.cc.o -o protobuf/writer $(pkg-config --libs protobuf)
# writer.cc.o: protobuf/writer.cc protobuf/dataset.hog.pb.cc  protobuf/dataset.hog.pb.h
# 	$(CC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o build/writer.cc.o   protobuf/writer.cc protobuf/dataset.hog.pb.cc


hogVisualize: #hogVisualize.cpp.o
	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS)  -o hogVisualize  hogVisualize.cpp $(OPENCV_LIBS)

# hogVisualize.cpp.o: hogVisualize.cpp

# 	$(CC)   $(CXX_FLAGS)   build/hogVisualize.cpp.o  -o hogVisualize $(OPENCV_LIBS)


# hogCalculate: hogCalculate.cpp.o
# 	$(CC)   $(CXX_FLAGS)  build/hogCalculate.cpp.o  -o hogCalculate $(OPENCV_LIBS)

# hogCalculate.cpp.o: hogCalculate.cpp
# 	$(CC)   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o build/hogCalculate.cpp.o -c hogCalculate.cpp
