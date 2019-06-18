import sys
import numpy as np
import cv2
import os
import pickle as pkl 
import glob
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression

import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import backend as K
from PIL import ImageFile
from keras import optimizers

# model = "SVM"
# kerneltype = "linear"
# kerneltype = "sigmoid"
# kerneltype = "poly"
# kerneltype = "rbf"

# model = "CNN"

model = "LOGISTIC"

kerneltype = ""

def rectClustering(pred, scaledStride,scaled96):
    """
    the pred area may have overlayed areas, thus combine the overlayed area
    """
    rects = []
    predInd = -np.ones_like(pred) #the index of rects in the pred array
    predInd = predInd.astype(int)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if(not pred[i,j]):
                continue
            # print("SCALE:",scaled96,scaledStride)
            # print(scaledStride[0],scaled96[0])  
            bound = [0,0]
            # bound[0],bound[1] = (i*scaledStride[0] + scaled96[0] ), ( j*scaledStride[1] + scaled96[1])
            bound = ((i*scaledStride[0] + scaled96[0] ) , ( j*scaledStride[1] + scaled96[1])) # the bottom right side
            if(i-1 > 0 and pred[i-1,j]): # join to the upper rect
                predInd[i,j] = predInd[i-1,j]
                rects[predInd[i,j]][2] = max(rects[predInd[i,j]][2], bound[0] )
            elif(j - 1>0 and pred[i,j-1]): # join to the left rect
                predInd[i,j] = predInd[i-1,j]
                rects[predInd[i,j]][3] = max(rects[predInd[i,j]][3], bound[1] )
            else:
                predInd[i,j] = len(rects)
                rects.append ( [ i*scaledStride[0], j*scaledStride[1],
                     i*scaledStride[0] + scaled96[0], j*scaledStride[1] + scaled96[1]
                                        ])
    return rects

def singlescaleDetect(img, stride, clf, feature = "raw"):
    """
    Detecting single scale face of the scaled image
    img : input scaled image
    l : the height & width of the sliding window
    clf : the classifier used to predict the face, return 1 and 0

    """
    rects = [] # all the locations relative to the full image size
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inputfeats = []
    # print(img.shape[0]-96+1)
    # print((img.shape[0]-96+stride -1)/stride)
    # print(len([i for i in range(0, img.shape[0] - 96, stride)]))
    predshape = (int((img.shape[0] - 97+stride)/stride), int((img.shape[1]-97+stride)/stride))
    # print("IMG SHAPE: ",img.shape)
    # print("PRED SHAPE: ",predshape)
    for i in range(0, img.shape[0] - 96, stride):
        for j in range(0, img.shape[1] - 96, stride):
            cut = img[i:i+96, j:j+96,:]
            if feature =="hog":
                hog = cv2.HOGDescriptor("hog.xml")
                feats = hog.compute(img=cut, winStride=None, padding=None, locations=None).reshape(900,)
            else:
                feats = cut
            inputfeats.append(feats)
    if(len(inputfeats) == 0):
        return []
    inputfeats = np.array(inputfeats)
    # print(inputfeats.shape)
    pred = clf.predict(inputfeats)

    pred = pred.reshape(predshape)
    # print(pred)
    scaledStride = (stride / img.shape[0], stride/img.shape[1] )
    scaled96 = (96/img.shape[0], 96/img.shape[1])
    # for i in range(predshape[0]):
    #     for j in range(predshape[1]):
    #         if(pred[i][j]):
    #             rects.append((i*scaledStride[0], j*scaledStride[1],
    #                  i*scaledStride[0] + scaled96[0], j*scaledStride[1] + scaled96[1] ))

    return rectClustering(pred,scaledStride,scaled96)


def multiscaleDetection(img, stride, scales, clf,feature,imgnum = 0):
    """
    Detecting image scaled by the scales array
    img : input image
    l : the height & width of the sliding window
    sclaes : given scales to be resized
    clf : clf used to predict the face
    """
    for scale in scales:
        tmp = cv2.resize(src=img, dsize=(0, 0), fx=scale, fy=scale)
        rects = singlescaleDetect(tmp, stride, clf,feature)
        # print(rects)
        # print("LEN RECTS: ", len(rects))
        for a,b,c,d in rects:
            a,b,c,d = a * img.shape[0] , b * img.shape[1], c * img.shape[0], d * img.shape[1]
            img = cv2.rectangle(img, (int(b), int(a)), (int(d), int(c)), (0, 0, 255))
    # cv2.imshow("r",img)
    # cv2.waitKey(0)
    cv2.imwrite("out/{}{}_{}.jpg".format(kerneltype,model,imgnum), img)


class CNNfaceCLF:
    def __init__(self, *args, **kwargs):
        img_width, img_height = 96,96
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)
        print("INPUT SHAPE: ", input_shape)
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Flatten())
        # model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])
        
        model.load_weights('first_try.h5')
        self.model = model
    
    def predict(self, feature ):
        """
        feature is a batch of pictures: NCHW, 
            feature's type is numpy ndarray
            this makes it easier to do vectorization
        """
        # print(feature.shape)
        feature = K.variable(feature)
        res = self.model(feature)
        print(res)
        res = K.eval(res)
        
        # return None
        return (res>0.5).astype(int)



if __name__=='__main__':
    # exampleData = cv2.imread("../../2003/04/03/big/img_449.jpg")
    originalPics = glob.glob('../../*/*/*/*/*.jpg')
    selectedPics = np.random.choice(originalPics,20)
    ##### CNN MODELS #####
    if(model == "CNN"):
        clf = CNNfaceCLF()
        feature = "raw"
    ##### SVM MODELS #####
    if(model == "SVM"):
        with open("{}_svm.pkl".format(kerneltype), "rb") as f:
            clf = pkl.load(f)
        feature = "hog"
    ###### LOGISTIC ######
    if(model == "LOGISTIC" ):
        with open("logistic_clf.pkl", "rb") as f:
            clf = pkl.load(f)
        feature = "hog"
    for i, pic in enumerate(selectedPics):
        picdata = cv2.imread(pic)
        multiscaleDetection(picdata,48,[0.5,0.7,1.0,1.3],clf,feature,i)
    # multiscaleDetection(exampleData,48,[1.5,2.0,2.5],clf)
    # print(exampleData.shape)
    # exampleData =  exampleData.transpose(2,0,1)
    # exampleData =  exampleData.reshape(1,96,96,3)
    # print(cnnclf.predict(exampleData) )

    
