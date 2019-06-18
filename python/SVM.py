import sys
import os
from sklearn import svm
sys.path.append("../protobuf/dataset/")
import hog_pb2
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

hogfeatures = hog_pb2.hogdataset()

f = open("../out/hog.ptbf","rb")
hogfeatures.ParseFromString(f.read())
f.close()

datasets = { d:{ c:None for c in ["pos","neg"] } for d in ["test","train"]}
imagepaths = { d:{ c:None for c in ["pos","neg"] } for d in ["test","train"]}
for dat in hogfeatures.data:
    classtype = "pos" if dat.classtype else "neg"
    datatype = "test" if dat.datatype else "train"
    featureMatrix = []
    paths = []
    for img in dat.image:
        featureMatrix.append(img.imghog)
        paths.append(img.imgpath)
    featureMatrix = np.array(featureMatrix)
    print(featureMatrix.shape)
    datasets[datatype][classtype] = featureMatrix
    imagepaths[datatype][classtype] = paths


def datamatrixgen(classtype):
    X = np.concatenate((datasets[classtype]["pos"], datasets[classtype]["neg"]),axis = 0)
    posnum = datasets[classtype]["pos"].shape[0]
    negnum = datasets[classtype]["neg"].shape[0]
    y = np.concatenate((np.ones((posnum,)), np.zeros((negnum,)) ) ,axis = 0 )
    paths = imagepaths[classtype]["pos"] + imagepaths[classtype]["neg"]
    return X, y, paths
trainX,trainy,trainpaths = datamatrixgen("train")
testX,testy,testpaths = datamatrixgen("test")


def test_one_Modle(kerneltype):
    print("TESTING MODEL: ", kerneltype)
    clf = svm.SVC(gamma='auto',kernel=kerneltype)
    clf.fit(trainX, trainy)  
    acc = np.count_nonzero(clf.predict(testX) == testy) / testy.shape[0]
    with open("{}_svm.pkl".format(kerneltype),"wb") as f:
        pkl.dump(clf,f)
    print("ACC: " ,acc)
    print("clf_support number: ",len(clf.support_ ))
    # number_of_visual = 21
    # for i in np.random.choice( clf.support_, number_of_visual):
    #     p = trainpaths[i]
    #     # print(p)
    #     support = plt.imread(os.path.join("../", p))
    #     plt.imsave(os.path.join("../reportPics/","supportimg_{}_{}.jpg".format(kerneltype,i)),support)
    #     # plt.imshow(support )
    #     # plt.show()
    

for m in ["linear", "poly", "rbf", "sigmoid"]:
    test_one_Modle(m)