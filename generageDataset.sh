bash
for I in {1..4}
do 
    ./processImage ../FDDB-folds/FDDB-fold-0$I-ellipseList.txt ./out/train/ 0${I}_ 1
done

for I in {5..8}
do 
    ./processImage ../FDDB-folds/FDDB-fold-0$I-ellipseList.txt ./out/train/ 0${I}_ 0
done

./processImage ../FDDB-folds/FDDB-fold-09-ellipseList.txt ./out/test/ 09_ 0
./processImage ../FDDB-folds/FDDB-fold-10-ellipseList.txt ./out/test/ 10_ 1

