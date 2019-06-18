/**
 * automatically crop the images and put them into target folder
 * usage:
 * ./processImage <ellipseList File> <output folder> <count start number> <put into neg>
 * e.g.
 * ./processImage ../FDDB-folds/FDDB-fold-09-ellipseList.txt ./out/test/ 09_ 0
 */
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <cstdio>

using namespace cv;

// const std::string OUTFOLDER = "./out/";

void saveimg(Mat in,int number = 0, std::string OUTFOLDER="./out/")
{
    // std::cout << OUTFOLDER+name<<std::endl;
    char num[100];
    std::string name;
    sprintf(num,"%d.jpg",number);
    name = num;
    imwrite(OUTFOLDER+name,in);
}

struct cropRes{
   Mat data[9]; 
};

Mat cropimg(Mat in,int xt,int yt ,int xb , int yb)
{
   int Borderw=0,Borders = 0,Bordera = 0,Borderd = 0; // USE WASD to represent top bottom right left
   Mat tmp;
   Bordera = max(0,-xt);
   Borderw = max(0,-yt);
   xt = max(0,xt);
   yt = max(0,yt);
   Borderd = max(0,xb-in.cols);
   Borders = max(0,yb-in.rows);
   xb+=Bordera;
   yb+=Borderw;
   copyMakeBorder( in, tmp, Borderw, Borders, Bordera, Borderd, BORDER_REPLICATE, 0 );
   // std::cout<<"figure:"<<tmp.rows<<" "<<tmp.cols<<std::endl;
   // std::cout<<xt<<" "<< yt<<" "<< xb<<" "<< yb<<" "<<std::endl;

   Mat out (tmp, Rect(xt, yt, xb-xt, yb-yt) );
   return out;
}

cropRes generateSample(Mat in,std::ifstream& fin)
{
    /*take in the input CV matrix 
    and read six numbers  
    and crop and return the image*/
    double xlen,ylen,d, xpos,ypos,a,xslide,yslide,xrsz,yrsz;
    Mat resized;
    cropRes res;
    int n = 0;
    fin>> ylen >> xlen >> d >> xpos >> ypos >> a;

    // std::cout<< xlen << ylen << d << xpos << ypos << a<< std::endl;
    xlen = xlen * 4/3;
    ylen = ylen * 4/3;
    xslide = 2*xlen/3;
    yslide = 2*ylen/3;

    //Calculate the factors after resize
    xrsz = 96/(2*xlen);
    yrsz = 96/(2*ylen);

    xpos = xpos * xrsz;
    ypos = ypos * yrsz;
    xlen = 48; //xlen * xrsz; // = 48
    ylen = 48; //ylen * yrsz; // = 48
    xslide = 32; //xslide * xrsz;
    yslide  = 32; //yslide * yrsz;
    // int top,bottom,left,right; 
    // top = (int) (0.05*in.rows); bottom = top;
    // left = (int) (0.05*in.cols); right = left;

   //  int xt= xpos - xlen, yt = ypos-ylen, xb = xpos+xlen, yb = ypos+ylen;
    
   
    resize(in,resized,Size(),xrsz,yrsz);
    for(int i = -1;i<2;i++){
       for (int j = -1;j<2;j++){
         res.data[n++] = cropimg(resized, xpos-xlen + i*xslide, ypos-ylen+j*yslide, 
                           xpos+xlen + i*xslide, ypos+ylen+j*yslide );
       }
    }
        
    return res;
    
}

int main( int argc, char** argv )
{
 char* foldName = argv[1];
 std::string outfolder;
 Mat image;
 Mat res_image;

 int n;
 std::string imagename;
 std::string imagepath;
 int num = 0;
 std::string fnum = "0_";
//  printf("hey\n");
 cropRes tmpres;
//  printf("yo\n");

//  image = imread( imageName, IMREAD_COLOR );
 
 
 std::ifstream fin (foldName);
 for(int i = 0;i<argc;i++){
    std::cout<<argv[i]<<" ";
 }
 std::cout<<std::endl;
 if( argc <4 || !fin )
 {
   printf( "no fold annotation data \n " );
   printf("input should be: ./processimage <input annotation data> <output folder> <startnum>");
   return -1;
 }
 if(argc >=3 )
   outfolder = argv[2];
 else 
   outfolder = "./out/";

 int put_into_neg = 1;

 if(argc >=4)
   fnum = argv[3];
 
 if(argc >=5 )
   put_into_neg = atoi(argv[4]);

 while(!fin.eof()){
     // read the picture
     fin>>imagename;
     if(fin.eof())
        break;
     imagepath = "../pics/"+imagename+".jpg";
     image = imread( imagepath, IMREAD_COLOR );
      //   std::cout<<"File "<<imagepath<<std::endl;
     if(!image.data){
        std::cout<<"no image data"<<std::endl;
        break;
     }
     fin>>n;
     for( int i = 0;i<n;i++){
        tmpres =  generateSample(image,fin);
      //   printf("%d\n",num);
        for (int j = 0;j<9;j++){
           res_image = tmpres.data[j];
           std::string pos_neg = (j==4)? "pos/" : "neg/";
           if(!(put_into_neg==1) && j!=4) //not positive image and don't have to put into neg
               continue;
           saveimg(res_image,num++,outfolder + pos_neg+fnum);
        }
        
     }
    // break;
 }
 fin.close();


//  cvtColor( image, gray_image, COLOR_BGR2GRAY );
//  imwrite( "./out/Gray_Image.jpg", gray_image );
//  saveimg(gray_image);
 printf("%d\n",num);
 return num;
}