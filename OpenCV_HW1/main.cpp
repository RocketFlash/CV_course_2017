//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//void showResults(vector<bool> computedAnswers, vector<bool> trueAnswers);
//
//int main( int argc, char** argv )
//{
//    //Path to the folder with test images
//    String imageName( "/Users/YagfarovRauf/Desktop/Computer vision/Images/BabyFood/" );
//
//
////    int correctAnswers[18] = {1,1,2,1,1,2,0,1,0,1,2,1,1,2,1,0,0,1};
//    vector<bool> correctAnswers = {true,true,false,true,true,false,false,true,false,true,false,true,true,false,true,false,false,true};
//    vector<bool> computedAnswers;
//
//    int answer = 0;
//    unsigned long numIm = 18;
//    computedAnswers.reserve(numIm);
//
//    //True positive, true negative, false positive and false negative
//    int TP = 0;
//    int TN = 0;
//    int FP = 0;
//    int FN = 0;
//
//    //Select region of interest
//    Rect myROI(135, 40, 340, 330);
//
//    //Process all images in the loop
//    for(int i = 1; i <=numIm ; i++) {
//
//        Mat imageIn, imageOut,mask1,mask2;
//
//        imageIn = imread(imageName+"BabyFood-Test"+to_string(i)+".JPG", IMREAD_COLOR);
//        //Crop ROI from image
//        imageOut = imageIn(myROI);
//        //Convert image to HSV
//        cvtColor(imageOut,imageOut,CV_BGR2HSV);
//
//        //Select pixels on image in certain range
//        inRange(imageOut, Scalar(0, 145, 100), Scalar(10, 255, 255), mask1);
//        inRange(imageOut, Scalar(170, 145, 100), Scalar(180, 255, 255), mask2);
//
//        //Combine two masks with red pixels
//        Mat imageOut1 = mask1 | mask2;
//
//        Mat imageOut2;
//        imageOut.copyTo(imageOut2,imageOut1);
//        cvtColor(imageOut2,imageOut2,CV_HSV2BGR);
//        if (imageOut.empty()) {
//            cout << "Could not open or find the image" << std::endl;
//            return -1;
//        }
//
//        // Count number of red pixels using range filtration from previous step
//        int numberOfRed = countNonZero(imageOut1);
//        cout<<"Test picture #"<<i<<endl;
//        cout<<"Number of nonzero pixels: "<<numberOfRed<<endl;
//
//        if(numberOfRed<1000){
//            answer = 0;
//        }else if(numberOfRed<15000){
//            answer = 1;
//        }else{
//            answer = 2;
//        }
//        computedAnswers.push_back(answer == 1);
//
//        cout<<"Algorithm answer: "<<answer<<" Correct answer: "<<correctAnswers[i-1]<<endl;
//        namedWindow("Display window", WINDOW_AUTOSIZE);
//        imshow("Display window", imageOut2);
////        waitKey(0);
//
//    }
//    showResults(computedAnswers,correctAnswers);
//    return 0;
//}
//
//void showResults(vector<bool> computedAnswers, vector<bool> trueAnswers){
//
//    //True positive, true negative, false positive and false negative
//    int TP = 0;
//    int TN = 0;
//    int FP = 0;
//    int FN = 0;
//    unsigned long computedAnswersSize = computedAnswers.size();
//    unsigned long trueAnswersSize = trueAnswers.size();
//
//    cout<<"Computed answers size: " << computedAnswersSize<<endl;
//    cout<<"Real answers size: " << trueAnswersSize<<endl;
//    if(computedAnswersSize != trueAnswersSize){
//        cout<<"Computed answers vector has not the same size as correct answers vector";
//        return;
//    }
//
//    for(int i = 0; i < computedAnswersSize; i++) {
//        if(computedAnswers[i] == true && trueAnswers[i] == true) {
//            TP++;
//        }
//        else if(computedAnswers[i] == false && trueAnswers[i] == false) {
//            TN++;
//        }
//        else if(computedAnswers[i] == true && trueAnswers[i] == false) {
//            FP++;
//        }
//        else if(computedAnswers[i] == false && trueAnswers[i] == true) {
//            FN++;
//        }
//    }
//
//    cout<<"Accuracy: " << (double(TP+TN)/double(TP+TN+FP+FN))<<endl;
//    cout<<"Precision: "<< (double(TP)/double(TP+FP))<<endl;
//    cout<<"Recall: "   << (double(TP)/double(TP+FN))<<endl;
//    cout<<"Specificity: "<<(double(TN)/double(TN+FP))<<endl;
//    cout<<"F1 score: "<< (double(2*TP)/double(2*TP+FP+FN))<<endl;
//}

#include <stdio.h>
#include <string.h>

#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char **argv) {

    cv::VideoCapture zed_cap(0);
//    zed_cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
//    zed_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    cv::Mat sbs,sbsN;
    Size size(1280,360);
    std::cout << zed_cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" << zed_cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    while (zed_cap.grab()) {
        zed_cap >> sbs;
        resize(sbs,sbsN,size);
        cv::imshow("SBS ZED", sbsN);
        cv::waitKey(20);
    }

    return 0;
}