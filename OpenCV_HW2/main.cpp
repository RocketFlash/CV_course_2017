#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void showResults(vector<bool> computedAnswers, vector<bool> trueAnswers);
Mat createMultiImage(vector<Mat> images);

int main(int argc, char **argv) {

    //Path to the folder with test images
    String imageName("/Users/YagfarovRauf/Desktop/Computer vision/Images/Glue/Glue");
    String iconName("/Users/YagfarovRauf/Desktop/Computer vision/Images/Glue/icon");

    Mat good = imread(iconName+to_string(2)+".png", IMREAD_COLOR);
    Mat bad =  imread(iconName+to_string(1)+".png", IMREAD_COLOR);
    cvtColor(good, good, COLOR_BGRA2BGR);
    cvtColor(bad, bad, COLOR_BGRA2BGR);

    bool answer;
    int boarder = 50;
    unsigned int numIm = 6;
    unsigned int numBottlesInOnePicture = 5;
    Mat imageIn, imageBlur;
    vector<Mat> edgedImages;
    vector<Mat> imagesWithLines;
    //Load images
    Mat* images = new Mat[numIm*5];


    namedWindow("Display lines", WINDOW_AUTOSIZE);
    namedWindow("Display edges", WINDOW_AUTOSIZE);

    //Read images and divide them on 5 subimages each
    for(int j = 0; j < numIm; j++){
        imageIn = imread(imageName+to_string(j+1)+".jpg", IMREAD_COLOR);

        int imW = imageIn.size().width;
        int imH = imageIn.size().height;
        int xStep = imW/numBottlesInOnePicture;
        int yStep = imH/2;


        for(int k = 0; k < numBottlesInOnePicture; k++){
            Rect roi(xStep*k, yStep, xStep-5, yStep-15);
            images[j*numBottlesInOnePicture+k] = imageIn(roi);
        }
    }


    vector<bool> correctAnswers = {false, false, true, true, false,
                              false, false, false, true, false,
                              false, false, false, false, true,
                              true, false, false, false, true,
                              false, true, true, false, false,
                              false, false, false, false, false};
    vector<bool> computedAnswers;
    computedAnswers.reserve(numIm*numBottlesInOnePicture);

    Mat currentImage, gray, edges;
    vector<Vec2f> lines;
    Size size1(120,150);

    int numConrolPoits = 9;

    //Process all images in the loop
    for (int i = 1; i <= numIm*numBottlesInOnePicture; i++) {
        Mat resizedCurrentImage, resizedEdgesImage;
        currentImage = images[i-1];

        // Convert image in grayscale and applying median filter
        cvtColor(currentImage, gray, COLOR_BGR2GRAY);
        medianBlur(gray, gray, 5);

        // Canny edge detector
        Canny(gray, edges, 50, 110);

        int w = currentImage.size().width;
        int h = currentImage.size().height;
        int currentPosition = 0;
        int count = 0;
        int distances[numConrolPoits-3][2];
        bool stop = false;

        cout <<"Picture number "<<i<<" : "<< endl;
        for(int j= 2; j < numConrolPoits-1; j++){

            currentPosition = h/numConrolPoits * j + 6;
            Point pt1, pt2;
            pt1.x = 0;
            pt1.y = currentPosition;
            pt2.x = w;
            pt2.y = currentPosition;
            line(currentImage, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);

            //Read first and second pixels from left and right sides
            int forward [] = {-1, -1};
            int backward[] = {-1, -1};

            count = 0;
            for(int k = 0; k < w; k++) {
                if(bool(edges.at<uchar>(currentPosition, k))){
                    forward[count] = k;
                    count++;
                }
                if(count==2) break;
            }


            count = 0;
            for(int z = w-1; z >= 0; z--) {
                if(bool(edges.at<uchar>(currentPosition, z))){
                    backward[count] = z;
                    count++;
                }
                if(count==2) break;
            }

            if(backward[0]==-1||forward[0]==-1){
                stop = true;
                cout<<"There is no bottle on the current image!"<<endl;
                break;
            }else if(backward[0]==forward[1]&&backward[1]==forward[0]){
                distances[j-2][0] = 0;
                distances[j-2][1] = 0;
            }else{
                distances[j-2][0] = abs(forward [1] - forward[0]);
                distances[j-2][1] = abs(backward[1] - backward[0]);
            }
            cout<<distances[j-2][0]<<"   "<<distances[j-2][1]<<endl;

        }

        //Calculate penalty score
        double score = 0;
        int noLabelScore = 0;
        if(stop){
            answer = false;
        }else{
            for(int p = 0; p < (numConrolPoits-3); p++){
                if(distances[p][0]==0){
                    noLabelScore++;
                }else{
                    score += abs(distances[p][0] - distances[p][1]);
                }

            }
            double mult = 1;

            for(int p = 0; p < (numConrolPoits-3)-1; p++){
                if(distances[p][0]!=0&&distances[p+1][0]!=0){
                    if(abs(distances[p][0])>=abs(distances[p+1][0])){
                        mult*=(abs(double(distances[p][0]))/abs(double(distances[p+1][0])));
                    }else{
                        mult*=(abs(double(distances[p+1][0]))/abs(double(distances[p][0])));
                    }

                }
                score += abs(distances[p][0] - distances[p+1][0]);
                score += abs(distances[p][1] - distances[p+1][1]);
            }
            score *= mult;


            if(noLabelScore==(numConrolPoits-3)){
                cout<<"There is no label"<<endl;
                answer = false;
            }else{
                for(int p = 0; p < (numConrolPoits-3)-1; p++){
                    if(abs(distances[p][0] - distances[p+1][0]) >= 3 || abs(distances[p][1] - distances[p+1][1]) >= 3){
                        cout<<"I am here"<<endl;
                        answer = false;
                        stop = true;
                        break;
                    }
                }
                if(!stop){
                    answer = score < boarder;
                    cout<<"Score is: "<<score<<endl;
                }
            }
        }
        computedAnswers.push_back(answer);

        cout << "Algorithm answer: " << answer << " Correct answer: " << correctAnswers[i - 1] << endl;
        cout << "________________________________________" << endl;


        resize(edges,resizedEdgesImage,size1);
        resize(currentImage,resizedCurrentImage,size1);
        if(answer){
            good.copyTo(resizedCurrentImage(Rect(0,0,good.cols, good.rows)));
        }else{
            bad.copyTo(resizedCurrentImage(Rect(0,0,bad.cols, bad.rows)));
        }

        imagesWithLines.push_back(resizedCurrentImage);
        edgedImages.push_back(resizedEdgesImage);

    }

    showResults(computedAnswers, correctAnswers);

    imshow("Display edges", createMultiImage(edgedImages));
    imshow("Display lines", createMultiImage(imagesWithLines));
    waitKey(0);

    return 0;
}

/**
 * This function shows precision metrics
 * @param computedAnswers
 * @param trueAnswers
 */
void showResults(vector<bool> computedAnswers, vector<bool> trueAnswers) {

    //True positive, true negative, false positive and false negative
    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;
    unsigned long computedAnswersSize = computedAnswers.size();
    unsigned long trueAnswersSize = trueAnswers.size();

    cout << "Computed answers size: " << computedAnswersSize << endl;
    cout << "Real answers size: " << trueAnswersSize << endl;
    if (computedAnswersSize != trueAnswersSize) {
        cout << "Computed answers vector has not the same size as correct answers vector";
        return;
    }

    for (int i = 0; i < computedAnswersSize; i++) {
        if (computedAnswers[i] == true && trueAnswers[i] == true) {
            TP++;
        } else if (computedAnswers[i] == false && trueAnswers[i] == false) {
            TN++;
        } else if (computedAnswers[i] == true && trueAnswers[i] == false) {
            FP++;
        } else if (computedAnswers[i] == false && trueAnswers[i] == true) {
            FN++;
        }
    }

    cout<<"---------------------------------------------------------------"<<endl;
    cout << "TP: " << TP <<endl;
    cout << "TN: " << TN <<endl;
    cout << "FP: " << FP <<endl;
    cout << "FN: " << FN <<endl;
    cout<<"---------------------------------------------------------------"<<endl;
    cout << "Accuracy   : " << (double(TP + TN) /double(TP + TN + FP + FN))<<endl;
    cout << "Precision  : " << (double(TP) / double(TP + FP)) << endl;
    cout << "Recall     : " << (double(TP) / double(TP + FN)) << endl;
    cout << "Specificity: " << (double(TN) / double(TN + FP)) << endl;
    cout << "F1 score   : " << (double(2 * TP) / double(2 * TP + FP + FN)) <<endl;
    cout<<"---------------------------------------------------------------"<<endl;
}


/**
 * Helper function to display multiple images
 * in a one window
 */
Mat createMultiImage(vector<Mat> images) {
    if (images.size() == 0) {
        Mat nothing;
        return nothing;
    }
    long size = images.size();
    int cols = (int) ceil((double) sqrt(size));
    int rows = (int) ceil((double) size / cols);
    int imrows = images[0].rows;
    int imcols = images[0].cols;

    Mat doubleImage(imrows * rows, imcols * cols, images[0].type());
    int t = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            images[t].copyTo(doubleImage(Rect(j * imcols, i * imrows, imcols, imrows)));
            if (t == size - 1) {
                return doubleImage;
            }
            t++;
        }
    }

    return doubleImage;
}



