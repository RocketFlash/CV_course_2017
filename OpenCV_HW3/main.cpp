#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <iostream>

using namespace cv;
using namespace std;

/**
 * Helper functions
 */
Mat createMultiImage(vector<Mat> images);
void morphOps(Mat &thresh, int er, int di);
double distance(Point2f p1, Point2f p2);
bool compareContourAreas( vector<Point> contour1, vector<Point> contour2 );
void setLabel(Mat& im, const string label, const Point pt);
void showResults(vector<vector<int>> computedAnswers, vector<vector<int>> trueAnswers);
double abandonedOrRemoved(Mat imgA, Mat imgB, Mat mask);


int main(int argc, char* argv[]) {

    string videoDirectoryPath = string("/Users/YagfarovRauf/ClionProjects/OpenCV_HW3/Videos/");
    string fileName = string("ObjectAbandonmentAndRemoval");


    // Determine ground truth values
    int videoLengths[] = {717, 692};
    int momentsAbandoned[] = {183, 215};
    int momentsRemoved[] = {509, 551};

    vector<vector<int>> trueValues(2);
    vector<vector<int>> calculatedValues(2);


    // Fill a vector with true values and calculated by algorithm
    // 0 - No object on image
    // 1 - Abandoned object on image
    // 2 - Removed object on image
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < momentsAbandoned[j]; i++) {
            trueValues[j].push_back(0);
            calculatedValues[j].push_back(0);
        }
        for (int i = momentsAbandoned[j]; i < momentsRemoved[j]; i++) {
            trueValues[j].push_back(1);
            calculatedValues[j].push_back(0);
        }
        for (int i = momentsRemoved[j]; i < videoLengths[j]; i++) {
            trueValues[j].push_back(0);
            calculatedValues[j].push_back(0);
        }
    }

    // Read all the files in the loop
    for (int fileNo = 1; fileNo < 3; fileNo++) {
        // Read videofile
        string file1 = videoDirectoryPath + fileName + to_string(fileNo) + ".avi";

        VideoCapture capture;
        capture.open(file1);


        if (!capture.isOpened()) {
            cerr << "Unable to open video file: " << file1 << endl;
            exit(EXIT_FAILURE);
        }

        Mat frame, back1, back2, back3, fore1, fore2, fore3;
        //Create two background substractors: one fast, another slow
        Ptr<BackgroundSubtractor> bg1 = createBackgroundSubtractorMOG2(600, 5000, false);
        Ptr<BackgroundSubtractor> bg2 = createBackgroundSubtractorMOG2(60, 1000, false);
        //Third background substractor created for convenience (it is possible to use one of above),
        // to select proper parameters
        Ptr<BackgroundSubtractor> bg3 = createBackgroundSubtractorMOG2(500, 200, false);

        namedWindow("Frame");
        namedWindow("Static");
        namedWindow("Contours", CV_WINDOW_AUTOSIZE);

        Mat difference, differenceGray, resultImage;
        Mat frameGray;
        vector<Point2f> pmc; // Previous mass centers

        // Abandoned/Removed object detection params
        double eps = 1;
        int thresholdWarning = 10;
        int thresholdAlert = 15;
        unsigned long maxNumberOfContours = 2;

        // This parameter is a boarder of algorithm suggestion
        // If number of similarity more or equals to this value - it's removed object
        // Else if number similarity less than this value - it's abandoned object
        double abandonedBoarder = 0.55;

        int staticCounter[maxNumberOfContours];
        for (int i = 0; i < maxNumberOfContours; i++) {
            staticCounter[i] = 0;
        }

        Mat total;
        int frameNo = 1;
        string frNo = "Frame: ";
        vector<int> abandonedAt;
        vector<int> removedAt;

        // Iterate through videoframes
        for (int y = 0; y < videoLengths[fileNo - 1]; y++) {

            vector<Mat> images;
            capture >> frame;
            GaussianBlur(frame, frame, Size(3, 3), 3.5, 3.5);

            bg1->apply(frame, fore1);
            bg1->getBackgroundImage(back1);
            bg2->apply(frame, fore2);
            bg2->getBackgroundImage(back2);
            bg3->apply(frame, fore3);
            bg3->getBackgroundImage(back3);

            morphOps(fore1, 2, 3);
            morphOps(fore2, 2, 3);

            subtract(back1, back2, difference);

            cvtColor(difference, differenceGray, CV_BGR2GRAY);
            threshold(differenceGray, resultImage, 15, 255, THRESH_BINARY);

            // Mask for the abandoned/removed detector
            Mat totti;
            resultImage.copyTo(totti);

            morphOps(resultImage, 2, 3);
            total = resultImage;

            vector<vector<Point> > v;
            vector<vector<Point> > contours;
            // Find all contours on binary image
            findContours(total, v, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            // Sort contours by area to select only biggest ones
            sort(v.begin(), v.end(), compareContourAreas);
            if (v.size() >= maxNumberOfContours) {
                vector<vector<Point> > biggestContours(maxNumberOfContours);
                for (int i = 0; i < maxNumberOfContours; i++) {
                    biggestContours[i] = v[v.size() - 1 - i];
                }
                contours = biggestContours;
            } else {
                vector<vector<Point> > biggestContours(v.size());
                for (int i = 0; i < v.size(); i++) {
                    biggestContours[i] = v[v.size() - 1 - i];
                }
                contours = biggestContours;
            }


            vector<Moments> mu(contours.size()); // Moments
            vector<Point2f> mc(contours.size()); // Mass centers
            double dist = 0;

            // Iterate through biggest contours
            for (int i = 0; i < contours.size(); i++) {

                //Draw a bounding box
                Rect rect = boundingRect(contours[i]);
                Point pt1, pt2;
                pt1.x = rect.x;
                pt1.y = rect.y;
                pt2.x = rect.x + rect.width;
                pt2.y = rect.y + rect.height;
                Scalar color;

                //Find centers of the contours
                mu[i] = moments(contours[i], false);
                mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);


                double minVal = 1000;
                for (int j = 0; j < pmc.size(); j++) {
                    dist = distance(mc[i], pmc[j]);
                    if (dist < minVal) {
                        minVal = dist;
                    }
                }
                if (minVal < eps) {
                    staticCounter[i]++;
                } else {
                    staticCounter[i] = 0;
                }

                // Change bounding box color if it is static
                // Yellow -if it is potentially could be static
                // Red - if object is static
                if (staticCounter[i] > thresholdWarning && staticCounter[i] < thresholdAlert) {
                    color = CV_RGB(255, 255, 0);

                    rectangle(frame, pt1, pt2, color, 1);
                } else if (staticCounter[i] >= thresholdAlert) {
                    color = CV_RGB(255, 0, 0);
                    Rect myROI(pt1, pt2);
                    vector<Mat> immg;
                    Mat im = frame(myROI);
                    Mat imObj = back3(myROI);
                    Mat newtot = totti(myROI);
                    Mat newim;
                    Mat newimObj;

                    im.copyTo(newim, newtot);
                    imObj.copyTo(newimObj, 255 - newtot);

                    immg.push_back(newim);
                    immg.push_back(newimObj);
                    Mat yy = createMultiImage(immg);
                    imshow("Static", yy);

                    // Check histograms similarities
                    double similarity = abandonedOrRemoved(newim, newimObj, newtot);
                    cout << "similarity: " << similarity << endl;
                    if (similarity < abandonedBoarder) {
                        setLabel(frame, "Abandoned", Point(pt1.x, pt1.y - 2));
                        abandonedAt.push_back(frameNo);

                    } else {
                        setLabel(frame, "Removed", Point(pt1.x, pt1.y - 2));
                        removedAt.push_back(frameNo);
                    }
                    rectangle(frame, pt1, pt2, color, 1);
                    break;
                }
            }


            cvtColor(frame, frameGray, CV_BGR2GRAY);
            // Previous mass centers
            pmc = mc;

            // Put information about the current frame number
            putText(frame, frNo + to_string(frameNo), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
            frameNo++;

            imshow("Frame", total);
            imshow("Contours", frame);

            waitKey(10);
        }


        for (int k = 0; k < abandonedAt.size(); k++) {
            calculatedValues[fileNo-1][abandonedAt[k]-1] = 1;
        }
        for (int k = 0; k < removedAt.size(); k++) {
            calculatedValues[fileNo-1][removedAt[k]-1] = 2;
        }

        //delete capture object
        capture.release();
        //destroy GUI windows
        destroyAllWindows();
    }

    // Calculate statistics (Performance measurement)
    showResults(trueValues, calculatedValues);

    return EXIT_SUCCESS;
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

/**
 * Helper function to make morphology operations
 */
void morphOps(Mat &thresh, int er, int di){

    //create structuring element that will be used to "dilate" and "erode" image.
    Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
    //dilate with larger element so make sure object is nicely visible
    Mat dilateElement = getStructuringElement( MORPH_RECT,Size(4,4));

    erode(thresh,thresh,erodeElement,Point(-1,1),er);
    dilate(thresh,thresh,dilateElement,Point(-1,1),di);
}

/**
 * Function to calculate distance between two points
 */
double distance(Point2f p1, Point2f p2){
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
}

/**
 * Function to compare two contours areas
 */
bool compareContourAreas( vector<Point> contour1, vector<Point> contour2 ) {
    double i = fabs(contourArea(Mat(contour1)) );
    double j = fabs(contourArea(Mat(contour2)) );
    return ( i < j );
}

/**
 * This function shows precision metrics
 */
void showResults(vector<vector<int>> computedAnswers, vector<vector<int>> trueAnswers) {

    unsigned long computedAnswersSize = computedAnswers[0].size();
    unsigned long trueAnswersSize = trueAnswers[0].size();

    vector<vector<bool>> trueStatic(2);
    vector<vector<bool>> computedStatic(2);

    int framerate = 25;


    cout << "Computed answers size: " << computedAnswersSize << endl;
    cout << "Real answers size: " << trueAnswersSize << endl;
    if (computedAnswersSize != trueAnswersSize) {
        cout << "Computed answers vector has not the same size as correct answers vector";
        return;
    }

    for(int j=0;j<2;j++){
        //True positive, true negative, false positive and false negative
        int TP = 0;
        int TN = 0;
        int FP = 0;
        int FN = 0;

        // True positive, true negative, false positive and false negative
        // for anandoned/removed statistics
        int xTP = 0;
        int xFP = 0;
        int xTN = 0;
        int xFN = 0;

        double timeAbandonedComputed = 0;
        double timeAbandonedTrue = 0;
        double timeRemovedComputed = 0;
        double timeRemovedTrue = 0;

        for(int i =0;i < computedAnswers[j].size();i++){
            if(computedAnswers[j][i] == 1){
                timeAbandonedComputed+=1;
            }else if(computedAnswers[j][i]==2){
                timeRemovedComputed+=1;
            }

            if(trueAnswers[j][i] == 1){
                timeAbandonedTrue+=1;
            }else if(trueAnswers[j][i]==2){
                timeRemovedTrue+=1;
            }
            computedStatic[j].push_back(computedAnswers[j][i]!=0);
            trueStatic[j].push_back(trueAnswers[j][i]!=0);

            if(computedAnswers[j][i] == trueAnswers[j][i]){
                xTP ++;
            }else if(computedAnswers[j][i]==0 & trueAnswers[j][i]!=0){
                xFP ++;
            }else if(computedAnswers[j][i] != trueAnswers[j][i]){
                xFN ++;
            }
        }

        for (int i = 0; i < trueStatic[j].size(); i++) {
            if (computedStatic[j][i] == true && trueStatic[j][i] == true) {
                TP++;
            } else if (computedStatic[j][i] == false && trueStatic[j][i] == false) {
                TN++;
            } else if (computedStatic[j][i] == true && trueStatic[j][i] == false) {
                FP++;
            } else if (computedStatic[j][i] == false && trueStatic[j][i] == true) {
                FN++;
            }
        }

        cout<<"---------------------------------------------------------------"<<endl;
        cout<<"---------------------------Video #"<<j+1<<"----------------------------"<<endl;
        cout<<"---------------------------------------------------------------"<<endl;
        cout << "TP: " << TP <<endl;
        cout << "TN: " << TN <<endl;
        cout << "FP: " << FP <<endl;
        cout << "FN: " << FN <<endl;
        cout<<"---------------------------------------------------------------"<<endl;
        cout << "Precision  : " << (double(TP) / double(TP + FP)) << endl;
        cout << "Recall     : " << (double(TP) / double(TP + FN)) << endl;
        cout << "DSC score   : " << (double(2 * TP) / double(2 * TP + FP + FN)) <<endl;
        cout<<"--------------Abandoned or removal statistics-------------------"<<endl;
        cout << "xTP: " << xTP <<endl;
        cout << "xFP: " << xFP <<endl;
        cout << "xFN: " << xFN <<endl;
        cout << "Precision  : " << (double(xTP) / double(xTP + xFP)) << endl;
        cout << "Recall     : " << (double(xTP) / double(xTP + xFN)) << endl;
        cout<<"---------------------------------------------------------------"<<endl;
        cout << "Time removed true: " << timeRemovedTrue/double(framerate) <<endl;
        cout << "Time removed computed: " << timeRemovedComputed/double(framerate) <<endl;
        cout << "Time abandoned true: " << timeAbandonedTrue/double(framerate) <<endl;
        cout << "Time abandoned computed: " << timeAbandonedComputed/double(framerate) <<endl;

    }

}

/**
 * Helper function to write text in frame
 */
void setLabel(Mat& im, const string label, const Point pt) {
    int fontface = FONT_HERSHEY_SIMPLEX;
    double scale = 0.3;
    int thickness = 1;
    int baseline = 0;

    Size text = getTextSize(label, fontface, scale, thickness, &baseline);
    rectangle(im, pt + Point(0, baseline), pt + Point(text.width, -text.height), CV_RGB(255,0,0), CV_FILLED);
    putText(im, label, pt, fontface, scale, CV_RGB(255,255,255), thickness, 8);
}

/**
 * Function to calculate similarity between two images using histogram comparison
 * @param imgA
 * @param imgB
 * @param mask
 * @return
 */
double abandonedOrRemoved(Mat imgA, Mat imgB, Mat mask){

    int hbins = 30, sbins = 32;
    int channels[] = {0,  1};
    int histSize[] = {hbins, sbins};
    float hranges[] = { 0, 180 };
    float sranges[] = { 0, 255 };
    const float* ranges[] = { hranges, sranges};

    Mat patch_HSV;
    MatND HistA, HistB;

    //cal histogram & normalization
    cvtColor(imgA, patch_HSV, CV_BGR2HSV);
    calcHist( &patch_HSV, 1, channels,  mask, //MaskForHistogram
              HistA, 2, histSize, ranges,
              true, // the histogram is uniform
              false );
    normalize(HistA, HistA,  0, 255, CV_MINMAX);


    cvtColor(imgB, patch_HSV, CV_BGR2HSV);
    calcHist( &patch_HSV, 1, channels,  255 - mask, //MaskForHistogram
              HistB, 2, histSize, ranges,
              true, // the histogram is uniform
              false );
    normalize(HistB, HistB, 0, 255, CV_MINMAX);


    double bc = compareHist(HistA, HistB, CV_COMP_CORREL);

    return bc;
}