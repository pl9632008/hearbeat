//
//  RPPG.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 7/07/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef RPPG_hpp
#define RPPG_hpp

#include <deque>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <ctime>
#include <stdio.h>
#include <chrono>

using namespace cv;
using namespace dnn;
using namespace std;


enum faceDetAlgorithm { haar, deep };

class RPPG {

public:

    // Constructor
    RPPG() {;}

    // Load Settings
    bool load(const faceDetAlgorithm faceDetAlg,
              const string &haarPath,
              const string &dnnProtoPath,
              const string &dnnModelPath,

            std::chrono::system_clock::time_point sstart);

    void processFrame(Mat &frameRGB,Mat &frameGray,int time);



private:

    void detectFace(Mat &frameRGB, Mat &frameGray);
    void setNearestBox(vector<Rect> boxes);
    void updateMask(Mat &frameGray);
    void updateROI();
    void invalidateFace();


    void setBloodPressure(double heartRate, int age, string sex, int weight, int height, string position);

    double calculateO2(double r, double nearInfrared);
    double o2_level(double hbO2, double hb);
    double configure(double red, double nir);



    // The classifier
    faceDetAlgorithm faceDetAlg;
    CascadeClassifier haarClassifier;
    Net dnnClassifier;

    // Settings
    Size minFaceSize;


    // State variables

    bool faceValid =false;


    // Mask
    Rect box;
    Mat1b mask;
    Rect roi;


    int buffer_size = 200;
    double fps_TEST=30;
    double bpm_TEST=0;
    Mat1d data_buffer;
    Mat1d data_buffer2;
    Mat1d data_buffer3;


    deque<double> times;

    deque<double>bpms_TEST;


    double hr=60.0;
    double hr_prev=60.0;
    double bpm_disp;
    bool decide_freq_flag =false;
    double decide_freq ;

   bool enough_info = true;


   double rr=0;
   double rr_prev=0;
   double rr_disp;
   deque<double>rrs_TEST;


    double  hrv_prev=0 ;

    double hrv =0;
    double hrv_result = 0;


    std::chrono::system_clock::time_point sstart ;



    std::vector<double> interp(const std::vector<double> &inputs, const std::deque<double> &times);
    Mat1d interp(const Mat1d &data_buffer, const std::deque<double> &times);
    std::vector<double> hamming = {
            0.08, 0.08022927, 0.08091685, 0.08206205, 0.08366374,
            0.08572031, 0.08822971, 0.09118945, 0.09459658, 0.09844769,
            0.10273895, 0.10746609, 0.11262438, 0.11820869, 0.12421345,
            0.13063268, 0.13745997, 0.14468852, 0.15231113, 0.16032019,
            0.16870773, 0.17746538, 0.18658441, 0.19605574, 0.20586991,
            0.21601716, 0.22648736, 0.23727007, 0.24835455, 0.25972975,
            0.27138433, 0.28330667, 0.29548489, 0.30790684, 0.32056016,
            0.33343221, 0.34651017, 0.35978102, 0.3732315 , 0.38684823,
            0.40061762, 0.41452595, 0.42855935, 0.44270384, 0.45694532,
            0.47126959, 0.48566237, 0.50010932, 0.51459603, 0.52910806,
            0.54363095, 0.55815022, 0.5726514 , 0.58712003, 0.60154169,
            0.615902  , 0.63018666, 0.64438141, 0.65847211, 0.67244472,
            0.68628531, 0.69998007, 0.71351536, 0.72687769, 0.74005374,
            0.75303036, 0.76579464, 0.77833383, 0.79063545, 0.80268724,
            0.81447716, 0.82599349, 0.83722473, 0.84815969, 0.85878747,
            0.86909747, 0.87907943, 0.88872338, 0.89801971, 0.90695917,
            0.91553283, 0.92373215, 0.93154896, 0.93897547, 0.94600427,
            0.95262835, 0.95884112, 0.96463638, 0.97000835, 0.97495168,
            0.97946144, 0.98353313, 0.9871627 , 0.99034653, 0.99308145,
            0.99536472, 0.99719408, 0.99856769, 0.9994842 , 0.99994268,
            0.99994268, 0.9994842 , 0.99856769, 0.99719408, 0.99536472,
            0.99308145, 0.99034653, 0.9871627 , 0.98353313, 0.97946144,
            0.97495168, 0.97000835, 0.96463638, 0.95884112, 0.95262835,
            0.94600427, 0.93897547, 0.93154896, 0.92373215, 0.91553283,
            0.90695917, 0.89801971, 0.88872338, 0.87907943, 0.86909747,
            0.85878747, 0.84815969, 0.83722473, 0.82599349, 0.81447716,
            0.80268724, 0.79063545, 0.77833383, 0.76579464, 0.75303036,
            0.74005374, 0.72687769, 0.71351536, 0.69998007, 0.68628531,
            0.67244472, 0.65847211, 0.64438141, 0.63018666, 0.615902  ,
            0.60154169, 0.58712003, 0.5726514 , 0.55815022, 0.54363095,
            0.52910806, 0.51459603, 0.50010932, 0.48566237, 0.47126959,
            0.45694532, 0.44270384, 0.42855935, 0.41452595, 0.40061762,
            0.38684823, 0.3732315 , 0.35978102, 0.34651017, 0.33343221,
            0.32056016, 0.30790684, 0.29548489, 0.28330667, 0.27138433,
            0.25972975, 0.24835455, 0.23727007, 0.22648736, 0.21601716,
            0.20586991, 0.19605574, 0.18658441, 0.17746538, 0.16870773,
            0.16032019, 0.15231113, 0.14468852, 0.13745997, 0.13063268,
            0.12421345, 0.11820869, 0.11262438, 0.10746609, 0.10273895,
            0.09844769, 0.09459658, 0.09118945, 0.08822971, 0.08572031,
            0.08366374, 0.08206205, 0.08091685, 0.08022927, 0.08
    };

};


#endif /* RPPG_hpp */
