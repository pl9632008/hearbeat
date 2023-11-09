//
//  Heartbeat.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 4/06/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "Heartbeat.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv.hpp"

#define DEFAULT_FACEDET_ALGORITHM "deep"
#define HAAR_CLASSIFIER_PATH "./haarcascade_frontalface_alt.xml"
#define DNN_PROTO_PATH "./opencv/deploy.prototxt"
#define DNN_MODEL_PATH "./opencv/res10_300x300_ssd_iter_140000.caffemodel"

using namespace cv;

Heartbeat::Heartbeat(int argc_, char * argv_[], bool switches_on_) {

    argc = argc_;
    argv.resize(argc);
    copy(argv_, argv_ + argc, argv.begin());
    switches_on = switches_on_;

    // map the switches to the actual
    // arguments if necessary
    if (switches_on) {

        vector<string>::iterator it1, it2;
        it1 = argv.begin();
        it2 = it1 + 1;

        while (true) {

            if (it1 == argv.end()) break;
            if (it2 == argv.end()) break;

            if ((*it1)[0] == '-')
                switch_map[*it1] = *(it2);

            it1++;
            it2++;
        }
    }
}

string Heartbeat::get_arg(int i) {

    if (i >= 0 && i < argc)
        return argv[i];

    return "";
}

string Heartbeat::get_arg(string s) {

    if (!switches_on) return "";

    if (switch_map.find(s) != switch_map.end())
        return switch_map[s];

    return "";
}

bool to_bool(string s) {
    bool result;
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    istringstream is(s);
    is >> boolalpha >> result;
    return result;
}




faceDetAlgorithm to_faceDetAlgorithm(string s) {
    faceDetAlgorithm result;
    if (s == "haar") result = haar;
    else if (s == "deep") result = deep;
    else {
        std::cout << "Please specify valid face detection algorithm (haar, deep)!" << std::endl;
        exit(0);
    }
    return result;
}

int main(int argc, char * argv[]) {

    Heartbeat cmd_line(argc, argv, true);

    const string input = cmd_line.get_arg("-i"); // Filepath for offline mode


    faceDetAlgorithm faceDetAlg;
    string faceDetAlgString = cmd_line.get_arg("-facedet");

    if (faceDetAlgString != "") {
        faceDetAlg = to_faceDetAlgorithm(faceDetAlgString);
    } else {
        faceDetAlg = to_faceDetAlgorithm(DEFAULT_FACEDET_ALGORITHM);
    }

    cout << "Using face detection algorithm " << faceDetAlg << "." << endl;


    bool offlineMode = input != "";

    VideoCapture cap;
    if (offlineMode) cap.open(input);
    else cap.open(0);
    if (!cap.isOpened()) {
        return -1;
    }


    const int WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double FPS = cap.get(cv::CAP_PROP_FPS);
    const double TIME_BASE = 0.001;


    cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
    cout << "FPS: " << FPS << endl;
    cout << "TIME BASE: " << TIME_BASE << endl;


    auto sstart = std::chrono::system_clock::now();
    RPPG rppg = RPPG();
    rppg.load( faceDetAlg,
              HAAR_CLASSIFIER_PATH,
              DNN_PROTO_PATH, DNN_MODEL_PATH,
              sstart
              );

    cout << "START ALGORITHM" << endl;

    int i = 0;
    Mat frameRGB, frameGray;

//     VideoWriter shipin;
//
//     shipin.open("../moive/out3.avi",VideoWriter::fourcc('M','J','P','G'),FPS,Size(WIDTH,HEIGHT));


//    Mat aaa = Mat::ones(3,3,CV_64F);
//    cv::InputArray b(aaa);
//    b.getMat()


    while (true) {

        // Grab RGB frame
        cap.read(frameRGB);

        if (frameRGB.empty())
            break;

        // Generate grayframe
        cvtColor(frameRGB, frameGray, COLOR_BGR2GRAY);
        equalizeHist(frameGray, frameGray);

        rppg.processFrame(frameRGB, frameGray, 1);

        // if (gui) {
        //     imshow(window_title.str(), frameRGB);
        //     if (waitKey(30) >= 0) break;
        // }

        //         shipin.write(frameRGB);
        i++;
    }

    return 0;
}