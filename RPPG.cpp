//
//  RPPG.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 7/07/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "RPPG.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/opencv.hpp>
#include "opencv.hpp"
#include <iostream>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <cstring>
#include <chrono>

using namespace cv;
using namespace dnn;
using namespace std;

bool RPPG::load(const faceDetAlgorithm faceDetAlg,
                const string &haarPath,
                const string &dnnProtoPath,
                const string &dnnModelPath,
               std::chrono::system_clock::time_point sstart) {

    this->faceDetAlg = faceDetAlg;
    this->sstart = sstart;

    // Load classifier
    switch (faceDetAlg) {
      case haar:
        haarClassifier.load(haarPath);
        break;
      case deep:
        dnnClassifier = readNetFromCaffe(dnnProtoPath, dnnModelPath);
        break;
    }

    return true;
}


void RPPG::processFrame(Mat &frameRGB, Mat &frameGray, int time) {


    if(!faceValid) detectFace(frameRGB, frameGray);
    if (!faceValid){

           return;
    } else{
        faceValid = false;
    }

    Mat subframe = frameRGB(box);
    Mat hsv ;
    cvtColor(subframe,hsv,COLOR_BGR2HSV);

    Mat mask_hsv;
    inRange(hsv,Scalar(1,0,0),Scalar(26,255,255),mask_hsv);

    Mat mask_fg;
    bitwise_and(subframe,subframe,mask_fg,mask_hsv);

    vector<Mat> channels;
    split(mask_fg,channels);

    Scalar blue  = mean(channels[0]);
    Scalar green = mean(channels[1]);
    Scalar red   = mean(channels[2]);

    if(data_buffer.rows>2){
        if (enough_info==false){
            bpms_TEST.clear();
            decide_freq_flag=false;
            decide_freq=0;
            data_buffer.release();
            data_buffer2.release();
            data_buffer3.release();
            times.clear();
            hr_prev=0;

            rr=0;
            rr_prev=0;
            rrs_TEST.clear();

        }else if( abs( data_buffer.at<double>(data_buffer.rows-1,0) -red[0])>30 ||
                abs(data_buffer2.at<double>(data_buffer2.rows-1,0) -green[0])>30||
                abs(data_buffer3.at<double>(data_buffer3.rows-1,0) -blue[0])>30
                ){
            cout<<"total_reset2.  face moving"<<" "<<data_buffer.at<double>(data_buffer.rows-1,0) <<" "<< red[0]
                    <<" "<<data_buffer2.at<double>(data_buffer2.rows-1,0) <<" "<< green[0]
                    <<" "<<data_buffer3.at<double>(data_buffer3.rows-1,0) <<" "<< blue[0]<<endl;
            bpms_TEST.clear();
            decide_freq_flag=false;
            decide_freq = 0;
            data_buffer.release();
            data_buffer2.release();
            data_buffer3.release();
            times.clear();
            hr_prev =0;

            rr=0;
            rr_prev=0;
            rrs_TEST.clear();
        }
    }



    data_buffer.push_back(Mat(1,1,CV_64F,red[0]));
    data_buffer2.push_back(Mat(1, 1, CV_64F, green[0]));
    data_buffer3.push_back(Mat(1, 1, CV_64F, blue[0]));


    auto eend = std::chrono::system_clock::now();
    auto TIME = (double)std::chrono::duration_cast<std::chrono::milliseconds>(eend-sstart).count()/1000.0;


    times.push_back(TIME);


//    printf("TIME:%.2f\n",TIME);

    Mat half_buffer,half_buffer2,half_buffer3;
    data_buffer.rowRange(data_buffer.rows/2,data_buffer.rows).copyTo(half_buffer);
    data_buffer2.rowRange(data_buffer2.rows/2,data_buffer2.rows).copyTo(half_buffer2);
    data_buffer3.rowRange(data_buffer3.rows/2,data_buffer3.rows).copyTo(half_buffer3);

    Scalar half_mean, half_std;
    meanStdDev(half_buffer, half_mean, half_std);

    Scalar half_mean2, half_std2;
    meanStdDev(half_buffer2, half_mean2, half_std2);

    Scalar half_mean3, half_std3;
    meanStdDev(half_buffer3, half_mean3, half_std3);


    if (half_std[0]>15 || half_std2[0]>15 || half_std3[0]>15){

        cout<<"total_reset. Not stable"<<endl;
        bpms_TEST.clear();
        decide_freq_flag=false;
        decide_freq = 0;
        data_buffer.release();
        data_buffer2.release();
        data_buffer3.release();
        times.clear();
        hr_prev =0;

        rr=0;
        rr_prev=0;
        rrs_TEST.clear();

    }

    int L = times.size();

    if(L>buffer_size){
        push(data_buffer);
        push(data_buffer2);
        push(data_buffer3);
        times.pop_front();


        if(bpms_TEST.size()>50) {
            bpms_TEST.pop_front();
            rrs_TEST.pop_front();
        }
        L = buffer_size;

    }else if (enough_info==false){
        enough_info=true;
    }


    if(L==buffer_size) {

        fps_TEST = L / (times.back() - times.front());
//        cout<<fps_TEST<<endl;



        Mat buffer1,buffer2,buffer3;
        interp(data_buffer, times).copyTo(buffer1);
        interp(data_buffer2, times).copyTo(buffer2);
        interp(data_buffer3, times).copyTo(buffer3);

//        Mat buffer1,buffer2,buffer3;
//        data_buffer.copyTo(buffer1);
//        data_buffer2.copyTo(buffer2);
//        data_buffer3.copyTo(buffer3);


        Scalar R_raw_mean, R_raw_std;
        meanStdDev(data_buffer, R_raw_mean, R_raw_std);

        Scalar B_raw_mean, B_raw_std;
        meanStdDev(data_buffer3, B_raw_mean, B_raw_std);

        double  R_merge = (R_raw_std[0]/R_raw_mean[0]) / (B_raw_std[0]/B_raw_mean[0]);
        double spo2 = 100-5*R_merge;
        int o2 = (int)spo2;
//        cout<<"spo2:"<<spo2<<endl;



//        for(int i = 0 ; i < data_buffer.rows;++i){
//            for(int j = 0 ; j <data_buffer.cols;++j){
//                buffer1.at<double>(i,j)*=(0.54-0.46*cos(2*3.1416*i/(L-1)));
//                buffer2.at<double>(i,j)*=(0.54-0.46*cos(2*3.1416*i/(L-1)));
//                buffer3.at<double>(i,j)*=(0.54-0.46*cos(2*3.1416*i/(L-1)));
//            }
//        }


        for(int i = 0 ; i < data_buffer.rows;++i){
            for(int j = 0 ; j <data_buffer.cols;++j){
                buffer1.at<double>(i,j)*=hamming[i];//(0.54+0.46*cos(2*3.1416*i/(L-1)));
                buffer2.at<double>(i,j)*=hamming[i];//(0.54+0.46*cos(2*3.1416*i/(L-1)));
                buffer3.at<double>(i,j)*=hamming[i];//(0.54+0.46*cos(2*3.1416*i/(L-1)));
            }
        }


        Scalar interpolated_mean, interpolated_std;
        meanStdDev(buffer1, interpolated_mean, interpolated_std);
        Mat norm1 = (buffer1 - interpolated_mean) / interpolated_std;

        Scalar interpolated2_mean, interpolated2_std;
        meanStdDev(buffer2, interpolated2_mean, interpolated2_std);
        Mat norm2 = (buffer2 - interpolated2_mean) / interpolated2_std;

        Scalar interpolated3_mean, interpolated3_std;
        meanStdDev(buffer3, interpolated3_mean, interpolated3_std);
        Mat norm3 = (buffer3 - interpolated3_mean) / interpolated3_std;


        Mat x = 3 * norm1 - 2 * norm2;
        Mat y = 1.5 * norm1 + norm2 - 1.5 * norm3;

        Scalar x_mean, x_std;
        meanStdDev(x, x_mean, x_std);

        Scalar y_mean, y_std;
        meanStdDev(y, y_mean, y_std);

        Mat norm4 = x - (x_std[0] / y_std[0]) * y;

        Mat norm5;



        if (decide_freq_flag){
            double tmp_freq = decide_freq/60.0;
            bandpass(norm4,norm5,data_buffer.rows*(tmp_freq-0.2)/fps_TEST,data_buffer.rows*(tmp_freq+0.2)/fps_TEST);
        }else{
            bandpass(norm4, norm5, data_buffer.rows*0.6/fps_TEST, data_buffer.rows*3.0/fps_TEST);

        }

        Mat norm6;
        bandpass(norm4, norm6, data_buffer.rows*0.08/fps_TEST, data_buffer.rows*0.5/fps_TEST);

        Mat raw = Mat(norm5.size(), CV_64F);
        timeToFrequency(norm5*30, raw, true);

        Mat raw2 = Mat(norm6.size(), CV_64F);
        timeToFrequency(norm6*30, raw2, true);


        vector<double > freqs;
        for (int i = 0; i < L /2 + 1; ++i) {
            freqs.push_back(fps_TEST/ L * (i+1) * 60.0);
        }

        vector<double > freqs2;
        for (int i = 0; i < L /2 + 1; ++i) {
            freqs2.push_back(fps_TEST/ L * (i+1) * 60.0);
        }



        vector<double > FFT;
        for(int i = 0 ; i<raw.rows;++i){
            for(int j = 0 ; j<raw.cols;++j){
                auto value = raw.at<double >(i,j);


                FFT.push_back(value);
            }
        }


        vector<double > FFT2;
        for(int i = 0 ; i<raw2.rows;++i){
            for(int j = 0 ; j<raw2.cols;++j){
                auto value2 = raw2.at<double >(i,j);

                FFT2.push_back(value2);
            }
        }



        auto it1=find_if(freqs.begin(),freqs.end(),[](double  x){return x>48;} );
        auto it2=find_if(freqs.begin(),freqs.end(),[](double  x){return x>180;} );

        int first_index =it1 - freqs.begin();
        int last_index  =it2 - freqs.begin();


        auto it3=find_if(freqs2.begin(),freqs2.end(),[](double  x){return x>10;} );
        auto it4=find_if(freqs2.begin(),freqs2.end(),[](double  x){return x>30;} );

        int first_index2 =it3 - freqs2.begin();
        int last_index2  =it4 - freqs2.begin();



        vector<double > pruned(FFT.begin()+first_index,FFT.begin()+last_index);
        FFT.clear();
        FFT.assign(pruned.begin(),pruned.end());

        vector<double > pruned2(FFT2.begin()+first_index2,FFT2.begin()+last_index2);
        FFT2.clear();
        FFT2.assign(pruned2.begin(),pruned2.end());



        vector<double >FREQUENCY_TEST(freqs.begin()+first_index,freqs.begin()+last_index);
        freqs.clear();
        freqs.assign(FREQUENCY_TEST.begin(),FREQUENCY_TEST.end());


        vector<double >FREQUENCY_TEST2(freqs2.begin()+first_index2,freqs2.begin()+last_index2);
        freqs2.clear();
        freqs2.assign(FREQUENCY_TEST2.begin(),FREQUENCY_TEST2.end());



        int idx3 = max_element(FFT.begin(),FFT.end())-FFT.begin();


        int idx4 = max_element(FFT2.begin(),FFT2.end())-FFT2.begin();


        if (cvIsNaN(freqs[idx3])){
            bpms_TEST.clear();
            decide_freq_flag=false;
            decide_freq = 0;
            data_buffer.release();
            data_buffer2.release();
            data_buffer3.release();
            times.clear();
            hr_prev =0;


            rr=0;
            rr_prev=0;
            rrs_TEST.clear();
        }else{

            bpm_TEST=freqs[idx3];
            bpms_TEST.push_back(bpm_TEST);

            rr=freqs2[idx4];
            rrs_TEST.push_back(rr);

//            printf("bpm_TEST:%.2lf,idx3 :%d  bpms_TEST.size :%d\n",bpm_TEST,idx3,bpms_TEST.size());

            if(bpms_TEST.size()>50){

                double  sum1=0;
                for_each(bpms_TEST.begin(),bpms_TEST.end(),[&](double  x){ sum1+=x ;});

                double  mean1 = sum1/bpms_TEST.size();

                double  accum1=0.0;
                for_each(bpms_TEST.begin(),bpms_TEST.end(),[&](double  x){ accum1 += (x-mean1)*(x-mean1); });

                double  std_bpms_TEST = sqrt(accum1/(bpms_TEST.size()-1));


                double  sum2=0;
                for_each(rrs_TEST.begin(),rrs_TEST.end(),[&](double  x){ sum2+=x ;});
                double  mean2 = sum2/rrs_TEST.size();


                if(std_bpms_TEST<=5){
                        hr = 0.5*hr_prev+0.5*mean1;
                        hr_prev = hr;
                        bpm_disp=hr;
                        decide_freq_flag =true;
                        decide_freq = mean1;

                        rr=0.5*rr_prev+0.5*mean2;
                        rr_prev=rr;
                        rr_disp=rr;



                        cout<<"bpm: "<<bpm_disp<<"  "<<" std: "<<std_bpms_TEST<<endl;
                    }
            }else{
                decide_freq_flag =false;
                bpm_disp=0;
                rr_disp=0;

            }

        }

    }

}


void RPPG::detectFace(Mat &frameRGB, Mat &frameGray) {

//    cout << "Scanning for faces…" << endl;
    vector<Rect> boxes = {};

    switch (faceDetAlg) {
      case haar:
        // Detect faces with Haar classifier
        haarClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, CASCADE_SCALE_IMAGE, minFaceSize);
        break;
      case deep:
        // Detect faces with DNN
        Mat resize300;
        cv::resize(frameRGB, resize300, Size(300, 300));

        Mat blob = blobFromImage(resize300, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));

        dnnClassifier.setInput(blob);
        Mat detection = dnnClassifier.forward();    

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        float confidenceThreshold = 0.5;

        for (int i = 0; i < detectionMat.rows; i++) {
          float confidence = detectionMat.at<float>(i, 2);
          if (confidence > confidenceThreshold) {
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frameRGB.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frameRGB.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frameRGB.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frameRGB.rows);
            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));
            boxes.push_back(object);
          }
        }
        break;
    }

    if (boxes.size() > 0) {

//        cout << "Found a face" << endl;

        setNearestBox(boxes);
//
//        updateROI();
//        updateMask(frameGray);
        faceValid = true;

    } else {

        cout << "Found no face" << endl;
        invalidateFace();
    }
}

void RPPG::setNearestBox(vector<Rect> boxes) {
    int index = 0;
    Point p = box.tl() - boxes.at(0).tl();
    int min = p.x * p.x + p.y * p.y;
    for (int i = 1; i < boxes.size(); i++) {
        p = box.tl() - boxes.at(i).tl();
        int d = p.x * p.x + p.y * p.y;
        if (d < min) {
            min = d;
            index = i;
        }
    }
    box = boxes.at(index);
}


void RPPG::updateROI() {
//    this->roi = Rect(Point(box.tl().x + 0.3 * box.width, box.tl().y + 0.1 * box.height),
//                     Point(box.tl().x + 0.7 * box.width, box.tl().y + 0.25 * box.height));

    this->roi = Rect(Point(box.tl().x + 0.35 * box.width, box.tl().y + 0.08 * box.height),
                     Point(box.tl().x + 0.65 * box.width, box.tl().y + 0.20 * box.height));

}

void RPPG::updateMask(Mat &frameGray) {

//    cout << "Update mask" << endl;

    mask = Mat::zeros(frameGray.size(), frameGray.type());
    rectangle(mask, this->roi, WHITE, FILLED);

//    rectangle(mask, this->roi_nose, WHITE, FILLED);

}



void RPPG::invalidateFace() {

    faceValid = false;
}


double RPPG::calculateO2(double r, double nearInfrared) {

    double scaledown = 1;
    double red = r;	//red intensity of face // replace 9 with red
    double nir = nearInfrared; //nir intensity of face // nearInfrared // TODO: replace .5 with nearInfrared
    double result = 0;	//o2 level of people

    scaledown = configure(red, nir);
    nir = nir/scaledown;
    result = o2_level(red, nir);
    return result;

}

double RPPG::o2_level(double hbO2, double hb) {
    return hbO2/(hbO2+hb);
}

double RPPG::configure(double red, double nir) {
    double target_o2_level = .98;
    double hbO2 = red;
    double hb = nir;
    double scaledown = 1;
    if(.02*hbO2 < hb) {
        scaledown = hb / (.02 * hbO2);
    }
    return scaledown;
}



std::vector<double> RPPG::interp(const std::vector<double> &inputs, const std::deque<double> &times) {
    assert(times.size() == inputs.size());
    std::size_t cnt = inputs.size();
    double interval = (times[cnt-1] - times[0]) / (cnt - 1);
    std::vector<double> time_stamp(cnt);
    for(std::size_t i{}; i<cnt; i++)
    {
        time_stamp[i] = times[0] + i * interval;
    }

    std::vector<double> result(cnt);
    result[0] = inputs[0];
    for(std::size_t i=1; i<cnt; i++)
    {
        result[i] = (inputs[i-1] * (times[i] - time_stamp[i]) + inputs[i] * (time_stamp[i] - times[i-1]))/(times[i] - times[i-1]);
    }

    return result;
}

Mat1d RPPG::interp(const Mat1d &data_buffer, const std::deque<double> &times) {
    std::vector<double> inputs = std::vector<double>(data_buffer.reshape(1, 1));
    std::vector<double> outputs = interp(inputs, times);
    cv::Mat1d mat = cv::Mat1d(outputs);
    return mat.reshape(1, times.size()).clone();
}