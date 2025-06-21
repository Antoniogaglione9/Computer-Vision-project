#ifndef HELP_H
#define HELP_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp> // For morphological operations
#include "pugixml.hpp"

using namespace cv;
using namespace std;

void colorizeImageBasedOnMask(cv::Mat& mainImage, const cv::Mat& maskImage) ;
   

// Function to calculate Intersection over Union (IoU) between two bounding boxes
float computeIoU(const cv::Rect& boxA, const cv::Rect& boxB);

// Function to calculate mean Intersection over Union (mIoU)
float computeMeanIoU(const std::vector<cv::Rect>& gtBoxes, const std::vector<cv::Rect>& predBoxes);

// Function to calculate Average Precision (AP)
float computeAveragePrecision(const std::vector<cv::Rect>& gtBoxes, const std::vector<cv::Rect>& predBoxes, float iouThreshold);

// Function to compute the mean Average Precision (mAP)
float computeMeanAP(const std::vector<std::vector<cv::Rect>>& allGtBoxes, const std::vector<std::vector<cv::Rect>>& allPredBoxes, float iouThreshold);

// Function to convert a rotated rectangle (defined by center, size, and angle) to an axis-aligned bounding rectangle
cv::Rect rotatedRectToBoundingRect(cv::Point2f center, cv::Size2f size, float angle);

// Function to check if a test bounding box is within any reference bounding box
bool isProperlyParked(const cv::Rect& testBox, const std::vector<cv::Rect>& referenceBoxes);

// Function to calculate Intersection over Union (IoU) of two bounding boxes
float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);

// Function to create a minimap plot with shaded areas for parking slots
cv::Mat createMinimap(const cv::Mat& referenceImage, const std::vector<cv::Rect>& referenceBoxes, const std::vector<cv::Rect>& detectedBoxes);
std::vector<cv::Rect> applyNMS(const std::vector<cv::Rect>& boxes, float threshold);

Rect computeIntersection(const Rect& rect1, const Rect& rect2);
double computeSlantAngle(int x1, int y1, int x2, int y2);


double computeAngle(const cv::Point& pt1, const cv::Point& pt2);
int computeArea(const Rect& rect);

vector<Rect> mergeRectangles(const vector<Rect>& rectangles, double overlapThresh);

vector<Rect> detectParkingLotsAndCars(const string& imagePath, double overlapThresh);

vector<Rect> detectCarsInBoundingBoxes(const Mat& image, const vector<Rect>& boundingBoxes, double minCarArea);
 
vector<Rect> carDetect(const Mat& img);
#endif // HELP_H
