//Author: Antonio Gaglione

#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp> // For morphological operations

#include "pugixml.hpp"
#include "help.h"


int main() {
    

    int flag = 0;  // set this flag to use internet image


    // Load the reference image 
    std::string referenceImagePath = "ParkingLot_dataset-20240908T160739Z-001/ParkingLot_dataset/sequence0/frames/2013-02-24_17_55_12.jpg";
    cv::Mat referenceImg = cv::imread(referenceImagePath);

    cv::Mat referenceImgForPlotting = cv::imread(referenceImagePath);  // needed for plotting

    if (referenceImg.empty()) {  // error handling when image failed to read
        std::cerr << "Failed to load reference image file: " << referenceImagePath << std::endl;
        return 1;
    }


    // Load reference image bounding boxes
    std::string referenceXmlPath = "ParkingLot_dataset-20240908T160739Z-001/ParkingLot_dataset/sequence0/bounding_boxes/2013-02-24_17_55_12.xml";
    pugi::xml_document referenceDoc;

    if (!referenceDoc.load_file(referenceXmlPath.c_str())) {  // error handling if xml file failed to load
        std::cerr << "Failed to load reference XML file: " << referenceXmlPath << std::endl;
        return 1;
    }
    
    // detect parking lot spaces
    double overlapThresh = 0.08; // Define overlap threshold here
    std::vector<cv::Rect> parkingspaces = detectParkingLotsAndCars(referenceImagePath, overlapThresh);
    
    for (const auto& box : parkingspaces) {
        rectangle(referenceImgForPlotting, box, cv::Scalar(0, 255, 0), 2);
    }

     
    // Load the test image : image for detection
    std::string testImagePath = "ParkingLot_dataset-20240908T160739Z-001/ParkingLot_dataset/sequence0/frames/2013-02-24_11_30_05.jpg";
    //std::string testImagePath = "test.jpg";
    
    cv::Mat testImg = cv::imread(testImagePath);       //  Needed for ground truth detection
    cv::Mat testImgorg = cv::imread(testImagePath);   // needed for detection
    cv::Mat testImgmain = cv::imread(testImagePath);   // needed to plot original image

    
    if (testImg.empty()) { // Error handling if test image failed to load
        std::cerr << "Failed to load test image file: " << testImagePath << std::endl;
        return 1;
    }
    

    if (flag==0){

    // Load test image xml file
    std::string testXmlPath = "ParkingLot_dataset-20240908T160739Z-001/ParkingLot_dataset/sequence0/bounding_boxes/2013-02-24_11_30_05.xml";
    pugi::xml_document testDoc;

    if (!testDoc.load_file(testXmlPath.c_str())) {  // Error handling if XML file failed to load
        std::cerr << "Failed to load test XML file: " << testXmlPath << std::endl;
        return 1;
    }

    // Load mask image 
    std::string masksFolderPath = "ParkingLot_dataset-20240908T160739Z-001/ParkingLot_dataset/sequence5/masks/2013-04-12_15_00_09";
    std::string maskPath = masksFolderPath + ".png";
    cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE); // load mask image as grayscale
    if (mask.empty()) {  // Error handling if mask image failed to load
         std::cerr << "Failed to load mask image: " << maskPath << std::endl;
        return 1;
    }

    // car detection using our algorithm 

    std::vector<cv::Rect> cardetect = carDetect(testImgorg); // extract the bounding boxes of car detection
    double minCarArea = 500; // Adjust based on your needs
    
    vector<Rect> boundingBoxes_detect = detectCarsInBoundingBoxes(testImgorg, parkingspaces, minCarArea);
    for (const auto& bbox : boundingBoxes_detect) {
        cv::rectangle(testImgorg, bbox, cv::Scalar(0, 255, 0), 2); // Red color for bounding boxes
        
    }
    

    // Variables to keep track of counts in detection
    int properlyParkedCount_detect = 0;
    int improperlyParkedCount_detect = 0;


    // Draw bounding boxes on the input image
    for (const auto& bbox : cardetect) {

        int bboxWidth = bbox.width;
        int bboxHeight = bbox.height;
        int area =bboxWidth*bboxHeight;

        // check for proper parking using reference bounding boxes and the box detected
        bool properlyParked = isProperlyParked(bbox,boundingBoxes_detect);  

        if (properlyParked) { // car properly parked
            properlyParkedCount_detect++;  // count number of properly parked car
            cv::rectangle(testImgorg, bbox, cv::Scalar(255, 0, 0), 2); // blue color for bounding boxes
        }
        else{
            improperlyParkedCount_detect++;  // count number of improperly parked cars
            cv::rectangle(testImgorg, bbox, cv::Scalar(0, 0, 255), 2); // Red color for bounding boxes
        }
           
    }

    //=============== Groundtruth labeling =============
    std::vector<cv::Rect> boundingBoxes_reference;  // reference bounding boxes
    boundingBoxes_reference.clear();     

    // Draw the reference bounding boxes to reference image
    for (pugi::xml_node space : referenceDoc.child("parking").children("space")) {
        float centerX = space.child("rotatedRect").child("center").attribute("x").as_float();
        float centerY = space.child("rotatedRect").child("center").attribute("y").as_float();
        float width = space.child("rotatedRect").child("size").attribute("w").as_float();
        float height = space.child("rotatedRect").child("size").attribute("h").as_float();
        float angle = space.child("rotatedRect").child("angle").attribute("d").as_float();

        cv::Point2f center(centerX, centerY);
        cv::Size2f size(width, height);
        
        // convert reference bounding boxes to rectangular
        cv::Rect boundingBox_reference = rotatedRectToBoundingRect(center, size, angle);  
        boundingBoxes_reference.push_back(boundingBox_reference);  // save the bounding boxes

        // Draw the bounding box in green
        cv::rectangle(referenceImg, boundingBox_reference, cv::Scalar(0, 255, 0), 2); // Red color for bounding boxes
    }
    
    // groundtruth car segmentation
    colorizeImageBasedOnMask(testImg, mask);  // color ground truth image base on the mask value
    
    // Extracting bounding boxes from mask
    std::vector<cv::Rect> allBoundingBoxes;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    // Extracting groundtruth car boundingboxes from mask
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // compute contour
    
    // loop through the contours to get all bounding boxes
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Rect boundingBox = cv::boundingRect(contours[i]);
        allBoundingBoxes.push_back(boundingBox);
    }
    
    
    // computing the necessary metrics:  meanIoU and meanAP between bounding box for detected parking spaces and ground truth spaces

    float iouThreshold1 = 0.5f; // threshold
    float meanIoU1 = computeMeanIoU(boundingBoxes_reference, parkingspaces);
    float meanAP1 = computeMeanAP({boundingBoxes_reference}, {parkingspaces}, iouThreshold1);
    

    // computing the necessary metrics:  meanIoU and meanAP between detected car and ground truth

    float iouThreshold2 = 0.5f; // threshold
    float meanIoU2 = computeMeanIoU(allBoundingBoxes, cardetect);
    float meanAP2 = computeMeanAP({allBoundingBoxes}, {cardetect}, iouThreshold2);
    
    //================== Displaying output ================= 
    std::string summary1 ="  meanIoU: " + std::to_string(meanIoU1) + "\n" + 
                          "  meanAP: " + std::to_string(meanAP1);

   cv::putText(referenceImgForPlotting, summary1, cv::Point(10, referenceImgForPlotting.rows - 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    
    // Add detection information to the test image
    //std::string summary2 = "Properly Parked: " + std::to_string(properlyParkedCount_detect) + "\n" + 
                        "  Improperly Parked: " + std::to_string(improperlyParkedCount_detect) + "\n" +
                        "  meanIoU: " + std::to_string(meanIoU2) + "\n" + 
                      "  meanAP: " + std::to_string(meanAP2);

    //cv::putText(testImgorg, summary2, cv::Point(10, testImgorg.rows - 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Create the minimap image for detection using our algorithm
    cv::Mat minimap_detect = createMinimap(referenceImgForPlotting,parkingspaces, boundingBoxes_detect);

    //resizing minimap plot
    //cv::Size newSize(320, 320); 

    // Resize the image
    //cv::resize(minimap_detect, minimap_detect, newSize, 0, 0, cv::INTER_LINEAR);
    
    // Making necessary plots
    cv::imshow("Reference Parking Lot (Proper parking positions))", referenceImg);
    
    cv::imshow("Reference Parking Lot Spaces detection", referenceImgForPlotting);
    
    cv::imshow("Input test image", testImgmain);
    cv::imshow("Ground Truth detection", testImg);

    
    cv::imshow("Car detection", testImgorg);
    cv::imshow("Detection minimap", minimap_detect);


     // bool success1 = cv::imwrite("results/parkinglot_2013-02-24_17_55_12.jpg", referenceImgForPlotting);
    
   // bool success2 = cv::imwrite("results/gt_parkinglot_2013-02-24_17_55_12.jpg", referenceImg);

   // bool success3 = cv::imwrite("results/car_detection_0_2013-02-24_10_35_04.jpg", testImgorg);

    //bool success4 = cv::imwrite("results/gt_car_detection_0_2013-02-24_10_35_04.jpg", testImg);
    
    //bool success5 = cv::imwrite("results/minimap_0_2013-02-24_10_35_04.jpg", minimap_detect);


    }
    else{
        // car detection using our algorithm 

        std::vector<cv::Rect> boundingBoxes_detect = carDetect(testImgorg); // extract the bounding boxes of car detection
    
        for (const auto& bbox : boundingBoxes_detect) {
            cv::rectangle(testImgorg, bbox, cv::Scalar(0, 0, 255), 2); // Red color for bounding boxes
        
        }
        cv::imshow("Car detection", testImgorg);

    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
