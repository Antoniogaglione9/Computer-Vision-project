# 🅿️ Parking Space Detection and Occupancy Monitoring

This project aims to develop a computer vision system for the detection of parking spaces and the identification of vehicle occupancy within those spaces. The solution combines classical image processing techniques to analyze parking lot images, detect space boundaries, and identify whether vehicles are correctly parked.

---

## 📌 Objectives

- Automatically localize parking spaces using line detection and geometric analysis
- Detect vehicles and determine if a parking space is occupied or free
- Generate a visual "mini map" showing the real-time occupancy of all detected parking spots
- Evaluate system performance using IoU and Average Precision metrics

---

## 🧠 Techniques Used

- **Hough Line Transform** for parking space boundary detection  
- **HSV color thresholding** to isolate vehicles  
- **Morphological operations** (dilation, erosion) to clean image masks  
- **Edge detection** (Canny) to find sharp contours  
- **Contour filtering** to identify valid vehicles  
- **Bounding box comparison** to verify car-to-slot alignment  
- **Mini map creation** to summarize the status of all parking spots

---

## 🧪 Dataset

- Dataset used: [PUCPR Parking Lot Database](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
- Includes:
  - RGB parking lot images
  - XML ground-truth files with slot annotations
  - Segmentation masks for parked cars (inside/outside spaces)

---

## ⚙️ Algorithm Overview

### 1. **Parking Lot Detection**
- Convert image to grayscale
- Apply Gaussian Blur and Canny edge detection
- Use Hough Transform to detect lines
- Group lines into bounding boxes representing parking spaces

### 2. **Car Detection**
- Convert image to HSV and apply color thresholding
- Use morphological operations to refine detection mask
- Extract contours and filter by size and aspect ratio
- Extract bounding boxes for detected vehicles

### 3. **Car-in-Space Matching**
- For each detected slot (ROI), apply car detection
- Mark the space as `occupied` if a car is detected inside

### 4. **Mini Map Computation**
- Visualize all slots and their status (free/occupied) on a 2D map

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Mean IoU** | Measures how well the detected bounding boxes match ground truth |
| **Mean AP**  | Precision score for vehicle detection across various images |

> Example Results (Sequence 0):

| Image                        | Mean IoU | Mean AP  |
|-----------------------------|----------|----------|
| 2013-02-24_10_05_04.jpg     | 0.327    | 0.545    |
| 2013-02-24_15_10_09.jpg     | 0.254    | 0.512    |
| 2013-02-24_17_55_12.jpg     | 0.234    | 0.500    |

---

## 📁 Project Structure

