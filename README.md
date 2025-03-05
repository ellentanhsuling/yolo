# YOLO Object Detection with Streamlit

This project implements object detection using YOLOv5 and provides an interactive web interface using Streamlit.

## Features

- Upload and detect objects in images and videos
- Adjust confidence and IoU thresholds through an interactive UI
- View detection results with bounding boxes
- Get detailed information about detected objects

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/yolo-detector.git
cd yolo-detector
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

This will start a local web server and open the app in your default web browser.

## How to Use the App

1. Select whether you want to process an image or video
2. Upload your file
3. Adjust detection parameters in the sidebar if needed
4. Click the "Detect Objects" button
5. View the results with detected objects highlighted

## Project Structure

- `app.py`: Streamlit application for the web interface
- `utils/helpers.py`: Helper functions for model loading
- `requirements.txt`: Required Python packages

## How YOLO Works

YOLO (You Only Look Once) is an object detection algorithm that divides the image into a grid and predicts bounding boxes and class probabilities directly in a single pass through the neural network. This makes it extremely fast compared to other object detection algorithms.

The key principles of YOLO:

1. Divides the input image into an SxS grid
2. For each grid cell, predicts N bounding boxes with confidence scores
3. For each bounding box, predicts class probabilities
4. Applies non-maximum suppression to remove duplicate detections

## License

This project is licensed under the MIT License.
```

## How to run the Streamlit app

1. Create the directory structure:

```bash
mkdir -p yolo-detector/utils
```

2. Navigate to the project directory:

```bash
cd yolo-detector
```

3. Create all the files mentioned above with the provided content.

4. Install the required packages:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit app:

```bash
streamlit run app.py
