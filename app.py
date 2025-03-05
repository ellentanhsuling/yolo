import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
from PIL import Image
import time
from pathlib import Path
from utils.helpers import download_model, get_model_info

def main():
    # Set page config
    st.set_page_config(
        page_title="YOLO Object Detection", 
        page_icon="🔍",
        layout="wide"
    )
    
    # Title and introduction
    st.title("YOLO Object Detection")
    st.markdown("""
    This app uses YOLOv5 to detect objects in images and videos. 
    Upload an image or video and adjust the detection parameters to see the results.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Detection Parameters")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.25, 
        step=0.05,
        help="Higher values filter out less confident detections"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.45, 
        step=0.05,
        help="Higher values allow more overlap between boxes"
    )
    
    # Load model
    with st.spinner("Loading YOLO model..."):
        model = download_model()
        model.conf = conf_threshold  # confidence threshold
        model.iou = iou_threshold    # NMS IoU threshold
    
    # Model info
    model_info = get_model_info(model)
    with st.expander("Model Information"):
        st.write(f"**Model:** {model_info['model_type']}")
        st.write(f"**Number of Classes:** {model_info['num_classes']}")
        st.write("**Classes:**")
        # Create a multi-column layout for class names
        cols = st.columns(4)
        for i, class_name in enumerate(model_info['class_names'].values()):
            cols[i % 4].write(f"- {class_name}")
    
    # File upload
    st.header("Upload Content")
    file_type = st.radio("Select input type:", ["Image", "Video"])
    
    if file_type == "Image":
        process_image(model)
    else:
        process_video(model)

def process_image(model):
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform detection on button click
        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                # Convert PIL Image to numpy array
                img_array = np.array(image)
                
                # Run detection
                start_time = time.time()
                results = model(img_array)
                end_time = time.time()
                
                # Show detection time
                st.info(f"Detection completed in {end_time - start_time:.2f} seconds")
                
                # Display results
                result_image = Image.fromarray(results.render()[0])
                st.image(result_image, caption="Detection Result", use_column_width=True)
                
                # Show detection details
                st.subheader("Detection Details")
                result_df = results.pandas().xyxy[0]  # Get detection results as DataFrame
                if len(result_df) > 0:
                    # Format the DataFrame for display
                    result_df = result_df.rename(columns={
                        'xmin': 'X Min', 
                        'ymin': 'Y Min', 
                        'xmax': 'X Max', 
                        'ymax': 'Y Max', 
                        'confidence': 'Confidence', 
                        'class': 'Class ID', 
                        'name': 'Object'
                    })
                    result_df = result_df[['Object', 'Confidence', 'X Min', 'Y Min', 'X Max', 'Y Max']]
                    result_df['Confidence'] = result_df['Confidence'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(result_df)
                else:
                    st.write("No objects detected.")

def process_video(model):
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.video(video_path)
        st.info(f"Video Info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Perform detection on button click
        if st.button("Detect Objects"):
            # Progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a temporary file for the processed video
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_path = output_file.name
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Update progress
                    progress = int(100 * frame_count / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress}%)")
                    
                    # Perform detection
                    results = model(frame)
                    
                    # Draw detections on frame
                    annotated_frame = results.render()[0]
                    
                    # Write frame to output video
                    out.write(annotated_frame)
                
                # Release resources
                cap.release()
                out.release()
                
                # Display the processed video
                st.success("Processing complete!")
                st.video(output_path)
                
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
                
            finally:
                # Clean up temporary files
                try:
                    Path(video_path).unlink()
                    Path(output_path).unlink()
                except:
                    pass

if __name__ == "__main__":
    main()
