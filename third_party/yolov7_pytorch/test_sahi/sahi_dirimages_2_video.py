import os
import cv2
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# Define the path to your model and images
yolov7_model_path = 'D:\masterPROJECT\laser_weeding\yolov7\yolov7-pytorch\logs\ep600-loss0.047-val_loss0.054.pth'
image_folder = r"D:\masterPROJECT\paper\object_detection\figures\result_video\images"
output_video_path = r'D:\masterPROJECT\paper\object_detection\figures\result_video/output_video.avi'
output_folder = r'D:\masterPROJECT\paper\object_detection\figures\result_video\test_output'  # Folder to save processed images
frame_rate = 5  # Adjustable frame rate for video
# Initialize the detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov7',
    model_path=yolov7_model_path,
    confidence_threshold=0.5,
    device="cuda:0"
)

# Prepare to write the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Iterate over all images in the directory
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for valid image files
        image_path = os.path.join(image_folder, filename)
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=640,
            slice_width=320,
            overlap_height_ratio=0.0,
            overlap_width_ratio=0.0,
            postprocess_type="NMS",
            postprocess_match_threshold=0.2,
            postprocess_class_agnostic=True
        )

        # Export visuals to the output directory
        export_path = os.path.join(output_folder, filename)
        result.export_visuals(export_dir=output_folder,file_name=filename)


        # Read the processed image back into OpenCV
        image_array = cv2.imread(export_path)

        # Initialize video writer
        if out is None and image_array is not None:
            h, w, _ = image_array.shape
            out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (w, h))

        # Write frame to video
        if image_array is not None:
            out.write(image_array)

# Release resources
if out:
    out.release()
    print("Video saved at", output_video_path)
else:
    print("No images processed.")