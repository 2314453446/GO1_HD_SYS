from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi import AutoDetectionModel

yolov7_model_path = 'D:\masterPROJECT\laser_weeding\yolov7\yolov7-pytorch\logs\ep600-loss0.047-val_loss0.054.pth'

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov7', #
    model_path=yolov7_model_path,
    confidence_threshold=0.5,
    device="cuda:0"
)

result = get_sliced_prediction(
    r"D:\masterPROJECT\laser_weeding\yolov7\yolov7-pytorch\img\img_1.png",
    detection_model,
    slice_height = 640,
    slice_width = 320,
    overlap_height_ratio = 0.0,
    overlap_width_ratio = 0.0,
    postprocess_type= "NMS",
    postprocess_match_threshold = 0.2,
    postprocess_class_agnostic=True

)

result.export_visuals(export_dir="./test_output")