import cv2 
from ultralytics import YOLO
import numpy as np 

def load_yolo_models(model_path):
    """"load YOLO model with the given path."""
    return YOLO(model_path)

def detect_objects(model,frame,class_name,display_width):

    """Detect segmentation masks,draw bounding boxes, and overlay labels."""
    #resize the frame first
    resized_frame = cv2.resized(frame,(display_width,int(display_width / frame.shape[1] * frame.shape))) 

    results= model(resized_frame, stream=True)
    for r in results:
        if r.masks:
            masks = r.masks.data.cpu().numpy()

            for i , mask in enumerate(masks):
                mask_resized = cv2.resize(mask, (resized_frame.shape[1],resized_frame.shape[0]),interpolation=cv2.INTER_NEAREST)

                                          
                overlay = np.zeros_like(resized_frame,dtype=np.uint8)
                overlay[mask_resized.astype(bool)] = [225, 0, 255]


                resized_frame = cv2.addWeighted(resized_frame, 0.8, overlay, 0.2, 0)


        if r.boxes:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf =round(box.conf[0].item(), 2)
                cls = int(box.cls[0])
                label = F"{class_name[cls]} {conf}"



                cv2.rectangle(resized_frame(x1,y1),(x2,y2),(0, 255,0),5)


                cv2.putText(resized_frame, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,225, 0), 3)

    return resized_frame

def process_video_frame(model,frame,class_names,display_width):
    """process a single video frame for segmentaition"""
    frame_with_objects = detect_objects(model,frame,class_names,display_width)
    rgb_frame = cv2.cvtColor(frame_with_objects, cv2.COLOR_BGR2RGB)
    return rgb_frame

