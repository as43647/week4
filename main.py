import cv2
import math
import numpy as np
import os
import torch


from ultralytics import YOLO
# from utils.sort import Sort


def checkDevice():
    # Test cuda availability
    try:
        torch.cuda.is_available()
    except:
        device = 'cpu'
    else:
        device = 'cuda:0'
    finally:
        print('Running on %s' % device)
        return device
    
def checkVideo(videoPath):
    if not os.path.exists(videoPath):
        print('Video not found')
        exit()
    else:
        video = cv2.VideoCapture(videoPath)
        return video



def draw_boxes(img, className, pred, color=(255, 0, 255)):
    for result in pred:
        for box in result.boxes:
            # Get the coordinates of the box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert to int
            w, h = x2 - x1, y2 - y1
            # Get the confidence score
            conf = math.ceil(box.conf[0] * 100) / 100
            # Get the predicted class label
            cls = className[int(box.cls[0])]

            if (cls == 'car' or cls == 'truck' or cls == 'bus') and conf > 0.3:
                # Draw the box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # Draw the label
                cv2.putText(img, '%s %.2f' % (cls, conf), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img
    

def main(videoPath, modelName):
    device = checkDevice()  # Check device for running the model
    model = YOLO(modelName).to(device)  # Load model
    video = checkVideo(videoPath)  # Load video
    classes = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]  # class list for COCO dataset
    
    # Loop
    while True:
        success, frame = video.read()  # Read frame
        if not success:
            break

        # Detect
        results = model(frame,verbose=False) # result list of detections

        # Draw
        frame = draw_boxes(frame, classes, results)

        # Show
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close all windows
    cv2.destroyAllWindows()
    os.system('clear')


if __name__ == '__main__': 
    videoPath = 'cars.mp4'
    modelName = 'yolov8n.pt'
    main(videoPath, modelName)