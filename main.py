import cv2
import math
import numpy as np
import os
import torch
import cvzone
from sort import *
from ultralytics import YOLO



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
    detection = np.empty((0,5))
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
                #cv2.putText(img, '%s %.2f' % (cls, conf), (x1, y1 - 10),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                #detect
                currentArray = np.array([x1, y1, x2, y2, conf])
                detection = np.vstack((detection, currentArray))
    return img, detection
    

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
 
limits = [400, 297, 673, 297]
totalCount = []    

def main(videoPath, modelName):
    device = checkDevice()  # Check device for running the model
    model = YOLO(modelName).to(device)  # Load model
    video = checkVideo(videoPath)  # Load video
    mask = cv2.imread("mask.png")
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
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
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        imgRegion = cv2.bitwise_and(frame, mask)
        
        
        img = cvzone.overlayPNG(frame, imgGraphics, (0, 0))

        
        results = model(imgRegion, verbose=False) # result list of detections
        
        
        frame, detections = draw_boxes(frame, classes, results)
        

        resultsTracker = tracker.update(detections)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                            scale=2, thickness=3, offset=10)
    
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
 
        # cvzone.putTextRect(frame, f' Count: {len(totalCount)}', (50, 50))
        cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
        # Show
        #cv2.imshow('frame', frame)
        #cv2.imshow('imgRegion', imgRegion)
        cv2.imshow("Image", img)
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