import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask2.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsPlus = [450, 200, 673, 200]
limitsMinus = [10, 600, 673, 600]
totalCountIn = []
totalCountOut = []
total = 0

# bbox = [300, 300, 673, 300]
# objek_di_kotak = 0

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    print(detections)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # if bbox[0] <= x1 and bbox[1] <= y1 and bbox[2] >= x2 and bbox[3] >= y2:
            #     # Objek terletak sepenuhnya dalam kotak, tambahkan ke hitungan
            #     objek_di_kotak += 1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]


            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    # cv2.line(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 3)
    cv2.line(img, (limitsPlus[0], limitsPlus[1]), (limitsPlus[2], limitsPlus[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsMinus[0], limitsMinus[1]), (limitsMinus[2], limitsMinus[3]), (0, 255, 0), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=2, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsPlus[0] < cx < limitsPlus[2] and limitsPlus[1] - 15 < cy < limitsPlus[1] + 15:
            if id not in totalCountIn:  # Periksa apakah ID belum ada dalam totalCount sebelum menambahkannya
                totalCountIn.append(id)
            cv2.line(img, (limitsPlus[0], limitsPlus[1]), (limitsPlus[2], limitsPlus[3]), (0, 255, 0), 5)

        if limitsMinus[0] < cx < limitsMinus[2] and limitsMinus[1] - 15 < cy < limitsMinus[1] + 15:
            if id not in totalCountOut:  # Periksa apakah ID ada dalam totalCount sebelum menghapusnya
                totalCountOut.append(id)
            cv2.line(img, (limitsMinus[0], limitsMinus[1]), (limitsMinus[2], limitsMinus[3]), (0, 0, 255), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    total = len(totalCountIn)-len(totalCountOut)
    # print(total)
    cv2.putText(img,str(total),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)


    # cv2.putText(img, f'Objek di dalam kotak: {objek_di_kotak}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
