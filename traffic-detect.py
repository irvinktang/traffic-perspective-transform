import numpy as np
import cv2 
import argparse
import imutils
import time
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detection")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load COCO class labels
labelsPath = os.path.sep.join([args["yolo"], "data/coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# assigning random colors to represent classes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "cfg/yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layernames = net.getLayerNames()
layernames = [layernames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize video stream
cap = cv2.VideoCapture(args["input"])
writer = None
width, height = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(cap.get(prop))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1


if not cap.isOpened():
    print("[ERROR] could not open file")
else:
    while True:
        # read next frame in video
        ret, frame = cap.read()

        if not ret:
            break
        
        if width is None or height is None:
            height, width = frame.shape[:2]

        # perform forward pass of YOLO object detector
        # gives bounding boxes and associate probabilities
        # blob = cv2.dnn.blobFromImage(
        #     frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(layernames)
        end = time.time()

        # initialize lists of bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions
                if confidence > args["confidence"]:
                    # YOLO returns the center (x,y) coordinates of the bounding box
                    # followed by the boxes' width and height
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, boxW, boxH) = box.astype("int")

                    # calculate top left corner of bounding box
                    x = int(centerX - (boxW / 2))
                    y = int(centerY - (boxH / 2))

                    boxes.append([x, y, int(boxW), int(boxH)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # nms: basically a sort of filter
        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, args["confidence"], args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract bounding box coordinates
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])

                # draw bounding box and label frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imshow("image", frame)
                cv2.waitKey(1)
        # if writer is None:
        #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #     writer = cv2.VideoWriter(args["output"], fourcc, 30,
        #         (frame.shape[1], frame.shape[0]), True)

        #     # some information on processing single frame
        #     if total > 0:
        #         elap = (end - start)
        #         print("[INFO] single frame took {:.4f} seconds".format(elap))
        #         print("[INFO] estimated total time to finish: {:.4f}".format(
        #             elap * total))

        # # write output
        # writer.write(frame)

print("[INFO] cleaning up...")
writer.release()
cap.release()
