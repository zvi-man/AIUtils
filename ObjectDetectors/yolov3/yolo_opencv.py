#############################################
# Object detection via RTSP - YOLO - OpenCV
# Author : Frank Schmitz   (Dec 11, 2018)
# Website : https://www.github.com/zishor
############################################

import os
import os.path
import cv2
import argparse
import numpy as np
import imageio_ffmpeg as imageio
import datetime

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=False,
                help='path to input image', default='sampledata/commuters.mp4')
ap.add_argument('-o', '--outputfile', required=False,
                help='filename for output video', default='output.mp4')
ap.add_argument('-od', '--outputdir', required=False,
                help='path to output folder', default='output')
ap.add_argument('-fs', '--framestart', required=False,
                help='start frame', default=0)
ap.add_argument('-fl', '--framelimit', required=False,
                help='number of frames to process (0 = all)', default=0)
ap.add_argument('-c', '--config', required=False,
                help='path to yolo config file', default='cfg/yolov3-tiny.cfg')
ap.add_argument('-w', '--weights', required=False,
                help='path to yolo pre-trained weights', default='yolov3-tiny.weights')
ap.add_argument('-cl', '--classes', required=False,
                help='path to text file containing class names', default='cfg/yolov3.txt')
ap.add_argument('-ic', '--invertcolor', required=False,
                help='invert RGB 2 BGR', default='False')
ap.add_argument('-fpt', '--fpsthrottle', required=False,
                help='skips (int) x frames in order to catch up with the stream for slow machines 1 = no throttle',
                default=1)
args = ap.parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_output_layers(net):
    layer_names = net.getLayerNames()

    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def save_bounded_image(image, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    dirname = os.path.join(args.outputdir, label, datetime.datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = label + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f') + '_conf' + "{:.2f}".format(
        confidence) + '.jpg'
    print('Saving bounding box:' + filename)
    roi = image[y:y_plus_h, x:x_plus_w]
    if roi.any():
        if str2bool(args.invertcolor) == False:
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dirname, filename), roi)


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 3)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)


def detect(image):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    orgImage = image.copy()
    for i in indices:
        # i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        save_bounded_image(orgImage, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    if str2bool(args.invertcolor) == True:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def processvideo(file):
    cap = cv2.VideoCapture(file)

    writer = imageio.write_frames(args.outputfile, (int(cap.get(3)), int(cap.get(4))))
    writer.send(None)
    frame_counter = 0
    while (cap.isOpened()):
        frame_counter = frame_counter + 1
        ret, frame = cap.read()
        print('Detecting objects in frame ' + str(frame_counter))
        if ret == True:
            if not frame is None:
                image = detect(frame)
                writer.send(frame)
            else:
                print('Frame error in frame ' + str(frame_counter))
        else:
            break
    cap.release()
    writer.close()


# Doing some Object Detection on a video
classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

if args.input.startswith('rtsp'):

    cap = cv2.VideoCapture(args.input)
    if int(args.framelimit) > 0:
        writer = imageio.write_frames(args.outputfile, (int(cap.get(3)), int(cap.get(4))))
        writer.send(None)
    frame_counter = 0
    while (True):
        if int(args.framelimit) > 0 and frame_counter > int(args.framestart) + int(args.framelimit):
            writer.close()
            break

        if frame_counter % int(args.fpsthrottle) == 0:
            ret, frame = cap.read()
            if ret and frame_counter >= int(args.framestart):
                print('Detecting objects in frame ' + str(frame_counter))
                frame = detect(frame)
                if int(args.framelimit) > 0:
                    writer.send(frame)
            else:
                print('Skipping frame ' + str(frame_counter))
        else:
            print('FPS throttling. Skipping frame ' + str(frame_counter))
        frame_counter = frame_counter + 1

else:
    if os.path.isdir(args.input):
        for dirpath, dirnames, filenames in os.walk(args.input):
            for filename in [f for f in filenames if f.endswith(".mp4")]:
                print('Processing video ' + os.path.join(dirpath, filename))
                processvideo(os.path.join(dirpath, filename))
    else:
        processvideo(os.path.join(args.input))
