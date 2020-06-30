import cv2
import numpy as np
import dlib
import torch
import time
import argparse
from predict import Classify


def main():
    opts = argparse.ArgumentParser()
    opts.add_argument('--mode', default=False, help='webcam or video', required=False)
    args = opts.parse_args()
    model = Classify()
    face_detector = dlib.get_frontal_face_detector()
    use_webcam = int(args.mode)
    video_path = '/home/phatngy/Downloads/IMG_0314.MOV'
    cam = cv2.VideoCapture(0 if use_webcam else video_path)
    while cam.isOpened():
        tic = time.time()
        _, frm = cam.read()
        
        cv2.imshow('input', frm)
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        for face in faces:
            # print(face)
            # if len(face) >= 1:
            x1, y1, x2, y2 = face.left()-20, face.top()-80, face.right()+30, face.bottom()+40
            # print(x1, y1, x2, y2)
            crop_face = np.zeros((int(x2-x1), int(y2-y1)), dtype=np.uint8)
            crop_face = gray[y1:y2, x1:x2]
            # crop_face
            crop_face = np.concatenate(([crop_face], [crop_face],[crop_face]), axis=0)
            crop_face = np.transpose(crop_face, [1, 2, 0])
            # print(crop_face.shape)
            # cv2.imshow('out', crop_face)
            score, pred = model.predict(crop_face)
            # print(score)
            txt = '{} - {:.2f}'.format(pred, score)
            # print(txt)
            t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            frm = cv2.rectangle(frm, (x1, y1), (x2, y2), (255, 0, 0), 2)
            frm = cv2.putText(frm, txt, (x1,y1+t_size[1]+4),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('output', frm)
        print(f"Time: \t{time.time() - tic}")
        # win.add_overlay(face)

        if cv2.waitKey(1) == 27:
            return
        
if __name__ == '__main__':
    main()