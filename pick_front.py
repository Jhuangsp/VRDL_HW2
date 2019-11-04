import cv2
import numpy as np
import os
import time
import glob

DATA_PATH = 'D:/dataset/CelebA/Img/img_align_celeba/'


def main():

    front_list = []
    train_lits = glob.glob(DATA_PATH + '*.jpg')
    allimg = len(train_lits)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    count = 0
    with open('front_face.txt', 'w') as f:
        for i, imagename in enumerate(train_lits):
            image = cv2.imread(imagename)
            faces = face_cascade.detectMultiScale(image, 1.1, 20)
            if i % 2000 == 0:
                print('{}/{}'.format(count, allimg))
            if len(faces) != 0:
                # print('Front')
                count += 1
                f.write("{}\n".format(imagename))

            else:
                # print('side')
                pass
            # cv2.imshow('123', image)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     os._exit(0)


if __name__ == '__main__':
    main()
