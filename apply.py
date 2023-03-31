# coding: utf-8

import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model



def cv2read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

def main():
    if len(sys.argv) < 2:
        print('usage: apply.py <train_h5_path> <image_path>')
        return
    train_path = sys.argv[1]
    image_path = sys.argv[2]

    model = load_model(train_path)

    image = cv2read(image_path)
    image_resized = cv2.resize(image, (100, 100))
    image_grayed = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    
    X_test = np.array([image_grayed])

    y_test = model.predict(X_test)
    y_test_classes = np.argmax(y_test, axis=1)

    print(chr(y_test_classes[0]))


if __name__ == '__main__':
    main()