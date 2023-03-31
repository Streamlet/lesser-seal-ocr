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

    rates = y_test[0]

    class_0 = np.argmax(rates)
    rate_0 = rates[class_0]
    rates = np.delete(rates, class_0)

    class_1 = np.argmax(rates)
    rate_1 = rates[class_1]
    rates = np.delete(rates, class_1)

    class_2 = np.argmax(rates)
    rate_2 = rates[class_2]
    rates = np.delete(rates, class_2)

    print('1.%s(%.2f%%); 2.%s(%.2f%%); 3.%s(%.2f%%)' % (
        chr(class_0 + 0x4E00), float(rate_0 * 100),
        chr(class_1 + 0x4E00), float(rate_1 * 100),
        chr(class_2 + 0x4E00), float(rate_2 * 100)))


if __name__ == '__main__':
    main()
