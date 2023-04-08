# coding: utf-8

import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def cv2read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


def main():
    if len(sys.argv) < 4:
        print(
            'usage: predict.py <trained_h5_path> <labels_npy_path> <image_path>'
        )
        return
    [trained_path, labels_path, image_path] = sys.argv[1:4]

    model = load_model(trained_path)
    labels = np.load(labels_path)

    image = cv2read(image_path)
    image_resized = cv2.resize(image, (75, 75))
    image_grayed = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

    X_test = np.array([image_grayed])
    y_test = model.predict(X_test)
    print(y_test)

    rates = y_test[0]

    category_0 = np.argmax(rates)
    rate_0 = rates[category_0]
    rates[category_0] = -1

    category_1 = np.argmax(rates)
    rate_1 = rates[category_1]
    rates[category_1] = -1

    category_2 = np.argmax(rates)
    rate_2 = rates[category_2]
    rates[category_2] = -1

    print(
        '1.%s(%.2f%%); 2.%s(%.2f%%); 3.%s(%.2f%%)' %
        (chr(labels[category_0]), float(rate_0 * 100), chr(labels[category_1]),
         float(rate_1 * 100), chr(labels[category_2]), float(rate_2 * 100)))


if __name__ == '__main__':
    main()
