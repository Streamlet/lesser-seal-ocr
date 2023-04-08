# coding: utf-8
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


def cv2read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


def load(path):
    images = []
    categories = []
    labels = []
    images_cache_file = os.path.join(path, 'images.npy')
    categories_cache_file = os.path.join(path, 'categories.npy')
    labels_cache_file = os.path.join(path, 'labels.npy')
    if os.path.exists(images_cache_file) and os.path.exists(
            categories_cache_file) and os.path.exists(labels_cache_file):
        return np.load(images_cache_file), np.load(
            categories_cache_file), np.load(labels_cache_file)
    i = 1
    labels.append(ord('_'))

    for label in os.listdir(path):
        label_dir = os.path.join(path, label)
        if not os.path.isdir(label_dir):
            continue
        print('loading [%d]%s ...' % (i, label))
        labels.append(ord(label))
        for file_name in os.listdir(label_dir):
            if not file_name.endswith('.png'):
                continue
            image = cv2read(os.path.join(label_dir, file_name))
            image_resized = cv2.resize(image, (75, 75))
            images.append(image_resized)
            categories.append(i)
        i = i + 1
    images_np = np.array(images)
    categories_np = np.array(categories)
    labels_np = np.array(labels)
    np.save(images_cache_file, images_np)
    np.save(categories_cache_file, categories_np)
    np.save(labels_cache_file, labels_np)
    return images_np, categories_np, labels_np


def train(images, categories, labels):
    # 将标签进行 one-hot 编码
    categories_one_hot = tf.keras.utils.to_categorical(categories)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(images,
                                                        categories_one_hot,
                                                        test_size=0.2,
                                                        random_state=42)

    # 创建模型
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3),
                            activation='relu',
                            input_shape=(75, 75, 1)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(categories_one_hot.shape[1], activation='softmax')
    ])

    # 编译模型
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # 训练模型
    model.fit(X_train,
              y_train,
              batch_size=32,
              epochs=10,
              validation_data=(X_test, y_test))

    return model


def main():
    if len(sys.argv) <= 1:
        print('usage: train.py <input_dir>')
        return
    img_path = sys.argv[1]

    images, categories, labels = load(img_path)
    model = train(images, categories, labels)
    model.save(os.path.join(img_path, 'trained.h5'))


if __name__ == '__main__':
    main()
