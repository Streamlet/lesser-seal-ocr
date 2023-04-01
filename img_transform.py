# coding: utf-8

import os, sys
import cv2
import numpy as np


def cv2read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


def cv2save(img, path):
    _, buf = cv2.imencode('.png', img)
    buf.tofile(path)


def save(img, target_dir, prefix, name):
    img_resized = cv2.resize(img, (75, 75))
    cv2save(img_resized, os.path.join(target_dir, prefix + name + '.png'))


def save_with_bg(img, target_dir, prefix, name):
    img_resized = cv2.resize(img, (75, 75))
    save(img_resized, target_dir, prefix, name)


def transform(char, source_path, out_dir, prefix):
    target_dir = os.path.join(out_dir, char)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    img = cv2read(source_path)
    # 原图
    save(img, target_dir, prefix, 'original')
    # 旋转 -30 度到 +30 度
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    for angle in range(-30, 30, 5):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (width, height))
        save_with_bg(rotated_img, target_dir, prefix, 'rotate_%d' % angle)

    # 梯形扭曲
    src_pts = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]],
                          [img.shape[1], img.shape[0]]])
    for i in range(1, int(img.shape[1] * 0.3), 5):
        dst_pts = np.float32([[i, 0], [img.shape[1] - i, 0], [0, img.shape[0]],
                              [img.shape[1], img.shape[0]]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        distorted_img = cv2.warpPerspective(img, M,
                                            (img.shape[1], img.shape[0]))
        save_with_bg(distorted_img, target_dir, prefix, 'distorte_up_%d.g' % i)

        dst_pts = np.float32([[0, 0], [img.shape[1], 0], [i, img.shape[0]],
                              [img.shape[1] - i, img.shape[0]]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        distorted_img = cv2.warpPerspective(img, M,
                                            (img.shape[1], img.shape[0]))
        save_with_bg(distorted_img, target_dir, prefix, 'distorte_down_%d' % i)

    for i in range(1, int(img.shape[0] * 0.3), 5):
        dst_pts = np.float32([[0, i], [img.shape[1], 0], [0, img.shape[0] - i],
                              [img.shape[1], img.shape[0]]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        distorted_img = cv2.warpPerspective(img, M,
                                            (img.shape[1], img.shape[0]))
        save_with_bg(distorted_img, target_dir, prefix, 'distorte_left_%d' % i)

        dst_pts = np.float32([[0, 0], [img.shape[1], i], [0, img.shape[0]],
                              [img.shape[1], img.shape[0] - i]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        distorted_img = cv2.warpPerspective(img, M,
                                            (img.shape[1], img.shape[0]))
        save_with_bg(distorted_img, target_dir, prefix,
                     'distorte_right_%d.png' % i)


def main():
    if len(sys.argv) < 3:
        print('Usage: img_transform.py <input_dir> <output_dir> <prefix>')
        return
    [img_path, out_dir, prefix] = sys.argv[1:4]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not prefix.endswith('_'):
        prefix = prefix + '_'
    for f in os.listdir(img_path):
        full_path = os.path.join(img_path, f)
        (char, ext) = os.path.splitext(f)
        if not os.path.isfile(full_path) or ext != '.png':
            continue
        unicode = ord(char)
        if unicode < 0x4E00 or unicode > 0x9FFF:
            continue
        transform(char, full_path, out_dir, prefix)


if __name__ == '__main__':
    main()