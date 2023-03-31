# coding: utf-8

import os, sys
import cv2
import numpy as np


def cv2read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


def cv2save(img, path):
    _, buf = cv2.imencode('.png', img)
    buf.tofile(path)


def transform(char, source_path, out_dir):
    target_dir = os.path.join(out_dir, char)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    img = cv2read(source_path)
    # 原图
    cv2save(img, os.path.join(target_dir, 'original.png'))
    # 旋转 -30 度到 +30 度
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    for angle in range(-30, 30):
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 进行旋转变换
        rotated_img = cv2.warpAffine(img, M, (width, height))
        cv2save(rotated_img, os.path.join(target_dir, 'rotate_%d.png' % angle))
    # 梯形扭曲
    # 原始图片的四个角点坐标
    src_pts = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]],
                          [img.shape[1], img.shape[0]]])
    for i in range(1, int(img.shape[1] * 0.2)):
        # 扭曲后的图片的四个角点坐标（上边变小）
        dst_pts = np.float32([[i, 0], [img.shape[1] - i, 0], [0, img.shape[0]],
                              [img.shape[1], img.shape[0]]])
        # 生成变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # 进行梯形扭曲变换
        distorted_img = cv2.warpPerspective(img, M,
                                            (img.shape[1], img.shape[0]))
        cv2save(distorted_img,
                os.path.join(target_dir, 'distorte_up_%d.png' % i))
        # 扭曲后的图片的四个角点坐标（下边变小）
        dst_pts = np.float32([[0, 0], [img.shape[1], 0], [i, img.shape[0]],
                              [img.shape[1] - i, img.shape[0]]])
        # 生成变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # 进行梯形扭曲变换
        distorted_img = cv2.warpPerspective(img, M,
                                            (img.shape[1], img.shape[0]))
        cv2save(distorted_img,
                os.path.join(target_dir, 'distorte_down_%d.png' % i))
    for i in range(1, int(img.shape[0] * 0.2)):
        # 扭曲后的图片的四个角点坐标（左边变小）
        dst_pts = np.float32([[0, i], [img.shape[1], 0], [0, img.shape[0] - i],
                              [img.shape[1], img.shape[0]]])
        # 生成变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # 进行梯形扭曲变换
        distorted_img = cv2.warpPerspective(img, M,
                                            (img.shape[1], img.shape[0]))
        cv2save(distorted_img,
                os.path.join(target_dir, 'distorte_left_%d.png' % i))
        # 扭曲后的图片的四个角点坐标（右边变小）
        dst_pts = np.float32([[0, 0], [img.shape[1], i], [0, img.shape[0]],
                              [img.shape[1], img.shape[0] - i]])
        # 生成变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # 进行梯形扭曲变换
        distorted_img = cv2.warpPerspective(img, M,
                                            (img.shape[1], img.shape[0]))
        cv2save(distorted_img,
                os.path.join(target_dir, 'distorte_right_%d.png' % i))


def main():
    if len(sys.argv) < 3:
        print('usage: img_transform.py <input_dir> <output_dir>')
        return
    img_path = sys.argv[1]
    out_dir = sys.argv[2]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for f in os.listdir(img_path):
        full_path = os.path.join(img_path, f)
        (char, ext) = os.path.splitext(f)
        if not os.path.isfile(full_path) or ext != '.png':
            continue
        unicode = ord(char)
        if unicode >= 0x4E00 and unicode <= 0x9FFF:
            transform(char, full_path, out_dir)


if __name__ == '__main__':
    main()