from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from typing import Union, Any, Tuple, List, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
import math
import numpy as np
def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     show_score=False,
                     drop_score=0.5,
                     font_path="./fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon([(i[0], i[1]) for i in box], fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if show_score:
            txt = txt + ':' + str(score)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_bbox = font.getbbox(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_bbox[3] - char_bbox[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)

def draw_ocr_results(image_fp: Union[str, Path, Image.Image], ocr_outs, font_path):
    # Credits: adapted from https://github.com/PaddlePaddle/PaddleOCR
    import cv2

    if isinstance(image_fp, (str, Path)):
        img = Image.open(image_fp).convert('RGB')
    else:
        img = image_fp

    txts = []
    scores = []
    boxes = []
    for _out in ocr_outs:
        txts.append(_out[1][0])
        scores.append(_out[1][1])
        boxes.append(_out[0])

    draw_img = draw_ocr_box_txt(
        img, boxes, txts, scores, drop_score=0.0, font_path=font_path
    )

    # cv2.imwrite(out_draw_fp, draw_img[:, :, ::-1])

    plt.figure(figsize=(draw_img.shape[0]/100, draw_img.shape[1]/100), dpi=100)
    plt.imshow(draw_img)
    plt.axis('off')  # Turn off axis numbers
    plt.show()
