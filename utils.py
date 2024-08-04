from pathlib import Path
import cv2
import os
import json
from paddleocr.tools.infer.predict_system import TextSystem
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from typing import Union, Any, Tuple, List, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
import math
import numpy as np
from const import default_config

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

class MyObject:
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

def json_to_class_obj(args):
    return MyObject(args)

def start_ocr(params):
    args = json_to_class_obj({
        **default_config,
        **params,
    })

    # legacy mode
    if args.rec_model_dir == './paddle/ch_ppocr_mobile_v2.0_rec_infer':
        from cnocr import CnOcr, read_img
        import numpy as np
        from cnocr.utils import draw_ocr_results

        ocr = CnOcr(
                det_model_name = 'ch_PP-OCRv3_det',
                #rec_model_name = 'ch_PP-OCRv3',
                rec_model_name = 'ch_ppocr_mobile_v2.0',
                #context = 'cpu',
                rec_model_backend = 'onnx',
                det_model_backend = 'onnx'
        )
        img_fp = args.image_dir
        img = read_img(img_fp)
        res = ocr.ocr(img,
                    resized_shape=1496,
                    preserve_aspect_ratio=True,
                    box_score_thresh=0.3,
                    min_box_size=10
                      )
        return [[[[
            [float(j) for j in [i['position'][0][0] if i['position'][0][0] > 0 else 0,
            i['position'][0][1] if i['position'][0][1] > 0 else 0,]],
            [float(j) for j in [i['position'][1][0] if i['position'][1][0] > 0 else 0,
            i['position'][1][1] if i['position'][1][1] > 0 else 0,]],
            [float(j) for j in [i['position'][2][0] if i['position'][2][0] > 0 else 0,
            i['position'][2][1] if i['position'][2][1] > 0 else 0,]],
            [float(j) for j in [i['position'][3][0] if i['position'][3][0] > 0 else 0,
            i['position'][3][1] if i['position'][3][1] > 0 else 0,]],
            ], [str(i['text']), float(i['score']) if i['score'] > 0 else 0]] for i in res]]

    imgs = []
    if os.path.isdir(args.image_dir):
        for i in os.listdir(args.image_dir):
            imgs.append(cv2.imread(os.path.join(args.image_dir, i)))
    else:
        imgs.append(cv2.imread(args.image_dir))

    ps = TextSystem(args)
    
    res = []
    for img in imgs:
        dt_boxes, rec_res, _ = ps.__call__(img, args.use_angle_cls)

        ocr_res = [[box.tolist(), res]
                   for box, res in zip(dt_boxes, rec_res)]
        
        res.append(ocr_res)
    return res
