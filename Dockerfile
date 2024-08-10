FROM python:3.10

RUN pip install "paddlepaddle==2.5.2"
RUN pip install "paddleocr==2.7.0.3"
RUN pip install "onnxruntime==1.18.1"

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install "numpy<2"

COPY ./paddle/ch_PP-OCRv3_rec_infer ./paddle/ch_PP-OCRv3_rec_infer
COPY ./paddle/ch_PP-OCRv3_det_infer ./paddle/ch_PP-OCRv3_det_infer
COPY ./paddle/ch_PP-OCRv4_rec_infer ./paddle/ch_PP-OCRv4_rec_infer
COPY ./paddle/ch_PP-OCRv4_det_infer ./paddle/ch_PP-OCRv4_det_infer
COPY ./paddle/onnx ./paddle/onnx
COPY ./paddle/ppocr_keys_v1.txt ./paddle/ppocr_keys_v1.txt
COPY ./const.py ./
COPY ./utils.py ./
COPY ./ocr.py ./
