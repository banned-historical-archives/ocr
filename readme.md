# v3 -onnx
python3 ~/.local/lib/python3.10/site-packages/paddleocr/tools/infer/predict_det.py --image_dir="./test.jpg"   --det_limit_type=max --det_limit_side_len=2436 --use_onnx=True --det_model_dir="./paddle/onnx/ch_PP-OCRv3_det_infer.onnx"

# v3
python3 ~/.local/lib/python3.10/site-packages/paddleocr/tools/infer/predict_det.py --image_dir="./test.jpg" --det_model_dir="./paddle/ch_PP-OCRv3_det_infer"  --det_limit_type=max --det_limit_side_len=1436

# v4
python3 ~/.local/lib/python3.10/site-packages/paddleocr/tools/infer/predict_det.py --image_dir="./test.jpg" --det_model_dir="./paddle/ch_PP-OCRv4_det_infer"  --det_limit_type=max --det_limit_side_len=1436

# v4 server
python3 ~/.local/lib/python3.10/site-packages/paddleocr/tools/infer/predict_det.py --image_dir="./test.jpg" --det_model_dir="./paddle/ch_PP-OCRv4_det_server_infer"  --det_limit_type=max --det_limit_side_len=1436

# test
* 使用cpu计算多张图片，多次调用，单次调用多实例，单次调用单实例耗时区别不大
python3 ocr.py '{    "image_dir": "./imgs/3.png",    "drop_score": 0.3,    "det_model_dir": "./paddle/ch_PP-OCRv4_det_infer",    "rec_model_dir": "./paddle/ch_PP-OCRv4_rec_infer",    "det_limit_side_len": 2496, "det_db_box_thresh": 0.3,    "use_angle_cls": false }'

# docker compose
docker-compose up -d --build

# docker test
docker run -it -v imgs:/app/imgs ocr_test_ocr /bin/bash