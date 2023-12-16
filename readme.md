# v3 -onnx
python3 ~/.local/lib/python3.10/site-packages/paddleocr/tools/infer/predict_det.py --image_dir="./test.jpg"   --det_limit_type=max --det_limit_side_len=2436 --use_onnx=True --det_model_dir="./paddle/onnx/ch_PP-OCRv3_det_infer.onnx"

# v3
python3 ~/.local/lib/python3.10/site-packages/paddleocr/tools/infer/predict_det.py --image_dir="./test.jpg" --det_model_dir="./paddle/ch_PP-OCRv3_det_infer"  --det_limit_type=max --det_limit_side_len=1436

# v4
python3 ~/.local/lib/python3.10/site-packages/paddleocr/tools/infer/predict_det.py --image_dir="./test.jpg" --det_model_dir="./paddle/ch_PP-OCRv4_det_infer"  --det_limit_type=max --det_limit_side_len=1436

# v4 server
python3 ~/.local/lib/python3.10/site-packages/paddleocr/tools/infer/predict_det.py --image_dir="./test.jpg" --det_model_dir="./paddle/ch_PP-OCRv4_det_server_infer"  --det_limit_type=max --det_limit_side_len=1436
