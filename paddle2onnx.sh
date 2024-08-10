paddle2onnx --model_dir ./paddle/ch_PP-OCRv4_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./paddle/onnx/ch_PP-OCRv4_det_infer.onnx \
--opset_version 10 \
--enable_onnx_checker True

paddle2onnx --model_dir ./paddle/ch_PP-OCRv4_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./paddle/onnx/ch_PP-OCRv4_rec_infer.onnx \
--opset_version 10 \
--enable_onnx_checker True
