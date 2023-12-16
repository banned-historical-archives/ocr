default_config = {
    'use_gpu': False,
    'use_xpu': False,
    'use_npu': False,
    'ir_optim': True,
    'use_tensorrt': False,
    'min_subgraph_size': 15,
    'precision': 'fp32',
    'gpu_mem': 500,
    'gpu_id': 0,

    'use_onnx': False,
    "page_num": 0,
    "det_algorithm": 'DB',
    'det_limit_side_len': 960,
    'det_limit_type': "max",
    "det_box_type": 'quad',

    # DB parmas
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.6,
    "det_db_unclip_ratio": 1.5,
    "max_batch_size": 10,
    "use_dilation": False,
    "det_db_score_mode": "fast",

    # EAST parmas
    "det_east_score_thresh": 0.8,
    "det_east_cover_thresh": 0.1,
    "det_east_nms_thresh": 0.2,

    # SAST parmas
    "det_sast_score_thresh": 0.5,
    "det_sast_nms_thresh": 0.2,

    # PSE parmas
    "det_pse_thresh": 0,
    "det_pse_box_thresh": 0.85,
    "det_pse_min_area": 16,
    "det_pse_scale": 1,

    # FCE parmas
    "scales": [8, 16, 32],
    "alpha": 1.0,
    "beta": 1.0,
    "fourier_degree": 5,

    # params for text recognizer
    'rec_algorithm': 'SVTR_LCNet',
    'rec_image_inverse': True,
    'rec_image_shape': "3, 48, 320",
    'rec_batch_num': 6,
    'max_text_length': 25,
    "rec_char_dict_path": './paddle/ppocr_keys_v1.txt',
    'use_space_char': True,
    # 'vis_font_path': "./doc/fonts/simfang.ttf",
    "drop_score": 0.5,

    # params for e2e
    # parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    # parser.add_argument("--e2e_model_dir", type=str)
    # parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    # parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    # parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    # parser.add_argument(
    #    "--e2e_char_dict_path", type=str, default="./ppocr/utils/ic15_dict.txt")
    # parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    # parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # params for text classifier
    "use_angle_cls": False,
    # "cls_model_dir"
    "cls_image_shape": "3, 48, 192",
    "label_list": ['0', '180'],
    "cls_batch_num": 6,
    "cls_thresh": 0.9,

    "enable_mkldnn": False,
    "cpu_threads": 10,
    "use_pdserving": False,
    "warmup": False,

    # SR parmas
    "sr_image_shape": "3, 32, 128",
    "sr_batch_num": 1,

    "draw_img_save_dir": "./inference_results",
    "save_crop_res": False,
    "crop_res_save_dir": "./output",

    # multi-process
    "use_mp": False,
    "total_process_num": 1,
    "process_id": 0,

    "benchmark": False,
    "save_log_path": "./log_output/",

    "show_log": False,

    
}
