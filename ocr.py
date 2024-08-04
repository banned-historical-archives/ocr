#from paddleocr.tools.infer.predict_system import TextSystem
import sys
import cv2
import os
import json
#from PIL import Image, ImageDraw, ImageFont
#import paddleocr.tools.infer.utility as utility
from utils import draw_ocr_results, start_ocr

with open(sys.argv[1]) as f:
  args = json.load(f)
  print(json.dumps(start_ocr(args)))
