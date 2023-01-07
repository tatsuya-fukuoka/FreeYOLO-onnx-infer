#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import logging

import cv2

from utils.utils import FREEYOLOONNX


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument("--mode", type=str, default="image",
                        help="Infer mode")
    parser.add_argument("--model", type=str, default="model/yolo_free_tiny_opset_11.onnx",
                        help="Input your onnx model.")
    parser.add_argument("-i", "--input_path", type=str, default='test.jpg',
                        help="Path to your input image.")
    parser.add_argument("-o", "--output_dir", type=str, default='outputs',
                        help="Path to your output directory.")
    parser.add_argument("-s", "--score_thr", type=float, default=0.35,
                        help="Score threshould to filter the result.")
    parser.add_argument("-size", "--img_size", type=int, default=640,
                        help="Specify an input shape for inference.")
    return parser


def infer_img(args, freeyolo_onnx):
    #image get
    origin_img = cv2.imread(args.input_path)

    # infer
    t0 = time.time()
    result_img = freeyolo_onnx.infer(origin_img)
    logging.info(f'Infer time: {(time.time()-t0)*1000:.2f} [ms]')

    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))
    cv2.imwrite(output_path, result_img)

    logging.info(f'save_path: {output_path}')
    logging.info(f'Inference Finish!')


def infer_vid(args, freeyolo_onnx):
    cap = cv2.VideoCapture(args.input_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir,os.path.basename(args.input_path))

    writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    frame_id = 1
    while True:
        ret_val, origin_img = cap.read()
        if not ret_val:
            break
        
        t0 = time.time()
        result_img = freeyolo_onnx.infer(origin_img)
        logging.info(f'Frame: {frame_id}/{frame_count}, Infer time: {(time.time()-t0)*1000:.2f} [ms]')
        
        writer.write(result_img)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        frame_id+=1
        
    writer.release()
    cv2.destroyAllWindows()
    
    logging.info(f'save_path: {save_path}')
    logging.info(f'Inference Finish!')
    

if __name__ == '__main__':
    args = make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")

    freeyolo_onnx = FREEYOLOONNX(
        args.model,
        args.img_size,
        args.score_thr
    )

    if args.mode == "image":
        infer_img(args, freeyolo_onnx)
    else:
        infer_vid(args, freeyolo_onnx)
    
    

    