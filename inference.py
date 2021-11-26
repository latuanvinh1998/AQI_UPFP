import torch
from efficientnet_pytorch import EfficientNet
import sys
from torchvision import transforms
from torch import nn
from PIL import Image
from pixellib.semantic import semantic_segmentation
from pixellib.torchbackend.instance import instanceSegmentation
from neurnet import *
from utils import *

import numpy as np
import tensorflow as tf
import argparse
import random
import csv
import json
import os

def inference(args):
	device = "cuda:" + str(args.cuda)
	device_tf = '/gpu:' + str(args.cuda)
	arr = ['PM25_lv1', 'PM25_lv2', 'PM25_lv3', 'PM25_lv4', 'PM25_lv5', 'PM25_lv6']

	transform = transforms.Compose([transforms.Resize((600,600)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
	model_ef = EfficientNet.from_pretrained('efficientnet-b7').to(device)
	neural = Neural_Net(2560, 6).to(device)

	with tf.device(device_tf):
		model_seg = semantic_segmentation()
		model_seg.load_ade20k_model(args.seg_model)
		model_ins = instanceSegmentation()
		model_ins.load_model(args.ins_model)

	neural.load_state_dict(torch.load(args.model_path))
	model_ef.eval()
	neural.eval()

	data_pm25 = []

	f = open(args.mined_pattern_path, 'r')
	files = f.read()
	files = files.split('\n')

	for data in files:
		data = data.split(' ')
		info = data[0:-3]
		info.append(data[-2].replace(']',''))
		data_pm25.append(info)

	img = transform(Image.open(args.input_image)).unsqueeze(0)
	features = extract(model_ef, img.to(device))
	
	with torch.no_grad():
		theta = neural(features)
	label_pm = np.argmax(theta.cpu().numpy())

	with tf.device(device_tf):
		segvalues, objects_masks, image_overlay = model_seg.segmentAsAde20k(args.input_image, extract_segmented_objects=True)
		results, output = model_ins.segmentImage(args.input_image)

	data_seg = []
	data_ins = []
	total_pixel = objects_masks[0]['masks'].shape[0] * objects_masks[0]['masks'].shape[1]

	for objects_mask in objects_masks:
		data_seg.append(objects_mask['class_name'])
		data_seg.append(np.count_nonzero(objects_mask['masks'])/total_pixel)

	for object_ in results['object_counts']:
		data_ins.append(object_)
		data_ins.append(results['object_counts'][object_])

	patterns, score = create_trans(data_seg, data_ins)
	period = get_period(patterns, data_pm25, label_pm, arr)

	print('PM25 level:', label_pm, 'Period:', period)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_image', type=str, default='test.jpg', help="Path to input image")
	parser.add_argument('--cuda', type=int, default=2, help="GPU using")
	parser.add_argument('--model_path', type=str, default='Model/model.pth', help="Path to trained neural model")
	parser.add_argument('--mined_pattern_path', type=str, default='PM_haze_MNR.txt', help="Path to mined patterns")
	parser.add_argument('--ins_model', type=str, default='pointrend_resnet50.pkl', help="Path to Instance Model")
	parser.add_argument('--seg_model', type=str, default='deeplabv3_xception65_ade20k.h5', help="Path to Instance Model")
	args = parser.parse_args()
	inference(args)