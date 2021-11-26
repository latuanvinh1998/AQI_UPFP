import pixellib
from pixellib.semantic import semantic_segmentation
from pixellib.torchbackend.instance import instanceSegmentation
from utils import *

import numpy as np
import argparse
import tensorflow as tf
import csv
import os

def extract(args):
	device = '/gpu:' + args.cuda
	with tf.device(device):
		model = semantic_segmentation()
		model.load_ade20k_model(args.seg_model)

	os.makedirs(args.output_path, exist_ok=True)

	path = args.path
	origin_dirs = os.listdir(path)

	with open(os.path.join(args.output_path, 'semantic.csv'), 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f)

		for img_name in origin_dirs:
			
			data = []

			img_path = os.path.join(path, img_name)
			with tf.device(device):
				segvalues, objects_masks, image_overlay = model.segmentAsAde20k(img_path, extract_segmented_objects=True)

			total_pixel = objects_masks[0]['masks'].shape[0] * objects_masks[0]['masks'].shape[1]
			data.append(img_name)

			for objects_mask in objects_masks:
				data.append(objects_mask['class_name'])
				data.append(np.count_nonzero(objects_mask['masks'])/total_pixel)
			writer.writerow(data)

		f.close()

	with tf.device(device):
		model = instanceSegmentation()
		model.load_model(args.ins_model)

	with open(os.path.join(args.output_path, 'instance.csv'), 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f)
		for img_name in origin_dirs:

			data = []

			img_path = os.path.join(path, img_name)
			with tf.device(device):
				results, output = model.segmentImage(img_path)

			data.append(img_name)

			for object_ in results['object_counts']:
				data.append(object_)
				data.append(results['object_counts'][object_])

			writer.writerow(data)

		f.close()

def generate_file(args):
	labels_instance = []
	label_segment = []
	datas_txt = []
	array = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]


	with open(os.path.join(args.output_path, 'semantic.csv'), newline='') as f:
		rows = csv.reader(f)
		for row in rows:
			label_segment.append(row)

	with open(os.path.join(args.output_path, 'instance.csv'), newline='') as f:
		rows = csv.reader(f)
		for row in rows:
			labels_instance.append(row)

	with open(os.path.join(args.output_path,'data.csv'), 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f)
		for i in range(len(labels_instance)):
			data = [labels_instance[i][0]]
			total_ = total(labels_instance[i])

			for j in range(int((len(labels_instance[i]) - 1) / 2)):

				if labels_instance[i][j*2 + 1] in array:
					item = labels_instance[i][j*2 + 1]
					score = float(labels_instance[i][(j + 1)*2])/total_
					infos = ins_to_label(score, item)

					for info in infos:
						data.append(info)

			for j in range(int((len(label_segment[i]) - 1) / 2)):
				item = label_segment[i][j*2 + 1]
				score = label_segment[i][(j + 1)*2]
				infos = seg_to_label(score, item)

				for info in infos:
					data.append(info)
			writer.writerow(data)
	f.close()

	datas = []
	with open(os.path.join(args.output_path,'data.csv'), newline='') as f:
		rows = csv.reader(f)
		for row in rows:
			datas.append(row)

	f = open(args.pm25, 'r', encoding="utf8", errors='ignore')

	datas_ = f.read()
	datas_ = datas_.split('\n')

	for data in datas_:
		data = data.split(',')
		datas_txt.append(data)

	with open(os.path.join(args.output_path,'data.csv'), 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f)

		for data_txt in datas_txt:

			for data in datas:
				img_name = data[0]
				if img_name == data_txt[0]:
					info = []
					for i in range(len(data)):
						info.append(data[i])
					info.append(data_txt[1])
					writer.writerow(info)
					break

	if args.haze == True:
		datas = []
		with open(os.path.join(args.output_path,'data.csv'), newline='') as f:
			rows = csv.reader(f)
			for row in rows:
				datas.append(row)
		f = open(args.haze_file, 'r')
		datas_haze = f.read().split('\n')
		print(datas_haze)

		with open(os.path.join(args.output_path,'data.csv'), 'w', encoding='UTF8', newline='') as f:
			writer = csv.writer(f)

			for data in datas:
				for data_haze in datas_haze:
					data_haze = data_haze.split('\t')
					idx = data_haze[0].rfind('/') + 1

					if data[0] == data_haze[0][idx:]:
						info = []
						for i in range(len(data) - 1):
							info.append(data[i])
						
						idx_start = data_haze[1].find('(')
						idx_end = data_haze[1].find(')')

						info.append(data_haze[1][:idx_start])
						info.append(data_haze[1][idx_start + 1:idx_end])
						info.append(data[-1])
						writer.writerow(info)

						break

	datas_csv = []
	idx = 0
	txt = open(os.path.join(args.output_path,'data.csv').replace('csv', 'txt'), 'w')

	with open(os.path.join(args.output_path,'data.csv'), newline='') as f:
		rows = csv.reader(f)

		for row in rows:
			txt.write(str(idx) + '\t')

			for i in range(1, len(row) - 1):
				if i%2 == 1:
					txt.write(row[i].replace(' ','_') + '(')
				else:
					txt.write(row[i] + ')\t')

			info = fuzzy_to_label(row[len(row) - 1])
			
			txt.write(info[0] + '(' + str(info[1])+ ')' + '\n')

			idx += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default='Data_Img', help="Path to txt file")
	parser.add_argument('--ins_model', type=str, default='pointrend_resnet50.pkl', help="Path to Instance Model")
	parser.add_argument('--seg_model', type=str, default='deeplabv3_xception65_ade20k.h5', help="Path to Instance Model")
	parser.add_argument('--output_path', type=str, default='Data', help="Output path")
	parser.add_argument('--pm25', type=str, default='Data/pm25.txt', help="Path to pm25 file, with txt format")
	parser.add_argument('--haze', type=bool, default=False, help="Path to haze file, with txt format")
	parser.add_argument('--haze_file', type=str, default='Data/haze.txt', help="Path to haze file, with txt format")
	#parser.add_argument('--dataset_path', required=True, help="Path to train dataset folder")
	parser.add_argument('--cuda', type=str, default='0', help="Cuda device using")
	args = parser.parse_args()
	# extract(args)
	generate_file(args)