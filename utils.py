import numpy as np
import torch

from torch import nn

import numpy as np

def convert_seg(value):
	if value <= 0.125:
		return 1, 0, 0
	elif value >= 0.425:
		return 0, 0, 1

	elif value > 0.125 and value < 0.275:
		low = 1 - (value - 0.125)/0.15
		med = (value - 0.125)/0.15
		return low, med ,0

	elif value >= 0.275 and value < 0.425:
		med = 1 - (value - 0.275)/0.15
		high = (value - 0.275)/0.15
		return 0, med, high

def convert_ins(value): # 0.1 - 0.25 - 0.4 - 0.55
	if value <= 0.175:
		return 1, 0, 0
	elif value >= 0.475:
		return 0, 0, 1

	elif value > 0.175 and value < 0.325:
		low = 1 - (value - 0.175)/0.15
		med = (value - 0.175)/0.15
		return low, med ,0

	elif value >= 0.325 and value < 0.475:
		med = 1 - (value - 0.325)/0.15
		high = (value - 0.325)/0.15
		return 0, med, high

def fuzzy_pm(value):
	if value <= 7:
		return 1, 0, 0, 0, 0, 0
	elif value >= 260.5:
		return 0, 0, 0, 0, 0, 1

	elif value > 7 and value < 17:
		lv1 = 1 - (value - 7)/10
		lv2 = (value - 7)/10
		return lv1, lv2, 0, 0, 0, 0

	elif value >= 17 and value < 45.5:
		lv2 = 1 - (value - 17)/28.5
		lv3 = (value - 17)/28.5
		return 0, lv2, lv3, 0, 0, 0

	elif value >= 45.5 and value < 103:
		lv3 = 1 - (value - 45.5)/57.5
		lv4 = (value - 45.5)/57.5
		return 0, 0, lv3, lv4, 0, 0

	elif value >= 103 and value < 200.5:
		lv4 = 1 - (value - 103)/97.5
		lv5 = (value - 103)/97.5
		return 0, 0, 0, lv4, lv5, 0

	elif value >= 200.5 and value < 260.5:
		lv5 = 1 - (value - 200.5)/60
		lv6 = (value - 200.5)/60
		return 0, 0, 0, 0, lv5, lv6

def ins_to_label(value, item):
	value = float(value)
	label = []

	if value < 0.1:
		return label

	low ,med, high = convert_ins(value)

	if low > med and low > high and low > 0:
		label.append('R_low_' + item)
		label.append(low)

	elif med > low and med > high and med > 0:
		label.append('R_med_' + item)
		label.append(med)

	elif high > low and high > med and high > 0:
		label.append('R_high_' + item)
		label.append(high)

	elif low > 0 and low == med:
		label.append('R_med_' + item)
		label.append(med)

	elif high > 0 and high == med:
		label.append('R_high_' + item)
		label.append(high)

	return label


def seg_to_label(value, item):
	value = float(value)
	label = []

	if value < 0.05:
		return label

	low ,med, high = convert_seg(value)

	if low > med and low > high and low > 0:
		label.append('S_low_' + item)
		label.append(low)

	elif med > low and med > high and med > 0:
		label.append('S_med_' + item)
		label.append(med)

	elif high > low and high > med and high > 0:
		label.append('S_high_' + item)
		label.append(high)

	elif low > 0 and low == med:
		label.append('S_med_' + item)
		label.append(med)

	elif high > 0 and high == med:
		label.append('S_high_' + item)
		label.append(high)

	return label

def fuzzy_to_label(value):
	value = float(value)
	label = []
	lv1, lv2, lv3, lv4, lv5, lv6 = fuzzy_pm(value)
	arr = np.array([lv1, lv2, lv3, lv4, lv5, lv6])
	idx = np.argmax(arr)

	if idx < 5 and arr[idx] == arr[idx + 1]:
		label.append('PM25_lv' + str(idx + 2))
		label.append(arr[idx])

	else:
		label.append('PM25_lv' + str(idx + 1))
		label.append(arr[idx])

	return label


def extract(model_ef, img):
	with torch.no_grad():
		feature = model_ef.extract_features(img)
		feature = nn.AdaptiveAvgPool2d(1)(feature)
		feature = torch.squeeze(feature, -1)
		feature = torch.squeeze(feature, -1)
	return feature

def total(data):
	array = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]

	total = 0
	for i in range(int((len(data) - 1) / 2)):
		if data[i*2 + 1] in array:
			total += int(data[(i + 1) * 2])
	return total

def total_(data):
	array = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]

	total = 0
	for i in range(int(len(data) / 2)):
		if data[i*2] in array:
			total += int(data[i*2 + 1])
	return total

def create_trans(data_seg, data_ins):
	print(data_seg)
	array = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
	pattern = []
	score_ = []

	total__ = total_(data_ins)
	for j in range(int(len(data_ins) / 2)):
		if data_ins[j*2] in array:
			item = data_ins[j*2]
			score = float(data_ins[j*2 + 1])/total__
			infos = ins_to_label(score, item)
			if len(infos) > 0:
				pattern.append(infos[0])
				score_.append(infos[1])

	for j in range(int(len(data_seg) / 2)):
		item = data_seg[j*2]
		score = data_seg[j*2 + 1]
		infos = seg_to_label(score, item)
		if len(infos) > 0:
			pattern.append(infos[0])
			score_.append(infos[1])
	return pattern, score_

def get_period(patterns, data_pm25, label_pm, arr):
	bow = []
	idx_value = []
	idx_pm = 0

	for mined_pattern in data_pm25:
		boolean = False

		for pattern in mined_pattern:
			if pattern in arr:
				if pattern == 'PM25_lv' + str(label_pm + 1):
					boolean = True
					break

		if boolean == True:
			total = 0
			for info in patterns:
				if info in mined_pattern[0:-1]:
					total += 1
			bow.append(total)
			idx_value.append(idx_pm)
		idx_pm += 1

	if len(bow) != 0:
		idx_max = np.argmax(np.asarray(bow))

		idx_arr = np.where(np.asarray(bow)==bow[idx_max])

		period = None

		for ids in idx_arr[0]:
			period = int(data_pm25[idx_value[idx_max]][-1])
			if period == 0:
				period = int(data_pm25[idx_value[ids]][-1])
			else:
				if period > int(data_pm25[idx_value[ids]][-1]):
					period = int(data_pm25[idx_value[ids]][-1])


	return period