import torch
from efficientnet_pytorch import EfficientNet
import sys
from torchvision import transforms
from torch import nn
from PIL import Image
from neurnet import *
from utils import *

import numpy as np
import argparse 
import random
import csv
import json
import os

def train(args):
	if torch.cuda.is_available():
		device = torch.device("cuda")
		torch.cuda.set_device(args.cuda)
	else:
		device = torch.device("cpu")

	transform = transforms.Compose([transforms.Resize((600,600)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
	model_ef = EfficientNet.from_pretrained('efficientnet-b7').to(device)
	neural = Neural_Net(2560, args.num_cls).to(device)

	batch_size = args.batch_size
	epoch = 1
	global_step = Accumulate_Loss = 0
	pre_val = 1

	os.makedirs("Model/", exist_ok=True)
	optimizer = torch.optim.Adam(neural.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()

	model_ef.eval()
	neural.train()

	idx = 0
	f = open(args.path, 'r')
	datas = f.read().split('\n')
	f_ = open('Model/Test.txt', 'w')

	num = random.sample(range(0, len(datas)), int(len(datas)/10))
	train_data = []
	test_data = []

	for data in datas:
		data = data.split('\t')
		if idx in num:
			test_data.append([data[0], int(data[1])])
			f_.write(data[0] + '\n')
		else:
			train_data.append([data[0], int(data[1])])
		idx += 1

	length = len(train_data)

	iters = int(length/batch_size)
	best_acc = 0

	while epoch < args.epoch:
		random.shuffle(train_data)

		for k in range(iters):
			imgs = []
			target = []

			for i in range(batch_size):
				idx = batch_size*k + i
				path = train_data[idx][0]
				img = transform(Image.open(path))
				imgs.append(img)

				target.append(train_data[idx][1] - 1)
			target = torch.Tensor(target).type(torch.LongTensor).to(device)

			batch_img = torch.stack([img for img in imgs])

			features = extract(model_ef, batch_img.to(device))

			optimizer.zero_grad()
			theta = neural(features)

			loss = criterion(theta, target)
			loss.backward()
			optimizer.step()

		neural.eval()

		acc = 0

		for k in range(len(test_data)):
			path = test_data[k][0]
			img = transform(Image.open(path)).unsqueeze(0)

			features = extract(model_ef, img.to(device))
			with torch.no_grad():
				theta = neural(features)

			label = np.argmax(theta.cpu().numpy())

			if label == test_data[k][1] - 1:
				acc += 1

		if acc > best_acc:
			torch.save(neural.state_dict(), 'Model/model.pth')

			txt = open('Model/stat.txt', 'w')
			txt.write('Validation: %.3f'%(acc))
			txt.close()
		print('epoch', epoch, ':', acc/len(test_data))
		epoch += 1




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default='', help="Path to txt file")
	parser.add_argument('--num_cls', type=int, default=2, help="Number of class to classify")
	#parser.add_argument('--dataset_path', required=True, help="Path to train dataset folder")
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--epoch', type=int, default=10000)
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=32)
	args = parser.parse_args()
	train(args)