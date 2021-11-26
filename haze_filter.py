import os

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def fuzzy_hazer(value):# 0.1 - 0.4
	value = float(value) * 2.5

	if value >= 1:
		return 1, 'Foggy'

	elif value <= 0.1:
		return 1, 'Haze_lv0'

	elif value <= 0.3 and value > 0.1:
		haze_lv0 = 1 - (value - 0.1)/0.2
		haze_lv1 = (value - 0.1)/0.2

		if haze_lv0 > haze_lv1:
			return truncate(haze_lv0, 2), 'Haze_lv0'
		else:
			return truncate(haze_lv1, 2), 'Haze_lv1'

	elif value <= 0.5 and value > 0.3:
		haze_lv1 = 1 - (value - 0.3)/0.2
		haze_lv2 = (value - 0.3)/0.2

		if haze_lv1 > haze_lv2:
			return truncate(haze_lv1, 2), 'Haze_lv1'
		else:
			return truncate(haze_lv2, 2), 'Haze_lv2'

	elif value <= 0.7 and value > 0.5:
		haze_lv2 = 1 - (value - 0.5)/0.2
		haze_lv3 = (value - 0.5)/0.2

		if haze_lv2 > haze_lv3:
			return truncate(haze_lv2, 2), 'Haze_lv2'
		else:
			return truncate(haze_lv3, 2), 'Haze_lv3'

	elif value <= 0.8 and value > 0.7:
		haze_lv3 = 1 - (value - 0.7)/0.1
		haze_lv4 = (value - 0.7)/0.1

		if haze_lv3 > haze_lv4:
			return truncate(haze_lv3, 2), 'Haze_lv3'
		else:
			return truncate(haze_lv4, 2), 'Haze_lv4'

	elif value <= 0.9 and value > 0.8:
		haze_lv4 = 1 - (value - 0.8)/0.1
		haze_lv5 = (value - 0.8)/0.1

		if haze_lv4 > haze_lv5:
			return truncate(haze_lv4, 2), 'Haze_lv4'
		else:
			return truncate(haze_lv5, 2), 'Haze_lv5'

	elif value < 1.0 and value > 0.9:
		haze_lv5 = 1 - (value - 0.9)/0.1
		foggy = (value - 0.9)/0.1

		if haze_lv5 > foggy:
			return truncate(haze_lv5, 2), 'Haze_lv5'
		else:
			return truncate(foggy, 2), 'Foggy'



def filter(args):
	f = open(args.input_path, 'r')
	datas = f.read()
	datas = datas.split('\n')

	file = open(output_path, 'w')

	for data in datas:
		if len(data) != 0:
			data_ = data.split('  ')

			value, label = fuzzy_hazer(data_[1][0:-4])
			file.write(data_[0][0: -6] + '\t' + label + '(' + str(value) + ')\t' + 'A0:' + data_[2] + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, default='Data_Img', help="Path to fuzzy txt file")
	parser.add_argument('--output_path', type=str, default='Data', help="Output path")
	args = parser.parse_args()
	# extract(args)
	generate_file(args)