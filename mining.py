import UPFPGrowth as alg
import argparse


def mining(args):
	arr = ['PM25_lv1', 'PM25_lv2', 'PM25_lv3', 'PM25_lv4', 'PM25_lv5', 'PM25_lv6']

	obj = alg.UPFPGrowth(args.input_file, args.minSup, args.maxPer, '\t')
	obj.startMine()

	periodicFrequentPatterns = obj.getPatterns()
	print("Total number of Periodic Frequent Patterns:", len(periodicFrequentPatterns))
	obj.savePatterns(args.output_file)

	f = open(args.output_file, 'r')
	datas = f.read()
	datas = datas.split('\n')

	file = open(args.output_file, 'w')

	total = 0

	for data in datas:
		datas_ = data.split(' ')[0:-3]

		if len(datas_) > 2:
			for data_ in datas_:
				if data_ in arr:
					total += 1
					file.write(data + '\n')

	print("After filter:", total)





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', type=str, default='Data/data.txt', help="Path to txt file")
	parser.add_argument('--output_file', type=str, default='Data/result.txt', help="Full name of output file")
	parser.add_argument('--minSup', type=float, default=0.2, help="Minimum Support")
	parser.add_argument('--maxPer', type=float, default=0.2, help="Maximum Period")
	args = parser.parse_args()
	mining(args)