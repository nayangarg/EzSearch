# from read_write_create import *
import csv,os, sys

li = sys.argv[1:]
path = li[0]
data_path = path + '/wordList'

wrtr = csv.writer(open(path[:-5] + 'glove/corpus.txt', 'q'), delimiter=' ')

for every_file in (os.listdir(data_path)):
	print(every_file)
	rdr = csv.reader(open(data_path + '/' + every_file, 'r'))

	for line in rdr:
		wrtr.writerow(line)

	wrtr.writerow('')