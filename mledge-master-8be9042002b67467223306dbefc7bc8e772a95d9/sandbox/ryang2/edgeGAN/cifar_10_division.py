from shutil import copy2
import argparse
import cPickle
import os
import numpy as np
import cv2

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--source_dir', required=True, help='Source directory of CIFAR_10')
parser.add_argument('-d', '--dest_dir', required=True, help='Destination root directory of CIFAR_10')
parser.add_argument('-ns', '--num_site', default=2, help='Number of sites')
parser.add_argument('-nt', '--num_time', type=int, default=3, help='Number of time slices')
args = parser.parse_args()

def unpickle(file):
    print 'Loading from: ', file
    with open(file, 'rb') as fin:
        dict = cPickle.load(fin)
    return dict
def write_in(file, data):
    print 'Writing to: ', file, ' with len ', len(data['labels'])
    with open(file, 'wb') as fout:
        cPickle.dump(data, fout)

data= None
label=[]

if not os.path.exists(args.source_dir):
	print 'No input directory'
	exit()
if not os.path.exists(args.dest_dir):
	os.makedirs(args.dest_dir)
	
fileNames = os.listdir(args.source_dir)
for f in fileNames:
  if 'data_batch' in f:
    tempDict = unpickle(os.path.join(args.source_dir, f))
    tempData = tempDict['data']
    tempLabel = tempDict['labels']
    if data is None:
        data = tempData
    else:
        data = np.concatenate([data, tempData], axis=0)
    label.extend(tempLabel)
    
# print 'len of label: ', len(label), '  shape of data: ', data.shape

# Divide data
size_per_time = len(label) / args.num_site / args.num_time
print size_per_time
start, end = 0, size_per_time

for site in xrange(args.num_site):
    for time in xrange(args.num_time):
        per_path = os.path.join(args.dest_dir, 'train', 'site-' + str(site + 1), 'time-%02d' % (time + 1))
        print per_path
        if not os.path.exists(per_path):
            os.makedirs(per_path)

        # The last time slice in the last site contains all remaining data
        if end + size_per_time > len(label):
            end = len(label)

        per_data = data[start:end]
        per_label = label[start:end]
        # print 'per_size: ', len(per_label)
        per_dict = {'data' : per_data, 'labels' : per_label}
        write_in(per_path+'/data_'+str(site+1)+'_'+str(time+1), per_dict)

        start +=size_per_time
        end += size_per_time

test_data_src = args.source_dir + '/test_batch'
test_data_path = os.path.join(args.dest_dir, 'validation')
if not os.path.exists(test_data_path):
    os.makedirs(test_data_path)
test_data_dst = test_data_path + '/test'
copy2(test_data_src, test_data_dst)