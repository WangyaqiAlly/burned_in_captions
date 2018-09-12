import os
from shutil import copyfile
import csv
import random

data_dir = '/Data/imagenet-10-32x32'
class_list = []
class_index = 0
len_list = []
image_list = []
time_num = 10
site_num = 5
validation_ratio = 0.2
valid_dir = 'validation'

if not os.path.exists(valid_dir):
    os.mkdir(valid_dir)

validation = ['validation']
header = ['']
meta_file = open('meta.txt', 'w')
for item in os.listdir(data_dir):
    class_path = os.path.join(data_dir, item)
    if os.path.isdir(class_path):
        header += [item]
        class_list += [item]
        class_index += 1
        meta_file.write('class-%02d %s\n' % (class_index, item))
        path_list = os.listdir(class_path)
        random.shuffle(path_list)
        image_list += [path_list]
        num = int(len(path_list) * (1 - validation_ratio))
        len_list += [num]
        valid_path = os.path.join(valid_dir, 'class-%02d' % class_index)
        if not os.path.exists(valid_path):
            os.mkdir(valid_path)
        valid_list = os.listdir(class_path)[num:]
        for k, image_name in enumerate(valid_list):
            image_postfix = image_name.split('.')[-1]            
            if (image_postfix == 'JPG') or (image_postfix == 'JPEG'):
                image_postfix = 'jpg'
            if image_postfix != 'jpg':
                continue
            dest_path = os.path.join(valid_path, '%06d-32x32.' % (k) + image_postfix)
            src_path = os.path.join(class_path, image_name)
            copyfile(src_path, dest_path)
        print 'Validation class=%s: %d' % ('class-%02d' % class_index, len(valid_list))
        validation += [len(valid_list)]

meta_file.close()
print len_list
result = []

if not os.path.exists('train'):
    os.mkdir('train')
train_dir = 'train'
for n in range(site_num):
    site_path = os.path.join(train_dir, 'site-%d' % (n+1))
    if not os.path.exists(site_path):
        os.mkdir(site_path)
    for t in range(time_num):
        time_path = os.path.join(site_path, 'time-%02d' % (t+1))
        if not os.path.exists(time_path):
            os.mkdir(time_path)
        temp = ['n=%d t=%d' % (n+1, t+1)]
        for index, class_name in enumerate(class_list):
            class_path = os.path.join(time_path, 'class-%02d' % (index+1))
            if not os.path.exists(class_path):
                os.mkdir(class_path)
            s = n * time_num + t
            e = s + 1
            s = s * (len_list[index] / (site_num * time_num))
            e = e * (len_list[index] / (site_num * time_num))
            k = 0
            for i in range(s, e):
                image_name = image_list[index][i]
                image_postfix = image_name.split('.')[-1]
                if (image_postfix == 'JPG') or (image_postfix == 'JPEG'):
                    image_postfix = 'jpg'
                if image_postfix != 'jpg':
                    continue
                dest_path = os.path.join(class_path, '%06d-32x32.' % (k) + image_postfix)
                src_path = os.path.join(data_dir, class_name, image_name)
                copyfile(src_path, dest_path)                
                k += 1
            print 'Finish n=%d, t=%d, class=%d, s=%d, e=%d' % (n+1, t+1, index+1, s, e)
            temp += [len_list[index] / (site_num * time_num)]
        result += [temp]

with open('result.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)
    for row in result:
        writer.writerow(row)
    writer.writerow(validation)

