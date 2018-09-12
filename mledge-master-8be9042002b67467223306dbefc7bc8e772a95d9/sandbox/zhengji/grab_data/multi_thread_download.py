import urllib
import socket
import hashlib
import os
from shutil import move
import time
import Queue
from threading import Thread
socket.setdefaulttimeout(2)

url_pool = Queue.Queue()
image_pool = Queue.Queue()

worker_num = 22 
url_file = 'ship_url.txt'
output_dir ='ship'
md5_dict = {}

def worker(name, url_pool, image_pool):
    global output_dir
    print 'Worker %s is setup.' % name
    count = 0
    while True:
        index, image_url, postfix = url_pool.get()
        if index == -1:
            print 'Worker %s is terminated' % name
            image_pool.put([None, None])
            break
        tempname = 'worker_'+name+'_temp_' + str(count) +'.' + postfix
        try:
            urllib.urlretrieve(image_url, tempname)
            image_pool.put([tempname, postfix])
            count += 1
        except:
            print 'Worker', name, ': URL', image_url, ' invalid.'
            pass


print 'I will sleep 0s'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

f = open(url_file)

# send to workers
for index, item in enumerate(f.readlines()):
    image_url = item.strip()
    postfix = image_url.split('.')[-1]
    if not (postfix in ['jpg', 'JPG', 'jpeg', 'png', 'PNG', 'JPEG']):
        continue

    url_pool.put([index, image_url, postfix])

f.close()

for i in range(worker_num):
    url_pool.put([-1, "", ""])

# initialize workers
worker_list = []
for i in range(worker_num):
    p = Thread(target=worker, args=('%02d' % (i+1), url_pool, image_pool,))
    p.setDaemon(True)
    p.start()
    worker_list += [p]


# collect result
count = 0
hasher = hashlib.md5()
live_left = worker_num
while True:
    tempname, postfix = image_pool.get()
    if tempname is None:
        live_left -= 1
        if live_left == 0:
            break
        continue
    filename = os.path.join(output_dir, str(count) + '.' + postfix)
    with open(tempname, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
        key = hasher.hexdigest()
    if md5_dict.get(key) is None:
        move(tempname, filename)
        md5_dict[key] = image_url
        count += 1
    print 'Finished #', count


for item in worker_list:
    item.join()

