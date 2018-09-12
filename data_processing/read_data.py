import tensorflow as tf
from PIL import Image
# import matplotlib.pyplot as plt
import os
import glob



swd = '../data/test/'
if not os.path.exists(swd):
    os.makedirs(swd)
classes=['none','en','es','ja','fr','it','de','ko']
for lang in classes[1:]:

    data_path = '../data/tfdataset/ori_size/{}/'.format(lang)

    print data_path
    for posneg in ['pos','neg']:
        data_files = glob.glob(data_path+'/*'+posneg+'*')
        print(data_files)

        filename_queue = tf.train.string_input_producer(data_files,shuffle=True,num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'height': tf.FixedLenFeature([], tf.int64),
                                               'width': tf.FixedLenFeature([], tf.int64),
                                               'depth': tf.FixedLenFeature([], tf.int64),
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'language': tf.FixedLenFeature([], tf.int64),
                                               'nlines': tf.FixedLenFeature([], tf.int64),
                                               'bbx': tf.VarLenFeature( tf.float32),
                                               'image':  tf.FixedLenFeature([], tf.string)})


        image = tf.decode_raw(features['image'], tf.uint8)
        height = tf.cast(features['height'],tf.int32)
        width = tf.cast(features['width'],tf.int32)
        label = tf.cast(features['label'], tf.int32)
        nlines =  tf.cast(features['nlines'],tf.int32)
        bbox = tf.cast(features['bbx'],tf.float32)
        language = tf.cast(features['language'],tf.int32)
        channel = 3

        image = tf.reshape(image, [width,height,channel])


        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord=tf.train.Coordinator()
            threads= tf.train.start_queue_runners(coord=coord)
            i =0
            while True:
                try:
                    single, l, w, h, n, b, lang = sess.run([image, label, width, height, nlines, bbox, language])
                    i += 1
                    if i %1000==0:
                        print i,
                except:
                    print "done!"
                    break

            print "images in {},{} : {}".format(lang,posneg,i)



            # for i in range(15):
            #     #image_down = np.asarray(image_down.eval(), dtype='uint8')
            #     # plt.imshow(image.eval())
            #     # plt.show()
            #     single,l ,w,h,n,b,lang= sess.run([image,label,width,height,nlines,bbox,language])
            #     # print "height,width:", w,h,n,b,lang
            #     print type(single),single.max(),single.min()
            #     # img=Image.fromarray(single, 'RGB')
            #     # img.save(swd+"test_ori_size"+str(i)+'_''Label_'+str(l)+'.jpg')
            #     #print(single,l)
            coord.request_stop()
            coord.join(threads)
        sess.close()