#import tensorflow as tf
from PIL import Image
#import matplotlib.pyplot as plt
import os
import glob



swd = '../data/test/'
if not os.path.exists(swd):
    os.makedirs(swd)
lang = 'fr'
data_path = '../data/tfdataset/{}/'.format(lang)
print data_path
data_files = glob.glob(data_path+'*')
print(data_files)
#
# filename_queue = tf.train.string_input_producer(data_files,shuffle=True)
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)
# features = tf.parse_single_example(serialized_example,
#                                    features={
#                                        'height': tf.FixedLenFeature([], tf.int64),
#                                        'width': tf.FixedLenFeature([], tf.int64),
#                                        'depth': tf.FixedLenFeature([], tf.int64),
#                                        'label': tf.FixedLenFeature([], tf.int64),
#                                        'language': tf.FixedLenFeature([], tf.int64),
#                                        'nlines': tf.FixedLenFeature([], tf.int64),
#                                        'bbx': tf.VarLenFeature( tf.float32),
#                                        'image':  tf.FixedLenFeature([], tf.string)})
#
#
# image = tf.decode_raw(features['image'], tf.uint8)
# height = tf.cast(features['height'],tf.int32)
# width = tf.cast(features['width'],tf.int32)
# label = tf.cast(features['label'], tf.int32)
# channel = 3
# print "height,width:",height,width
# image = tf.reshape(image, [width,height,channel])
#
#
# with tf.Session() as sess:
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     coord=tf.train.Coordinator()
#     threads= tf.train.start_queue_runners(coord=coord)
#     for i in range(15):
#         #image_down = np.asarray(image_down.eval(), dtype='uint8')
#         plt.imshow(image.eval())
#         plt.show()
#         single,l = sess.run([image,label])
#         img=Image.fromarray(single, 'RGB')
#         img.save(swd+str(i)+'_''Label_'+str(l)+'.jpg')
#         #print(single,l)
#     coord.request_stop()
#     coord.join(threads)
# sess.close()
