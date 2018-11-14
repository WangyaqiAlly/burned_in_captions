from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys
import sklearn.metrics
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from tensorflow.python.platform import gfile

sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer

from data_loader import  inputs
import  logging
FLAGS = tf.app.flags.FLAGS
logger = logging.getLogger()
logger.setLevel(logging.INFO)

tf.app.flags.DEFINE_string('img_save_path', '../../data/test/ctpn/',
                           'where to store the dataset')

area_img = FLAGS.img_width *FLAGS.img_height
if not os.path.exists(FLAGS.img_save_path):
    os.makedirs(FLAGS.img_save_path)

classes = ['neg', 'en', 'es', 'ja', 'fr', 'it', 'de', 'ko']

def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    with open(FLAGS.img_save_path + '/res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(FLAGS.img_save_path, base_name), img)

def save_textlines(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    for i,box in enumerate(boxes):
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        if box[8]<0.5:
            continue
        min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))


        text_line = img[min_y:max_y, min_x:max_x]
        cv2.imwrite(os.path.join(FLAGS.img_save_path, base_name+'{}.png'.format(i)), text_line)



def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

# def load_data_tfrecord():
#     swd = '../data/test/'
#     print("testing here!")
#     with tf.device("/cpu:0"):
#         with tf.Session() as sess:
#             images, labels = inputs(batch_size=100, train=True,
#                                     shuffle=True, num_epochs=None)
#             init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#             sess.run(init_op)
#             all_labels = []
#             langs = []
#             all_images = []
#             coord = tf.train.Coordinator()
#             threads = tf.train.start_queue_runners(coord=coord)
#             while True:
#                 try:
#                     _images, _labels = sess.run([images, labels])
#                     j = 0
#                     for single, l in zip(_images, _labels):
#                         img = Image.fromarray(single, 'RGB')
#                         all_images.append(img)
#                         all_labels.append(l)
#                         j += 1
#                         print(j)
#                 except:
#                     print("finish!")
#                     break
#             coord.request_stop()
#             coord.join(threads)
#         sess.close()
#         return all_iamges,all_labels

def load_data_tfrecord():
    data_path = '../../data/tfdataset/language/'
    data_files = glob.glob(data_path + '/testdata_de*')
    print(data_files)
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True, num_epochs=1)
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
                                           'bbx': tf.VarLenFeature(tf.float32),
                                           'image': tf.FixedLenFeature([], tf.string)})

    image = tf.decode_raw(features['image'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    nlines = tf.cast(features['nlines'], tf.int32)
    bbox = tf.cast(features['bbx'], tf.float32)
    language = tf.cast(features['language'], tf.int32)
    channel = 3

    image = tf.reshape(image, [ height,width ,channel])
    image= tf.reverse(image, axis=[-1])


    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        i = 0
        all_images = []
        all_labels =[]
        all_langs =[]
        while True:
        # while i<1005:
            try:
                single, l, w, h, n, b, lang = sess.run([image, label, width, height, nlines, bbox, language])
                i += 1
                all_images.append(single)
                all_labels.append(l)
                all_langs.append(lang)
                print(i)
            except:
                print
                "done!"
                break

        print
        "images in test  {}".format( i)

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
    return all_images,all_labels,all_langs


def accuacry_evaluate(preds,scores, labels):
    assert labels.shape[0] == pred_labels.shape[0]
    recall = sklearn.metrics.recall_score(labels, preds,labels =[0,1],pos_label=1 )
    acc_from_sk = sklearn.metrics.accuracy_score(labels, preds)
    precision = sklearn.metrics.precision_score(labels, preds)
    acc = np.count_nonzero(labels==preds) / float(labels.shape[0])

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores, pos_label=1)
    auc_score = sklearn.metrics.roc_auc_score(labels,scores)
    return acc_from_sk,acc,precision,recall,auc_score, fpr, tpr, thresholds


def boxes_filter(boxes,scale,shape):
    num = len(boxes)
    res = 0
    filtered_boxes=[]
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue

        min_x = min(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
        min_y = min(int(box[1]), int(box[3]), int(box[5]), int(box[7]))
        max_x = max(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
        max_y = max(int(box[1]), int(box[3]), int(box[5]), int(box[7]))
        area = (max_x - min_x) * (max_y - min_y)
        # print(max_y,max_x,min_y,min_x)
        if area * 2.0 > shape[0]*shape[1]:
            continue
        if max_y < shape[0]*2.0 / 3.0:
            continue
        # if box[8] < 0.8:
        #     continue
        filtered_boxes.append(box)
        res= max(res,box[8])

    return res,filtered_boxes



if __name__ == '__main__':
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")


    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile('model/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

    all_images, all_labels,all_langs= load_data_tfrecord()
    all_labels = np.asarray(all_labels,dtype=int)
    pred_labels = np.zeros((len(all_labels),),dtype=int)
    class_scores = np.zeros((len(all_labels),),dtype=np.float32)
    for i,img in enumerate(all_images):
        print(i)
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print(('Demo for {:s}'.format(im_name)))
        # img = cv2.imread(im_name)
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
        print(im_blob.shape)
        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        lang = classes[all_langs[i]]
        im_name = '{:04d}_{}'.format(i,lang)
        # draw_boxes(img, im_name, boxes, scale)

        class_score,filtered_boxes = boxes_filter(boxes,scale,im_blob.shape[1:3])
        save_textlines(img, im_name, filtered_boxes, scale)
        pred_labels[i] = int(class_score >0.5)
        class_scores[i] =  class_score
    acc_sk, acc,precision , recall,auc, fpr, tpr, thresholds = accuacry_evaluate(pred_labels,class_scores,all_labels)
    # print(pred_labels,all_labels)

    # plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("roc_from_ctpn.jpg")
    print("reslut from ctpn detection:acc{:.3f},presicion:{:.3f},recall:{:.3f}, acc from sk:{:.3f}, auc:{:.3f}".format(acc,precision,recall,acc_sk,auc))



