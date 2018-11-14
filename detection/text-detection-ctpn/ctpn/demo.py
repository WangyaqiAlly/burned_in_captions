from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer

def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), f

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
        # if area * 2.0 > shape[0]*shape[1]:
        #     continue
        # area = (max_x - min_x) * (max_y - min_y)
        # print(max_y,max_x,min_y,min_x)
        # if area * 2.0 > shape[0]*shape[1]:
        #     continue
        # if max_y < shape[0]*2.0 / 3.0:
        #     continue
        # print(area, (min_x + max_x) / 2.0,min_x,max_x)
        if box[8] < 0.9:
            continue
        if (min_x + max_x) / 2.0 > 0.8 * shape[1]:
            continue
        # or (min_x + max_x) / 2.0 < 0.2 * shape[1]:
        #     continue
        if max_x - min_x < 0.05 * shape[1]:
            continue
        if area < 1500:
            continue
        # if max_y < shape[0]*2.0 / 3.0:
        #     continue
        # if box[8] < 0.8:
        #     continue
        box_minmax = [min_x,min_y,max_x,max_y]
        filtered_boxes.append(box_minmax)
        res= max(res,box[8])

    return res,filtered_boxes

def draw_boxes(img, base_name, boxes, scale,save_path):
    with open(save_path + '/{}_res.txt'.format(base_name), 'w') as f:
        for box in boxes:
            color = (0, 255, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), color, 2)
            cv2.line(img, (int(box[2]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[3])), (int(box[2]), int(box[3])), color, 2)

            line = ','.join([str(box[0]), str(box[1]), str(box[2]), str(box[3])]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(save_path, base_name+'_bbx.jpg'), img)

def _get_image_blob(imgs):
    # num_images = len(imgs)
    # im_shape = imgs[0].shape
    # im_size_min = np.min(im_shape[0:2])
    # im_size_max = np.max(im_shape[0:2])
    # target_size = cfg.TEST.SCALES[0]
    # im_scale = float(target_size) / float(im_size_min)
    #
    # # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
    #     im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    #
    # resizec_h = int(im_shape[0] * im_scale)
    # resized_w = int(im_shape[1] * im_scale)
    # blob = np.zeros((num_images, 480,640, 3),
    #                 dtype=np.float32)
    blob = np.asarray(imgs, dtype=np.float32)
    # print(blob.shape)
    # assert blob.shape[1] == (200,320,3)
    # for i,im in enumerate(imgs):
    #     im = im.astype(np.float32, copy=True)
    #     # im_orig -=
    #     # im = cv2.resize(im_orig,(resized_w,resizec_h),interpolation=cv2.INTER_LINEAR)
    #     blob[i]=im
    blob -= cfg.PIXEL_MEANS
    im_scale =1.0
    return blob,im_scale


def _get_blobs(imgs, rois):
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factor = _get_image_blob(imgs)
    return blobs, im_scale_factor

def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    height, width = img.shape[:2]
    img = img[int(2*height/3.0):height,:]
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)
    # for box in boxes:
    #     color = (0, 255, 0)
    #     cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), color, 2)
    #     cv2.line(img, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), color, 2)
    #     cv2.line(img, (int(box[2]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    #     cv2.line(img, (int(box[0]), int(box[3])), (int(box[2]), int(box[3])), color, 2)
    # base_name = image_name.split('/')[-1]
    # cv2.imwrite("data/results/test_"+base_name, img)
    # draw_boxes(img, image_name, boxes, scale)
    # print(boxes)
    # assert 0
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

def test_ctpn(sess, net, im, boxes=None):
    blobs, im_scales = _get_blobs(im, boxes)
    # print(im_scales)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales]],
            dtype=np.float32)
    # forward pass
    if cfg.TEST.HAS_RPN:
        feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}

    rois = sess.run([net.get_output('rois')[0]],feed_dict=feed_dict)
    rois=rois[0]
    # print(rois)
    scores = rois[:, 0]
    if cfg.TEST.HAS_RPN:
        # assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales
    return scores,boxes


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



if __name__ == '__main__':
    save_path = '/home/CORP/yaqi.wang/pycharm/data/video_level/frames/training_set/ko_new_detected'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cfg_from_file('ctpn/text.yml')

    # init session
    # config = tf.ConfigProto(allow_soft_placement=True)
    # sess = tf.Session(config=config)
    # load network
    # net = get_network("VGGnet_test")
    # load model
    # print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    # saver = tf.train.Saver()
    #
    # try:
    #     # ckpt = tf.train.get_checkpoint_state('/home/CORP/yaqi.wang/pycharm/detection/text-detection-ctpn/model/textline_v3')
    #     ckpt = tf.train.get_checkpoint_state('/home/CORP/yaqi.wang/pycharm/detection/text-detection-ctpn/model/checkpoints')
    #     print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     print('done')
    # except:
    #     raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    print('model loading...')
    with gfile.FastGFile('data/textline_ori.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
    output_rois = sess.graph.get_tensor_by_name('rois/Reshape:0')
    #
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in range(2):
    #     _, _ = test_ctpn(sess, net, im)
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in range(2):
    #     _, _ = test_ctpn(sess, net, im)
    # im_names = glob.glob(os.path.join('/home/CORP/yaqi.wang/pycharm/data/video_level/frames/training_set', 'en/*', '*.jpg'))\
    #     # + glob.glob(os.path.join('/home/CORP/yaqi.wang/pycharm/data/video_level/frames/training_set', 'es/*', '*.jpg'))\
    #     # + glob.glob(os.path.join('/home/CORP/yaqi.wang/pycharm/data/video_level/frames/training_set', 'ja/*', '*.jpg'))
    im_names = glob.glob(os.path.join('/home/CORP/yaqi.wang/pycharm/data/video_level/frames/training_set', 'ko_new/', '*.jpg'))
    print(len(im_names))
    cnt = 0
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        timer = Timer()
        timer.tic()
        img = cv2.imread(im_name)
        height, width = img.shape[:2]
        img = img[int(2 * height / 3.0):height, :]
        img, scale = resize_im(img, scale=200, max_scale=1000)
        blobs, im_scales = _get_blobs([img],None)
        # print(im_scales)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales]],
                dtype=np.float32)
        # forward pass
        # if cfg.TEST.HAS_RPN:
            # feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
        # cls_prob, box_pred, rois = sess.run([output_cls_prob, output_box_pred, output_rois],
        #                                         feed_dict={input_img: blobs['data']})
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
            # _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
        # print(rois.shape)
        # rois = rois[0]
        scores = rois[:, 0]
        # rois = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        # print(rois)
        if cfg.TEST.HAS_RPN:
            # assert len(im_scales) == 1, "Only single-image batch implemented"
            boxes = rois[:, 1:5] / im_scales
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        class_score, filtered_box = boxes_filter(boxes, scale, img.shape[:2])
        if len(filtered_box)>0:
            print(cnt)
            cnt +=1
            save_name = os.path.basename(im_name)[:-4]
            # save_name = '_'.join(im_name.split('/')[-3:])[:-4]
            print(save_name)
            draw_boxes(img, save_name, filtered_box, scale,save_path)
            shutil.move(im_name,save_path+'/'+save_name+'.jpg')
            # assert 0
        print (cnt,"frames detected")
        # scores, boxes = test_ctpn(sess, net, img)
        # ctpn(sess, net, im_name)
    # print(im_names)
    # for i,im_name in enumerate(im_names):
    #     img = cv2.imread(im_name)
    #     height,width=img.shape[:2]
    #     img = img[int(2*height/3.0):height,:]
    #     if i == 0:
    #         images = img[np.newaxis,:]
    #     else:
    #         images = np.vstack((images, img[np.newaxis,:]))
    #     # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     # print(('Demo for {:s}'.format(im_name)))
    #     # ctpn(sess, net, im_name)
    #
    #     # img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # print(images.shape)
    # blobs, im_scale = _get_blobs(images, None)
    # if cfg.TEST.HAS_RPN:
    #     im_blob = blobs['data']
    #     blobs['im_info'] = np.array(
    #         [[im_blob.shape[1], im_blob.shape[2], im_scale]],
    #         dtype=np.float32)
    #
    # if cfg.TEST.HAS_RPN:
    #     feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
    #
    # rois_all = sess.run([net.get_output('rois')[0]],feed_dict=feed_dict)
    # scores_all = rois_all[:, 0]
    # if cfg.TEST.HAS_RPN:
    #     # assert len(im_scales) == 1, "Only single-image batch implemented"
    #     boxes_all = rois_all[:, 1:5] / im_scale
    # textdetector = TextDetector()
    #
    #     boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    #     draw_boxes(img, image_name, boxes, im_scale)
