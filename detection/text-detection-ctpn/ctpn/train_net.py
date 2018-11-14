import os.path
import pprint
import sys

sys.path.append(os.getcwd())
from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir, cfg
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network
from easydict import EasyDict as edict
# from lib.fast_rcnn.config import

if __name__ == '__main__':
    # print(cfg.TRAIN.restore)
    cfg_from_file('ctpn/text.yml')
    # print(cfg.TRAIN)
    # cfg_test= cfg
    #
    # print(cfg_test.TRAIN.restore)
    print('Using config:')
    imdb = get_imdb('voc_2007_train')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, 'textline_v4')
    log_dir = get_log_dir(imdb)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('Logs will be saved to `{:s}`'.format(log_dir))

    device_name = '/gpu:0'
    print(device_name)

    network = get_network('VGGnet_train')
    pprint.pprint(cfg)
    # print(bool(int(cfg.TRAIN.restore)))
    # print(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
    # print(cfg.TRAIN.restore)
    # assert 0
    train_net(network, imdb, roidb,
              output_dir=output_dir,
              log_dir=log_dir,
              pretrained_model='data/pretrain/VGG_imagenet.npy',
              max_iters=int(cfg.TRAIN.max_steps),
              pretrained_dir = '/home/CORP/yaqi.wang/pycharm/detection/text-detection-ctpn/model/textline_v4',
              restore=bool(int(cfg.TRAIN.restore)))
