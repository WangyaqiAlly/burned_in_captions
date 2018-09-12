import argparse
import numpy as np
import tensorflow as tf
from EdgeNode import EdgeNode
import os
import cv2

RECORD_FILE_PREFIX = '/home2/ryang2/mledge/sandbox/ryang2/edgeGAN/record'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--edge_id', type=str, nargs='+', default='0',
                        help='list of ids')
    parser.add_argument('-t', '--start_time', type=int, default=0,
                        help='Start folder, 0 by default')
    parser.add_argument('-bz', '--batch_size', type=int, default=100,
                        help='The batch size, 100 by default.')
    parser.add_argument('-type', '--image_type', type=str, default='mnist',
                        choices=['mnist', 'imageNet'],
                        help='The type of input images, MNIST by default.')
    parser.add_argument('-dir', '--graph_dir', type=str, default='./graph',
                        help='The root directory of the input graph,')
    parser.add_argument('-max_e', '--max_edge_epoch', type=int, default=10,
                        help='Maximal epoch number that Edge node runs, 100 by default.')
    parser.add_argument('-elr', '--edge_learning_rate', type=float, default=0.0001,
                        help='The initial learning rate of the edge nodes')
    parser.add_argument('-rp', '--record_path_prefix', type=str,
                        default='/home2/ryang2/mledge/sandbox/ryang2/edgeGAN/record',
                        help='The path prefix for recording the model.')

    args = parser.parse_args()

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    is_first_time = True

    print '========BEGIN SESSION========='
    snapshot_path = os.path.join(args.record_path_prefix, args.edge_id, 'model.ckpt')
    edge_proc = EdgeNode(sess, args.graph_dir, args.edge_id, args.image_type,
                         snapshot_path, args.batch_size, args.edge_learning_rate
                         )
    print '==========Construction finished============'

    edge_proc.initialize()
    print '==========Initialization finished ========='
    for folder in range(args.start_time, 10):
        edge_proc.restore_model(snapshot_path)
        print '========== Restore Mode finished =============='

        edge_proc.train(args.max_edge_epoch)
        print '========== Train finished ========='

        edge_proc.save_model(snapshot_path)
        print '========== Save Mode finished =============='


