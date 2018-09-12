# CUDA_VISIBLE_DEVICES='1' python test_multiedge.py
import argparse
import numpy as np
import tensorflow as tf
from EdgeNode import EdgeNode
import Utils as utils
import os
import coloredlogs
import logging
from labelGAN import labelGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', '--edge_ids', type=str, nargs='+', default=['0', '1', '2'],
                        help='list of ids')
    parser.add_argument('-t', '--start_time', type=int, default=0,
                        help='Start folder, 0 by default')
    parser.add_argument('-bz', '--batch_size', type=int, default=100,
                        help='The batch size, 100 by default.')
    parser.add_argument('-type', '--image_type', type=str, default='mnist',
                        choices=['mnist', 'imageNet'],
                        help='The type of input images, MNIST by default.')
    parser.add_argument('-dir', '--graph_dir', type=str,
                        default='../Data/imagenet-10-32x32-fold',
                        help='The root directory of the input graph,')
    parser.add_argument('-max_e', '--max_edge_epoch', type=int, default=10,
                        help='Maximal epoch number that Edge node runs, 100 by default.')
    parser.add_argument('-elr', '--edge_learning_rate', type=float, default=0.0001,
                        help='The initial learning rate of the edge nodes')
    parser.add_argument('-rp', '--record_path_prefix', type=str,
                        default='./record',
                        help='The path prefix for recording the model.')
    parser.add_argument('-d', '--debug', type=str, default='False',
                        help='whether print debug info')

    args = parser.parse_args()

    if args.debug is 'False':
        coloredlogs.install(level='INFO')
    else:
        coloredlogs.install(level='DEBUG')

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    is_first_time = np.ones(len(args.edge_ids))  # Could be improved to each site record

    print '========BEGIN SESSION========='
    # TODO: Detect # of folder
    edge_proc = []
    for time in range(args.start_time, 10):
        for i, edge_id in enumerate(args.edge_ids):

            snapshot_path = os.path.join(args.record_path_prefix, edge_id, 'time_' + str(time), 'model.ckpt')
            if not os.path.exists(snapshot_path):
                os.makedirs(snapshot_path)

            sess = tf.Session(config=run_config)

            if is_first_time[i]:
                edge_proc.append(wacGAN(sess, args.graph_dir, edge_id, args.image_type,
                                          snapshot_path, args.batch_size, args.edge_learning_rate
                                          ))
            edge_proc[i].initialize()

            if not is_first_time[i]:
                previous_snap_path = os.path.join(args.record_path_prefix, edge_id, 'time_' + str(time - 1),
                                                  'model.ckpt')
                edge_proc[i].restore_model(previous_snap_path)

            is_first_time[i] = False

            edge_proc[i].train(args.max_edge_epoch)
            print '========== Edge ', edge_id, ' time ', time, ' finished ============='
            edge_proc[i].save_model(snapshot_path)
