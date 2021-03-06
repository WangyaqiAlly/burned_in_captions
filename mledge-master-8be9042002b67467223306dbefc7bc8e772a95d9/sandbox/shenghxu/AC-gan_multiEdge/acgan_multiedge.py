# CUDA_VISIBLE_DEVICES='1' python test_multiedge.py
import argparse
import numpy as np
import tensorflow as tf
from EdgeNode import EdgeNode
import Utils as utils
import os
import coloredlogs
import logging
from acGAN import acGAN
from CentralNode import CentralNode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', '--edge_ids', type=str, nargs='+', default=['1', '2', '3'],
                        help='list of ids')
    parser.add_argument('-t', '--start_time', type=int, default=1,
                        help='Start folder, 1 by default')
    parser.add_argument('-bz', '--batch_size', type=int, default=100,
                        help='The batch size, 100 by default.')
    parser.add_argument('-type', '--image_type', type=str, default='mnist',
                        choices=['mnist', 'imageNet'],
                        help='The type of input images, MNIST by default.')
    parser.add_argument('-dir', '--graph_dir', type=str,
                        default='/home2/shenghxu/imageNet/Data/imagenet-10-32x32-fold',
                        help='The root directory of the input graph,')
    parser.add_argument('-max_e', '--max_edge_epoch', type=int, default=10,
                        help='Maximal epoch number that Edge node runs, 100 by default.')
    parser.add_argument('-elr', '--edge_learning_rate', type=float, default=0.0001,
                        help='The initial learning rate of the edge nodes')
    parser.add_argument('-rp', '--record_path_prefix', type=str,
                        default='/home2/shenghxu/MultiEdge2/record',
                        help='The path prefix for recording the model.')
    parser.add_argument('-d', '--debug', type=str, default='False',
                        help='whether print debug info')
    parser.add_argument('-labelr', '--label_rate', type=float, default=1.0,
                        help='percentage of data with labels')
    parser.add_argument('-catd', '--cat_dim', type=int, default=10,
                        help='category dimension')
    parser.add_argument('-cond', '--con_dim', type=int, default=2,
                        help='continuous dimension')
    parser.add_argument('-latentd', '--latent_dim', type=int, default=110,
                        help='latent dimension')
    parser.add_argument('-max_c', '--max_center_epoch', type=int, default=100,
                        help='Maximal epoch number that Center node runs, 100 by default.')
    parser.add_argument('-clr', '--center_learning_rate', type=float, default=0.0001,
                        help='The learning rate of the central classifier')
    parser.add_argument('-ng', '--num_generated_images', type=int, default=100,
                        help='In each epoch, the number of fake images generated by one edge.')
    parser.add_argument('-recordt', '--record_time', type=int, default=0,
                        help='Choose the record time that the center want to use. 0 by default.')


    args = parser.parse_args()

    if args.debug is 'False':
        coloredlogs.install(level='INFO')
    else:
        coloredlogs.install(level='DEBUG')

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    is_first_time = np.ones(len(args.edge_ids))  # Could be improved to each site record

    print '========BEGIN SESSION========='
    edge_proc = []
    for time in range(args.start_time, 11):
        for i, edge_id in enumerate(args.edge_ids):

            snapshot_path = os.path.join(args.record_path_prefix, edge_id, 'time_' + str(time), 'model.ckpt')
            if not os.path.exists(snapshot_path):
                os.makedirs(snapshot_path)

            sess = tf.Session(config=run_config)

            if is_first_time[i]:
                # This is the only line that needs to be changed if just for a simple test
                edge_proc.append(acGAN(sess, args.graph_dir, edge_id, args.image_type, time,
                                          snapshot_path, args.batch_size, args.edge_learning_rate,args.label_rate,args.cat_dim,args.con_dim,args.latent_dim
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

        print '========BEGIN SESSION========='
        center = CentralNode(args.graph_dir, args.image_type,
                         args.batch_size, True, args.record_path_prefix, 1000,
                         args.edge_ids, time+1, args.center_learning_rate)
        print '==========Construction finished============'

        center.initialize()
        print '==========Initialization finished ========='

        center.train(args.max_center_epoch)
        print '========== Train finished ========='

