# CUDA_VISIBLE_DEVICES='1' python labelgan_multiedge.py
import argparse
import numpy as np
import tensorflow as tf
import os
import coloredlogs
import datetime
from labelGAN import labelGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', '--edge_ids', type=str, nargs='+', default=['1', '2', '3'],
                        help='list of ids')
    parser.add_argument('-t', '--start_time', type=int, default=1,
                        help='Start folder, 1 by default')
    parser.add_argument('-bz', '--batch_size', type=int, default=100,
                        help='The batch size, 100 by default.')
    parser.add_argument('-type', '--image_type', type=str, default='mnist',
                        choices=['mnist', 'imageNet', 'cifar10'],
                        help='The type of input images, MNIST by default.')
    parser.add_argument('-dir', '--graph_dir', type=str,
                        default='/home2/ryang2/mledge/sandbox/ryang2/edgeGAN/graph/imagenet-10-32x32-fold',
                        help='The root directory of the input graph,')
    parser.add_argument('-max_e', '--max_edge_epoch', type=int, default=10,
                        help='Maximal epoch number that Edge node runs, 100 by default.')
    parser.add_argument('-elr', '--edge_learning_rate', type=float, default=0.0001,
                        help='The initial learning rate of the edge nodes')
    parser.add_argument('-rp', '--record_path_prefix', type=str,
                        default='/home2/ryang2/mledge/sandbox/ryang2/edgeGAN/record',
                        help='The path prefix for recording the model.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='whether print debug info')
    parser.add_argument('-i', '--save_images', action='store_true',
                        help='whether save intermediate generated images')

    args = parser.parse_args()

    if not args.debug:
        coloredlogs.install(level='INFO')
    else:
        coloredlogs.install(level='DEBUG')

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    is_first_time = np.ones(len(args.edge_ids))  # Could be improved to each site record

    print '========BEGIN SESSION========='
    edge_proc = []
    record_root = os.path.join(args.record_path_prefix, '{:%Y.%m.%d.%H.%M.%S}'.format(datetime.datetime.now()))

    for time in range(args.start_time, 11):
        for i, edge_id in enumerate(args.edge_ids):

            snapshot_path = os.path.join(record_root, edge_id, 'time_' + str(time), 'model.ckpt')
            image_path = os.path.join(record_root, edge_id, 'time_' + str(time), 'images')

            if not os.path.exists(snapshot_path):
                os.makedirs(snapshot_path)

            sess = tf.Session(config=run_config)

            if is_first_time[i]:
                # This is the only line that needs to be changed if just for a simple test
                edge_proc.append(labelGAN(sess, args.graph_dir, edge_id, args.image_type, 1, time,
                                          snapshot_path, args.batch_size,
                                          args.edge_learning_rate, 16, args.save_images, image_path))
                edge_proc[i].build_network()

            edge_proc[i].initialize()

            if not is_first_time[i]:
                previous_snap_path = os.path.join(args.record_path_prefix, edge_id, 'time_' + str(time - 1),
                                                  'model.ckpt')
                edge_proc[i].restore_model(previous_snap_path)
                edge_proc[i].update_data(1, time, image_path)

            is_first_time[i] = False

            edge_proc[i].train(args.max_edge_epoch)
            print '========== Edge ', edge_id, ' time ', time, ' finished ============='
            edge_proc[i].save_model(snapshot_path)
