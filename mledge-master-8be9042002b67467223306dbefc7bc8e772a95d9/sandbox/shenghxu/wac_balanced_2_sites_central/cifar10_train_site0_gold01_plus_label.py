# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================

from resnet import *
from datetime import datetime
import time
from cifar10_input import *
#import pandas as pd
USE_AUGMENTATION = True
train_dir = 'wac_Site0_gold_01_plus_label'
#Data_File = './cifar10_50percent/orig_data0.npy'
#Label_File = './cifar10_50percent/orig_label0.npy'
# Scope_Prefix = 'ali_unconditional_100_'
Scope_Prefix = 'wac_Site0_gold_01_plus_label'
Total_iteration = 2000000
Total_batchn = 50000
Record_file = Scope_Prefix + ('bn_%06d' % Total_batchn )+ ('_iter_%06d' % Total_iteration )+'record.txt'
acc_file_name = Scope_Prefix + ('bn_%06d' % Total_batchn )+ ('_iter_%06d' % Total_iteration )+'record_each_class.txt'
acc_file_name_train = Scope_Prefix + ('bn_%06d' % Total_batchn )+ ('_iter_%06d' % Total_iteration )+'record_each_class_train.txt'
mtx_file_name = Scope_Prefix + ('bn_%06d' % Total_batchn )+ ('_iter_%06d' % Total_iteration )+'confusion_mtx.txt'

class Train_site0(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self, id=Scope_Prefix):
        # Set up all the placeholders
        self.id = id
        self.placeholders()


    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily


        '''

        with tf.variable_scope(self.id+'classifier_' ):
            self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                    shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                            IMG_WIDTH, IMG_DEPTH])
            self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

            self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

            self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])



    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.
        
        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reusef=False,id=self.id)
        print '************************************************************************************'
        print logits
        print '************************************************************************************'
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reusef=True,id=self.id)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)

        self.acc_each_sample_train = tf.cast(
                    tf.equal(
                        tf.to_int32(tf.argmax(logits, dimension=1)),self.label_placeholder
                    ),
                    tf.float32
                )

        self.predicted_train_labels = tf.to_int32(tf.argmax(logits, dimension=1))

        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)


        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)
        self.predicted_vali_labels = tf.to_int32(tf.argmax(vali_logits, dimension=1))
        self.acc_each_sample = tf.cast(
                    tf.equal(
                        tf.to_int32(tf.argmax(vali_logits, dimension=1)),self.vali_label_placeholder
                    ),
                    tf.float32
                )

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)


    def read_site_data(self):
        data = np.load(Data_File)
        num_data = data.shape[0]
        self.data_length = num_data
        data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
        data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
        data = data.astype(np.float32)
        label = np.load(Label_File)
        return data, label


    def train(self):
        '''
        This is the main function for training
        '''

        # For the first step, we are loading all training images and validation images into the
        # memory
        #all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size,augment = USE_AUGMENTATION)
        #all_data, all_labels = self.read_site_data()

        


        # print 'data_shape',all_data.shape
        
        vali_data, vali_labels = read_validation_data()

        vali_data = vali_data[...,[2,1,0]]

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()


        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print 'Restored from checkpoint...'
        else:
            sess.run(init)

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        val_error_list = []

        print 'Start training...'
        print '----------------------------'

        for step in xrange(Total_iteration):
            load_n = step % Total_batchn

            # train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
            #                                                             FLAGS.train_batch_size,augment = USE_AUGMENTATION)

            train_batch_data, train_batch_labels = self.generate_augment_train_batch(load_n,augment = USE_AUGMENTATION)


            validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                                                           vali_labels, FLAGS.validation_batch_size)

            # Want to validate once before training. You may check the theoretical validation
            # loss first
            if step % FLAGS.report_freq == 0:

                if FLAGS.is_full_validation is True:
                    validation_loss_value, validation_error_value, acc_each_c_mean, vali_predict_mtx_int  = self.full_validation(loss=self.vali_loss,
                                            top1_error=self.vali_top1_error, vali_data=vali_data,
                                            vali_labels=vali_labels, session=sess,
                                            batch_data=train_batch_data, batch_label=train_batch_labels)


                    facc = open(acc_file_name,'ab')
                    facc.write("%09d %2.8f %2.8f %2.8f %2.8f, %2.8f %2.8f %2.8f, %2.8f %2.8f %2.8f \n" % (step, acc_each_c_mean[0],acc_each_c_mean[1],acc_each_c_mean[2],acc_each_c_mean[3],\
                    acc_each_c_mean[4],acc_each_c_mean[5],acc_each_c_mean[6],acc_each_c_mean[7],acc_each_c_mean[8],acc_each_c_mean[9]))
                    facc.close()

                    if step % (FLAGS.report_freq*10) == 0:

                        vali_predict_fraction = np.zeros((10,10)) 
                        for mtxii in xrange(10):
                            tempsum = 0
                            for mtxjj in xrange(10):
                                tempsum = tempsum + vali_predict_mtx_int[mtxii,mtxjj]
                            for mtxjj in xrange(10):
                                vali_predict_fraction[mtxii,mtxjj] = vali_predict_mtx_int[mtxii,mtxjj]*1.0 / tempsum




                        fmtx = open(mtx_file_name,'ab')
                        fmtx.write("step: %09d \n" % (step))

                        for mtxi in xrange(10):
                            fmtx.write("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n" % (vali_predict_fraction[mtxi,0],vali_predict_fraction[mtxi,1],vali_predict_fraction[mtxi,2],vali_predict_fraction[mtxi,3],vali_predict_fraction[mtxi,4],\
                                vali_predict_fraction[mtxi,5], vali_predict_fraction[mtxi,6], vali_predict_fraction[mtxi,7], vali_predict_fraction[mtxi,8], vali_predict_fraction[mtxi,9]) )
                        fmtx.close()



                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:
                    _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                     self.vali_top1_error,
                                                                 self.vali_loss],
                                                {self.image_placeholder: train_batch_data,
                                                 self.label_placeholder: train_batch_labels,
                                                 self.vali_image_placeholder: validation_batch_data,
                                                 self.vali_label_placeholder: validation_batch_labels,
                                                 self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)


            start_time = time.time()

            _, _, train_loss_value, train_error_value, acc_eachc_train = sess.run([self.train_op, self.train_ema_op,
                                                           self.full_loss, self.train_top1_error, self.acc_each_sample_train],
                                {self.image_placeholder: train_batch_data,
                                  self.label_placeholder: train_batch_labels,
                                  self.vali_image_placeholder: validation_batch_data,
                                  self.vali_label_placeholder: validation_batch_labels,
                                  self.lr_placeholder: FLAGS.init_lr})

            if step % FLAGS.report_freq == 0:
                eclass_wrong_train = np.zeros(10)
                eclass_right_train = np.zeros(10)
                for templab, temprw in zip(train_batch_labels,acc_eachc_train):
                    if temprw==1:
                        eclass_right_train[templab] = eclass_right_train[templab] + 1
                    else:
                        eclass_wrong_train[templab] = eclass_wrong_train[templab] + 1

                acc_each_c_mean_train = np.zeros(10)
                for ci in xrange(10):
                    acc_each_c_mean_train[ci] = eclass_right_train[ci]/(eclass_right_train[ci]+eclass_wrong_train[ci])

                facc_train = open(acc_file_name_train,'ab')
                facc_train.write("%09d %2.8f %2.8f %2.8f %2.8f, %2.8f %2.8f %2.8f, %2.8f %2.8f %2.8f \n" % (step, acc_each_c_mean_train[0],acc_each_c_mean_train[1],acc_each_c_mean_train[2],acc_each_c_mean_train[3],\
                acc_each_c_mean_train[4],acc_each_c_mean_train[5],acc_each_c_mean_train[6],acc_each_c_mean_train[7],acc_each_c_mean_train[8],acc_each_c_mean_train[9]))
                facc_train.close()

            duration = time.time() - start_time


            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                    self.label_placeholder: train_batch_labels,
                                                    self.vali_image_placeholder: validation_batch_data,
                                                    self.vali_label_placeholder: validation_batch_labels,
                                                    self.lr_placeholder: FLAGS.init_lr})
                summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                    sec_per_batch)
                print 'Train top1 error = ', train_error_value
                print 'Validation top1 error = %.4f' % validation_error_value
                print 'Validation loss = ', validation_loss_value
                print '----------------------------'

                f = open(Record_file,'ab')
                f.write("%06d %2.8f %2.8f %2.8f %2.8f \n" % (step, train_error_value, train_loss_value, validation_error_value, validation_loss_value) )
                f.close()

                step_list.append(step)
                train_error_list.append(train_error_value)



            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print 'Learning rate decayed to ', FLAGS.init_lr

            # Save checkpoints every 10000 steps
            if step % 10000 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                #df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                #                'validation_error': val_error_list})
                #df.to_csv(train_dir + FLAGS.version + '_error.csv')


    
    
    def test(self, test_image_array):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print '%i test batches in total...' %num_batches

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reusef=False,id=self.id)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print 'Model restored from ', FLAGS.test_ckpt_path

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print '%i batches finished!' %step
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reusef=True,id=self.id)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array



    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)


    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset+vali_batch_size]
        return vali_data_batch, vali_label_batch


    def generate_augment_train_batch(self, batch_num, augment = False):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        batch_data = np.load('./data0/'+'img'+('%06d' %(batch_num))+'.npy')
        # batch_data = batch_data*255
        # batch_data = np.rint(batch_data)  
        batch_data = batch_data.astype(np.int32)
        batch_data = batch_data*1.0
        batch_data = batch_data.astype(np.float)
        #batch_data = batch_data[..., [2, 1, 0]]           

        batch_label_one_hot = np.load('./data0/'+'gold_01_plus_label'+('%06d' %(batch_num))+'.npy')
        batch_label = batch_label_one_hot

        rng_state = np.random.get_state()
        np.random.shuffle(batch_data)
        np.random.set_state(rng_state)
        np.random.shuffle(batch_label)

        #print batch_label

        if USE_AUGMENTATION == True:
            padding_size = 2
            pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
            batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)

        if augment == True:
            batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

        batch_data = whitening_image(batch_data)
        #batch_label = train_labels[offset:offset+FLAGS.train_batch_size]

        return batch_data, batch_label




    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op

    def predict_img(self, test_image_array,modelpath):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print '%i test batches in total...' %num_batches

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reusef=False,id=self.id)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.id)
        saver = tf.train.Saver(var_list=var_list)
        sess = tf.Session()

        saver.restore(sess, modelpath)
        print 'Model restored from ', modelpath

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print '%i batches finished!' %step
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        # if remain_images != 0:
        #     self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
        #                                                 IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        #     # Build the test graph
        #     logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reusef=False,id=self.id)
        #     predictions = tf.nn.softmax(logits)

        #     test_image_batch = test_image_array[-remain_images:, ...]

        #     batch_prediction_array = sess.run(predictions, feed_dict={
        #         self.test_image_placeholder: test_image_batch})

        #     prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        eclass_wrong = np.zeros(10)
        eclass_right = np.zeros(10)
        predict_mtx = np.zeros((10,10))

        for step in range(num_batches):

            offset = step * FLAGS.validation_batch_size
            #print vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...].shape
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value, acc_eachc, predicted_labels = session.run([loss, top1_error, self.acc_each_sample, self.predicted_vali_labels], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)
            for templab, temprw in zip(vali_labels_subset[offset:offset+FLAGS.validation_batch_size],acc_eachc):
                if temprw==1:
                    eclass_right[templab] = eclass_right[templab] + 1
                else:
                    eclass_wrong[templab] = eclass_wrong[templab] + 1

            for orig_label, predict_label in zip(vali_labels_subset[offset:offset+FLAGS.validation_batch_size],predicted_labels):
                predict_mtx[orig_label,predict_label] = predict_mtx[orig_label,predict_label] +1

            acc_each_c_mean = np.zeros(10)
            for ci in xrange(10):
                acc_each_c_mean[ci] = eclass_right[ci]/(eclass_right[ci]+eclass_wrong[ci])


        return np.mean(loss_list), np.mean(error_list) , acc_each_c_mean ,  predict_mtx


maybe_download_and_extract()
# Initialize the Train object
with open(Record_file, 'w+') as f:
    #text = f.read()
    #text = re.sub('foobar', 'bar', text)
    #f.seek(0)
    f.write("Step TrainTop1Error TrainLoss TestError TestLoss \n")
    f.truncate()
    f.close()

with open(acc_file_name, 'w+') as f:
    #text = f.read()
    #text = re.sub('foobar', 'bar', text)
    #f.seek(0)
    f.write("iter 0 1 2 3 4 5 6 7 8 9\n")
    f.truncate()
    f.close()

with open(acc_file_name_train, 'w+') as f:
    #text = f.read()
    #text = re.sub('foobar', 'bar', text)
    #f.seek(0)
    f.write("iter 0 1 2 3 4 5 6 7 8 9\n")
    f.truncate()
    f.close()

with open(mtx_file_name, 'w+') as f:
    #text = f.read()
    #text = re.sub('foobar', 'bar', text)
    #f.seek(0)
    f.write("0 1 2 3 4 5 6 7 8 9\n")
    f.truncate()
    f.close()




train = Train_site0()
train.train()
#train.generate_label()
    

    



