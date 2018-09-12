import tensorflow as tf
from model_test import *
from EdgeNode import EdgeNode
import Utils as utils

num_units_per_layer = 16
latent_dimension = 128
decay_steps = 10000
decay_rate = 0.95
n_critic = 5
cat_dim = 10

class wacGAN(EdgeNode):
    def __init__(self, sess, graph_dir, edge_id, image_type, 
        data_start_time=1, data_end_time=10, record_path='./record',
        batch_size=100, learning_rate=0.0001, save_images=True, image_path='./images'):
        EdgeNode.__init__(self, sess, graph_dir, edge_id, image_type, 
            data_start_time, data_end_time, record_path,
            batch_size, learning_rate, save_images, image_path)
        # self.gnet = gnet
        # self.enet_wgan=enet_wgan
    def build_network(self):
        self.logger.debug('-------------Begin building network --------------------')

        with tf.name_scope('input_' + str(self.edge_id)):
            self.x = tf.placeholder('float32', [None, 3, 32, 32], name='x')
            self.z = tf.placeholder('float32', [None, latent_dimension], name='z')
            self.logger.debug('z: %s' % (self.z,))
            self.l = tf.placeholder('int32', [None, ], name='l')
            self.z_cat = tf.placeholder('int32', [None, ], name='z_cat')

        gx = self.gnet(self.batch_size,self.l)
        real_and_fake_label = tf.concat([self.l,self.l],axis=0)
        real_and_fake_data = tf.concat([self.x,gx],axis=0)
        ex_egz,cat_real_cat_fake = self.enet_wgan(real_and_fake_data,real_and_fake_label)
        self.ex = ex_egz[:self.batch_size]
        self.egz = ex_egz[self.batch_size:]
        cat_real = cat_real_cat_fake[:self.batch_size]
        cat_fake = cat_real_cat_fake[self.batch_size:]
        self.eloss = tf.reduce_mean(self.egz)-tf.reduce_mean(self.ex)
        alpha = tf.random_uniform(
                    shape=[self.batch_size], 
                    minval=0.,
                    maxval=1.
                )
        differences = gx - self.x
        interpolates = self.x + (alpha[0]*differences)
        gradients = tf.gradients(self.enet_wgan(interpolates,self.l)[0], [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1])+1e-7)
        self.gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
        self.eloss = self.eloss + self.gradient_penalty
        self.closs_e = 40*tf.reduce_mean(0.5* tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.l,logits=cat_fake) \
                + 0.5* tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.l,logits=cat_real))
        # self.closs_e = 40*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.l,logits=cat_real))

        self.gx_train = self.gnet(self.batch_size,self.z_cat) # g(z)
        egz_train,self.cat_fake_train = self.enet_wgan(self.gx_train,self.z_cat)
        self.gloss = -tf.reduce_mean(egz_train)
        self.closs_g = 40*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.z_cat,logits=self.cat_fake_train))

        self.logger.debug('------------- Apply gradients ----------------------')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, decay_steps,
                                                        decay_rate, staircase=True)


        self.eopt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.,beta2=0.9)
        
        self.logger.debug("E_optimizer: %s" % (self.eopt,))
        # self.print_vars()
        self.egrads = self.eopt.compute_gradients(self.eloss+self.closs_e, 
                                                  var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                             'enet'))
        self.etrain = self.eopt.apply_gradients(self.egrads, global_step=self.global_step)
        self.enorm = tf.global_norm([i[0] for i in self.egrads])

        self.gopt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.,beta2=0.9)
        self.ggrads = self.gopt.compute_gradients(self.gloss+self.closs_g,
                                                  var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                             'gnet'))
        self.gtrain = self.gopt.apply_gradients(self.ggrads, global_step=self.global_step)
        self.gnorm = tf.global_norm([i[0] for i in self.ggrads])

        self.saver = tf.train.Saver()



    def gnet(self,n_samples, labels, noise=None,reuse=None):

        with tf.name_scope('gnet'):

            if noise is None:
                noise = tf.random_normal([n_samples, 128])
            l = tf.one_hot(labels,10,on_value=1.0,off_value=0.0,axis=-1,dtype='float32')
            input = tf.concat([noise,l],axis=1)
            output = lib.ops.linear.Linear('Generator.Input', 128+10, 4*4*DIM_G, input)
            output = tf.reshape(output, [-1, DIM_G, 4, 4])
            output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
            output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
            output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
            output = Normalize('Generator.OutputN', output)
            output = nonlinearity(output)
            output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
            output = tf.tanh(output)
            output = tf.identity(output,name='gout')
            self.logger.debug("gnet_out: %s", (output,))
            #return tf.reshape(output, [-1, OUTPUT_DIM])
            return output

    def enet_wgan(self,inputs, labels,reuse=None):
        with tf.name_scope('enet'):
            output = tf.reshape(inputs, [-1, 3, 32, 32])
            output = OptimizedResBlockDisc1(output)
            output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
            output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
            output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
            output = nonlinearity(output)
            output = tf.reduce_mean(output, axis=[2,3])
            output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
            output_wgan = tf.reshape(output_wgan, [-1])
            output_wgan = tf.identity(output_wgan,name='eout_w')
            if ACGAN:
                output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)
                output_acgan = tf.identity(output_acgan,name='eout_ac')
                self.logger.debug("enet_ac_out: %s", (output_acgan,))
                return output_wgan, output_acgan
            else:
                self.logger.debug("enet_out: %s", (output_acgan,))
                return output_wgan, None

    def train(self, total_epoch):

        for i in range(total_epoch):
            # Shuffle data
            data = self.train_dataset.images
            data = data.transpose([0,3,1,2])
            data = data.astype('float32')
            label = self.train_dataset.labels
            rng_state = np.random.get_state()
            np.random.shuffle(self.train_dataset.images)
            np.random.set_state(rng_state)
            np.random.shuffle(self.train_dataset.labels)
            #label = np.eye(10)[label]

            elb = 0.
            glb = 0.
            cleb=0.
            clgb=0.
            gpb=0.
            enb = 0.
            gnb = 0.
            et = 0.
            gt = 0.
            for j in range(0,(data.shape[0]/(self.batch_size * n_critic))*(self.batch_size * n_critic),self.batch_size * n_critic):
                for k in range(n_critic):
                    ind = j+k*self.batch_size

                    _,el_,cle_,gp_,en_ = self.sess.run([self.etrain,self.eloss,self.closs_e,self.gradient_penalty,self.enorm],feed_dict={self.x:data[ind:ind+self.batch_size],
                                                                     self.l:label[ind:ind+self.batch_size]})
                    et += 1.
                    elb += el_
                    enb += en_
                    gpb+=gp_
                    cleb+=cle_
                    # print el_, cle_, gp_
                    # print ex_,egz_

                _, gl_, clg_,gn_ = self.sess.run([self.gtrain, self.gloss, self.closs_g,self.gnorm],
                                            feed_dict={self.x:data[j:j+self.batch_size],
                                                       self.l:label[j:j+self.batch_size],
                                                       self.z_cat: np.random.randint(0,cat_dim,self.batch_size)})
                gt += 1.
                glb += gl_
                gnb += gn_
                clgb+=clg_
                # print gl_
                utils.update_progress(i, int((j + 1.0) / ((data.shape[0]/(self.batch_size * n_critic))*(self.batch_size * n_critic) * 1.0) * 1000 + 1) / 1000.0)
            utils.update_progress(i, 1)


            self.logger.info('Edge {:2s}: epoch {:5d}  eloss {:12.8f}  gloss {:12.8f}'
                             'egrad {:12.8f}  ggrad {:12.8f}  learning_rate {:12.8f}'
                             'closs_e {:12.8f} closs_g {:12.8f} gradient_penalty {:12.8f}'
                             .format(self.edge_id, i, elb / et, glb / gt, enb / et, gnb / gt, self.sess.run(self.learning_rate), cleb/et, clgb/gt, gpb/et))
            z_cat_temp = np.concatenate([np.array(xrange(10))]*10)
            x0, l0 = self.sess.run([self.gx_train, self.cat_fake_train],
                                   feed_dict={self.z_cat:z_cat_temp})
            x0 = (np.clip(x0, -0.5, 0.5)+0.5) * 255.
            x0=x0.transpose([0,2,3,1])
            utils.visualization_color(x0,z_cat_temp,save_img=self.save_images,img_dir=self.image_path,edge_id=self.edge_id)
            l0 = np.argmax(l0, axis=1)
