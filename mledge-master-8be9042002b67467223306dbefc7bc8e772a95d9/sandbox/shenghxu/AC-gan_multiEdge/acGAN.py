import tensorflow as tf
from EdgeNode import EdgeNode
import sys


decay_steps = 10000
decay_rate = 0.95


class acGAN(EdgeNode):
    def __init__(self, sess, graph_dir, edge_id, image_type, time, record_path,
                 batch_size, learning_rate, label_rate,cat_dim,con_dim,latent_dim):
        EdgeNode.__init__(self, sess, graph_dir, edge_id, image_type, time, record_path,
                          batch_size, learning_rate, label_rate,cat_dim,con_dim,latent_dim)
    def leaky_relu(x):
        return tf.where(tf.greater(x, 0), x, 0.2 * x)

    def _build_network(self):
        self.logger.debug('-------------Begin building network --------------------')

        with tf.variable_scope('input_' + str(self.edge_id)):
            self.x = tf.placeholder('float32', [None, 32, 32, 3], name='x')
            self.phase = tf.placeholder(tf.bool,name='phase')
            self.keep_rate = tf.placeholder('float32',name='keep_rate')
            #self.latent_dim = tf.placeholder(tf.int32,name='latent_dim')
            self.z = tf.placeholder('float32', [None, self.latent_dim], name='z')
            self.logger.debug('z: %s' % (self.z,))
            self.l = tf.placeholder('float32', [None, 10], name='l')
            
            self.z_cat = tf.placeholder('float32',[None,self.cat_dim],name='z_cat')
            self.z_con = tf.placeholder('float32',[None,self.con_dim],name='z_con')
            self.z_rand = tf.placeholder('float32',[None,self.latent_dim],name='z_rand')


        self.z  = tf.concat(axis=1, values=[self.z_cat, self.z_con, self.z_rand]) 

        self.ex,self.cat_real,_ = self.enet(self.x,0.5,self.phase)
        self.logger.debug('------------ DISCRIMINATOR NET constructed --------------')

        self.gx, self.gl = self.gnet(self.z,self.phase)
        self.logger.debug('------------- GENERATOR NET constructed -----------------')

        self.egz,self.cat_fake,self.con_fake = self.enet(self.gx,0.5, self.phase, reuse=True)
        self.logger.debug('------------- egz NET constructed ----------------------')

        self.egz2,self.cat_fake_full,self.con_fake2 = self.enet(self.gx,0.5,self.phase,reuse=True)

        self.eloss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.ex,0.00000001,sys.float_info.max)) \
            + tf.log(tf.clip_by_value((1.-self.egz),0.00000001,sys.float_info.max)))
        
        self.gloss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.egz,0.00000001,sys.float_info.max)))
        
        self.closs = tf.reduce_mean( 0.5* tf.nn.softmax_cross_entropy_with_logits(labels=self.z_cat,logits=self.cat_fake) \
           +1/self.label_rate *0.5* tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=self.l,logits=self.cat_real),tf.reduce_sum(self.l,axis=1)) )

        self.conloss = tf.losses.mean_squared_error(labels=self.z_con,predictions=self.con_fake)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, decay_steps,
                                                        decay_rate, staircase=True)

        self.eopt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        self.egrads = self.eopt.compute_gradients(self.eloss+self.closs+self.conloss,
                                                  var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                             'enet_' + str(self.edge_id)))
        self.etrain = self.eopt.apply_gradients(self.egrads, global_step=self.global_step)
        self.enorm = tf.global_norm([i[0] for i in self.egrads])

        self.gopt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        self.ggrads = self.gopt.compute_gradients(self.gloss+self.closs+self.conloss,
                                                  var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                             'gnet_' + str(self.edge_id)))
        self.gtrain = self.gopt.apply_gradients(self.ggrads, global_step=self.global_step)
        self.gnorm = tf.global_norm([i[0] for i in self.ggrads])

        self.saver = tf.train.Saver()

    def enet(self, x,keep_rate,phase,reuse=None):
        with tf.variable_scope('enet_'+ str(self.edge_id),reuse=reuse):
            e = tf.layers.conv2d(inputs=x, filters=16, kernel_size=3, strides=2,activation=None, padding='same') ;
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
            e = tf.nn.dropout(e,keep_rate)
            e = tf.layers.conv2d(inputs=e, filters=32, kernel_size=3, strides=1,activation=None, padding='same') ; 
            e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn')
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
        
            e = tf.nn.dropout(e,keep_rate)
        
            #e = tf.image.resize_bilinear(images=e,size=[16,16]) ; print e
            e = tf.layers.conv2d(inputs=e, filters=64, kernel_size=3, strides=2,activation=None, padding='same') ; 
            e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn2')
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
            e = tf.nn.dropout(e,keep_rate)
        
            e = tf.layers.conv2d(inputs=e, filters=128, kernel_size=3, strides=1,activation=None, padding='same') ; 
            e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn3')
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
            e = tf.nn.dropout(e,keep_rate)
        
            #e = tf.image.resize_bilinear(images=e,size=[8,8]) ; print e
            e = tf.layers.conv2d(inputs=e, filters=256, kernel_size=3, strides=2,activation=None, padding='same'); 
            e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn4')
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
            e = tf.nn.dropout(e,keep_rate)
        
            e = tf.layers.conv2d(inputs=e, filters=512, kernel_size=3, strides=1,activation=None, padding='same') ; 
            e = tf.contrib.layers.flatten(e) ; print e
        
            cat = tf.layers.dense(inputs=e,units=10, activation=None); 
            con = tf.layers.dense(inputs=e,units=2,  activation=tf.nn.sigmoid); 
            e = tf.layers.dense(inputs=e, units=10, activation=tf.nn.elu) ; 
            e = tf.layers.dense(inputs=e, units=1, activation=tf.sigmoid) ;
            e = tf.identity(e,name='eout') ; 
            self.logger.debug('e: %s' % (e,))
        return e, cat, con

    def gnet(self, z,phase, reuse=None):
        with tf.variable_scope('gnet_' + str(self.edge_id), reuse=reuse):
            ind = tf.constant([0,1,2,3,4,5,6,7,8,9])
            #l = tf.transpose(tf.nn.embedding_lookup(tf.transpose(z),ind))
            l = tf.transpose(tf.nn.embedding_lookup(tf.transpose(z),ind))
            gl = tf.identity(l,name='glout')
            g = tf.layers.dense(inputs=z, units=384, activation=tf.nn.relu) ; print g
            g = tf.reshape(g,[-1,4,4,24]) ; print g
            g = tf.layers.conv2d_transpose(inputs=g, filters=192, kernel_size=5, strides=2,activation=None, padding='same') ; print g
            g = tf.contrib.layers.batch_norm(g,center=True,scale=True,is_training=phase,scope='gbn1')
            g = tf.nn.relu(g)
            g = tf.layers.conv2d_transpose(inputs=g, filters=96, kernel_size=5, strides=2,activation=None, padding='same') ; print g
            g = tf.contrib.layers.batch_norm(g,center=True,scale=True,is_training=phase,scope='gbn2')
            g = tf.nn.relu(g)
            #g = tf.image.resize_bilinear(images=g,size=[16,16]) ; print g
            g = tf.layers.conv2d_transpose(inputs=g, filters=3, kernel_size=5, strides=2,activation=tf.nn.tanh, padding='same') ; print g
            self.logger.debug('g: %s' % (g,))
            gx = tf.identity(g,name='gxout')
        return gx,gl
