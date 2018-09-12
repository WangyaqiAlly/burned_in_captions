# CUDA_VISIBLE_DEVICES='1' python age_sphere1.py
import argparse
import struct
import time
import sys
import os
import numpy as np
print 'numpy ' + np.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print 'tensorflow ' + tf.__version__
import cv2
print 'cv2 ' + cv2.__version__

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nz', help='latent space dimensionality', default=128, type=int)
parser.add_argument('--nc', help='number of channels of input', default=3, type=int)
parser.add_argument('--nef', help='factor of number of channels in encoder', default=64, type=int)
parser.add_argument('--ngf', help='factor of number of channels in generator', default=64, type=int)
parser.add_argument('--ne_updates', help='number of updates for encoder', default=1, type=int)
parser.add_argument('--ng_updates', help='number of updates for generator', default=2, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0002, type=float)
parser.add_argument('--drop_lr', default=4000, type=int, help='')
parser.add_argument('--batch', help='batch size', default=64, type=int)
parser.add_argument('--epochs', help='training epochs', default=2000, type=int)
parser.add_argument('--model', help='output model', default='model.proto.age')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--step', help='step to start training', default=0, type=int)
parser.add_argument('--cat_dim', help='total categorical factors', default=10, type=int)
parser.add_argument('--bn', default=True)


args = parser.parse_args()
print args


tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]

if not os.path.exists(tag):
    os.makedirs(tag)


alpha = 0.2

height = 32
width = 32



# ######################## Load the Data
import cPickle
parent_path = './data/'
fileNames = os.listdir(parent_path)
print fileNames
images = None
labels = None
for f in fileNames:
    if 'data_batch' in f:
        with open(parent_path + '/' + f, 'rb') as fo:
            tempDict = cPickle.load(fo)
        tempArr = tempDict['data']
        tempLabel = tempDict['labels']
        per_images = np.zeros((len(tempLabel), 32, 32, 3))
        per_labels = np.zeros(len(tempLabel), dtype=np.int)
        for i in range(len(tempLabel)):
            im = np.reshape(tempArr[i], (32, 32, 3), order='F')
            rows, cols, _ = im.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
            im = cv2.warpAffine(im, M, (cols, rows))
            per_images[i] = im / 255.0
            per_labels[i] = tempLabel[i]
        if images is None:
            images, labels = per_images, per_labels
        else:
            images = np.concatenate([images, per_images], axis=0)
            labels = np.concatenate([labels, per_labels], axis=0)
print images.shape
d = images[..., [2, 1, 0]]
d = (d-0.5)/0.5
labels = np.eye(10)[labels]

label_name=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


### Designed based on AGE-GAN paper
def enet(args, x, reuse=None):
    print 'encoder network, reuse',reuse
    with tf.variable_scope('enet',reuse=reuse):
        # 32 X 32 X 1 -> 16 X 16 X nef
        e = tf.layers.conv2d(inputs=x, filters=args.nef, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='same')
        e = tf.layers.batch_normalization(e, training=args.bn)
        e = tf.maximum(alpha*e,e) # LeakyReLU
        # 16 X 16 X nef -> 8 X 8 X nef * 2
        e = tf.layers.conv2d(inputs=e, filters=args.nef*2, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='same')
        e = tf.layers.batch_normalization(e, training=args.bn)
        e = tf.maximum(alpha*e,e) # LeakyReLU
        # 8 X 8 X nef*2 -> 4 X 4 X nef * 4
        e = tf.layers.conv2d(inputs=e, filters=args.nef*4, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='same')
        # print e
        e = tf.layers.batch_normalization(e, training=args.bn)
        e = tf.maximum(alpha*e,e) # LeakyReLU
        # 4 X 4 X nef * 4 -> 2 X 2 X nz
        e = tf.layers.conv2d(inputs=e, filters=args.nz, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='same')
        #e = tf.layers.batch_normalization (e, training=args.bn)
        #e = tf.maximum(alpha*e,e) # LeakyReLU

        # 2 X 2 X nz -> 1 X 1 X nz
        e = tf.nn.pool(e, window_shape = [2,2], pooling_type = 'AVG', padding = 'VALID')
        e = normalize(e)
        # ac_output = tf.contrib.layers.flatten(e)
        # ac_output = tf.contrib.layers.fully_connected(ac_output, 10, activation_fn=None)
        ac_output = tf.layers.conv2d(inputs=e, filters=args.cat_dim, kernel_size=1, strides=1, data_format='channels_last',activation=None, padding='same')
        #e = tf.layers.conv2d(inputs=e, filters=args.nz, kernel_size=1, strides=1, data_format='channels_last',activation=None, padding='same')
        e = tf.identity(e,name='eout') ; print e
        ac_output = tf.identity(ac_output, name='elabels'); print ac_output
        return e, ac_output


def gnet(args, z, z_cat, reuse=None):
    print 'generator network, reuse', reuse
    with tf.variable_scope('gnet',reuse=reuse):
        print "Shape of Z:",z.shape
        # 1 X 1 X nz -> 4 X 4 X ngf*8
        # z_input = tf.concat(axis=3, values=[z, z_cat])
        z_input = z
        g = tf.layers.conv2d_transpose(inputs=z_input, filters=args.ngf*8, kernel_size=4, strides=[1,1], padding='valid', data_format='channels_last', activation=None)
        g = tf.layers.batch_normalization(g, training=args.bn)
        g = tf.nn.elu(g) #; print g
        # 4 X 4 X ngf*8 -> 8 X 8 X ngf*4
        g = tf.layers.conv2d_transpose(inputs=g, filters=args.ngf*4, kernel_size=4, strides=[2,2], padding='same', data_format='channels_last', activation=None)
        g = tf.layers.batch_normalization(g, training=args.bn)
        g = tf.nn.elu(g)
        # 8 X 8 X ngf*4 -> 16 X 16 X ngf*2
        g = tf.layers.conv2d_transpose(inputs=g, filters=args.ngf*2, kernel_size=4, strides=[2,2], padding='same', data_format='channels_last', activation=None)
        g = tf.layers.batch_normalization(g, training=args.bn)
        g = tf.nn.elu(g)
        # 16 X 16 X ngf*2 -> 32 X 32 X ngf*2
        g = tf.layers.conv2d_transpose(inputs=g, filters=args.ngf*2, kernel_size=4, strides=[2,2], padding='same', data_format='channels_last', activation=None)
        g = tf.layers.batch_normalization(g, training=args.bn)
        g = tf.nn.elu(g)
        # 32 X 32 X ngf*2 -> 32 X 32 X nc
        g = tf.layers.conv2d(inputs=g, filters=args.nc, kernel_size=4, strides=1,data_format='channels_last',activation=None, padding='same')
        g = tf.nn.tanh(g)
        g = tf.identity(g,name='gout') ; print g
    return g

def kl_loss(args, samples, direction = 'paper', minimize=True):
    s = tf.contrib.layers.flatten(samples)
    mean, var = tf.nn.moments(s, axes = [0])
    t1 = (tf.pow(mean,2) + var*args.nz) 
    t2 = -tf.log(var*args.nz)
    # print '-----MEAN, VAR SHAPE------', mean, var
    kl = tf.reduce_mean((t1 + t2) / 2)
    # return kl
    return kl, tf.reduce_mean(mean), tf.reduce_mean(var)

# def kl_loss(args, samples, direction = 'qp', minimize = True):
# 	s = tf.contrib.layers.flatten(samples)
# 	mean, var = tf.nn.moments(s, axes = [0])
# 	t1 = (tf.pow(mean,2) + tf.pow(var,2)) / 2
# 	t2 = -tf.log(var)
# 	kl = tf.reduce_mean(t1 + t2 - 0.5)
# 	# return kl
# 	return kl, tf.reduce_mean(mean), tf.reduce_mean(var)

# def kl_loss(args, samples, direction = 'dan', minimize=True):
#     s = tf.contrib.layers.flatten(samples)
#     mean, var = tf.nn.moments(s, axes = [0])
#     t1 = (1 + tf.pow(mean,2) + var)
#     t2 = tf.log(var)
#     kl = tf.reduce_mean(t1 + t2 - 0.5)
#     # return kl
#     return kl, tf.reduce_mean(mean), tf.reduce_mean(var)


# def kl_loss_pq(args, samples, direction = 'pq', minimize = True):
# 	s = tf.contrib.layers.flatten(samples)
# 	mean, var = tf.nn.moments(s, axes = [0])
# 	t1 = (1 + tf.pow(mean,2)) / (2 * tf.pow(var,2))
# 	t2 = tf.log(var)
# 	kl = tf.reduce_mean(t1 + t2 - 0.5)
# 	return kl

def match_l2(x, y):
    return tf.reduce_mean(tf.pow(x-y, 2))

def match_l1(x, y):
    return tf.reduce_mean(tf.abs(x-y))

def match_cos(x, y):
    x_norm = normalize(x)
    y_norm = normalize(y)
    return 2.0 - tf.reduce_mean(x_norm * y_norm)


# projects points to a sphere
def normalize(x, axis = 3):
    x_norm = tf.norm(x, axis = axis, keep_dims=True)
    #x_norm = tf.tile(x_norm, tf.shape(x_norm))
    return x / x_norm 

def creat_z_cat(args):
    z_cat = np.zeros((args.batch,args.cat_dim),dtype='float32')
    for i in xrange(args.batch):
        ind=np.where(np.random.multinomial(1, np.ones((args.cat_dim)) / args.cat_dim)==1)[0][0]
        np.put(z_cat[i],ind,1)
    return z_cat

def visualization_color(images, labels, edge_id='', fake_label=None, save_img=False, img_dir='', imgs=None):
    samples_per_class = 3
    generated_label = labels
    classes = np.unique(generated_label)
    # show_label= (-1) * np.ones((samples_per_class, len(classes)))
    show_label = None
    # index = 0
    # For each class, choose `samples_per_class` images to show
    for _class in classes:
        one_class_all_ids = np.where(generated_label == _class)[0]
        if len(one_class_all_ids) != 0:
            one_class_num_ids = np.random.choice(one_class_all_ids, samples_per_class)  # .tolist()
            if imgs is None:
                imgs = np.concatenate(images[one_class_num_ids], axis=0)
                if fake_label is not None:
                    show_label = [fake_label[one_class_num_ids]]
            else:
                imgs = np.concatenate((imgs, np.concatenate(images[one_class_num_ids], axis=0)), axis=1)
                if fake_label is not None:
                    show_label = np.concatenate([show_label, [fake_label[one_class_num_ids]]], axis=0)
                   
    print "show label:", show_label.shape, "class num: ", len(classes)
    # Generate the image, append the label, and show
    img = imgs.astype('uint8')
    width = img.shape[0]
    length = img.shape[1]
    img = cv2.resize(img, (length * 3, width * 3), interpolation=cv2.INTER_CUBIC)
    for i, k in enumerate(classes):
        cv2.putText(img, "{:d}".format(k), (i * 32 * 3, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
   
    if fake_label is not None:
        for i, _class in enumerate(classes):
            for _image in xrange(samples_per_class):
                cv2.putText(img, "{:d}".format(show_label[i, _image]), (i * 32 * 3, _image * 32 * 3 + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('Images' + edge_id, img)

    if save_img:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = img_dir + '/' + edge_id + '_{:%Y.%m.%d.%H.%M.%S.%f}.png'.format(datetime.datetime.now())
        cv2.imwrite(img_path, img)
    cv2.waitKey(10)
    return img

######################## Build the Graph
x = tf.placeholder('float32', [None,width,height,3],name='x') ; print x
z = tf.placeholder('float32', [None,1,1,args.nz],name='z') ; print z
l = tf.placeholder('float32', [None,1,1, 10], name='l'); print l
z_cat = tf.placeholder('float32', [None,1,1,10], name='z_cat'); print z_cat

z = normalize(z)
e_x, e_l = enet(args, x) # e(x)
g_e_x = gnet(args, e_x, e_l) # g(e(x)) 

g_z = gnet(args, z, z_cat, reuse = True)
e_g_z, e_g_z_l = enet(args, g_z, reuse = True) # e(g(z))

# kl_real = kl_loss(args, e_x, minimize = True) ; print kl_real
# kl_fake_e = -kl_loss(args, e_g_z, minimize = False) ; print kl_fake_e
# kl_fake_g = kl_loss(args, e_g_z, minimize = True) ; print kl_fake_g

kl_real, ex_mean, ex_var = kl_loss(args, e_x, minimize = True) ; print kl_real
kl_fake_e, egz_mean, egz_var = kl_loss(args, e_g_z, minimize = False) ; print kl_fake_e
kl_fake_e *= -1
kl_fake_g, _, _ = kl_loss(args, e_g_z, minimize = True) ; print kl_fake_g


ac_lamda = 1.0
# closs = 100 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat([l, z_cat], axis=0), logits=tf.concat([e_l, e_g_z_l], axis=0)))
closs = tf.Variable(0.0, trainable=False, dtype=tf.float32)

match_x = 10 * match_l1(g_e_x, x)
match_z = 1000 * match_cos(e_g_z, z)
eloss = tf.add(kl_real, kl_fake_e) + match_x #+ match_cos(e_g_z, z); print eloss # + match(g_e_x, x, 'L1') * 0 + match(e_g_z, z, 'cos') * 0
print eloss
gloss = kl_fake_g + match_z; print gloss#+ match(e_g_z, z, 'cos') * 1

global_step = tf.Variable(0, name='global_step', trainable=False)
step_per_droplr = d.shape[0]/args.batch * args.drop_lr
learning_rate = tf.multiply(tf.pow(2.0, -tf.cast(tf.cast(tf.div((global_step + 1), step_per_droplr), tf.int32), tf.float32)), args.lr)
# Have question: is this the upper bound?

eopt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1= 0.5)
egrads = eopt.compute_gradients(eloss + closs,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
etrain = eopt.apply_gradients(egrads, global_step=global_step)
enorm = tf.global_norm([i[0] for i in egrads])

gopt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1= 0.5)
ggrads = gopt.compute_gradients(gloss + closs,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet'))
gtrain = gopt.apply_gradients(ggrads)
gnorm = tf.global_norm([i[0] for i in ggrads])


saver = tf.train.Saver()
step = args.step


######################## Train the Model
with tf.Session() as sess:
    if step == 0:
        sess.run(tf.global_variables_initializer())
    else:
        model_path = os.path.join(tag, 'model-' + str(step))
        saver.restore(sess, model_path)
    for i in range(args.epochs):

        ind = np.arange(d.shape[0])
        np.random.shuffle(ind)
        d = d[ind]
        labels = labels[ind]
        #d = tf.random_shuffle(d)

        for j in range(0,int(d.shape[0])/int(args.batch)*args.batch,args.batch):
            #print j
            input_x = d[j:j+args.batch]
            input_l = labels[j: j+args.batch].reshape(args.batch, 1, 1, args.cat_dim)

            # train the encoder
            for _ in range(args.ne_updates):
                input_z = np.random.randn(args.batch,args.nz)
                input_z = input_z / np.sqrt((input_z*input_z).sum(axis=1))[:,np.newaxis]	
                input_z = input_z.reshape(args.batch,1,1,args.nz)
                input_zcat = creat_z_cat(args).reshape(args.batch, 1, 1, args.cat_dim)

                # _, el, en, kl_real_, kl_fake_e_, match_x_ = sess.run([etrain,eloss,enorm, kl_real, kl_fake_e, match_x], 
                _, el, en, kl_real_, kl_fake_e_, match_x_, ex_mean_, ex_var_, egz_mean_, egz_var_, egzl_ = sess.run([
                    etrain,eloss,enorm,
                    kl_real, kl_fake_e, match_x, 
                    ex_mean, ex_var, egz_mean, egz_var, e_g_z_l], 
                    feed_dict={x:input_x, 
                                z:input_z,
                                l:input_l,
                                z_cat: input_zcat})
                        # print "####################x.shape: ", input_x.shape
            # train the generator
            for _ in range(args.ng_updates):
                input_z = np.random.randn(args.batch,args.nz)
                input_z = input_z / np.sqrt((input_z*input_z).sum(axis=1))[:,np.newaxis]	
                input_z = input_z.reshape(args.batch,1,1,args.nz)
                input_zcat = creat_z_cat(args).reshape(args.batch, 1, 1, args.cat_dim)

                        # print '##################z.shape: ', input_z.shape
                _, gl, gn, kl_fake_g_, match_z_, cl = sess.run([gtrain,gloss,gnorm, kl_fake_g, match_z, closs], 
                    feed_dict={x:input_x, 
                                z:input_z,
                                l:input_l,
                                z_cat: input_zcat})

            t=1.
            step += 1
            if (step % 50) == 0:
                savepath = saver.save(sess, os.path.join(tag, 'model'), global_step=step, write_meta_graph=False)
                print 'saving ',savepath
            print '\nepoch %d image number %.5f ecost %.5f gcost %.5f lcost %.5f'% (i, j, el/t, gl/t, cl/t)
            print 'enorm',en/t,'gnorm',gn/t, 'elr', sess.run(eopt._lr), 'glr', sess.run(gopt._lr)
            print 'kl_real %.5f, kl_fake_e %.5f, match_x %.5f' % (kl_real_, kl_fake_e_, match_x_)
            print 'kl_fake_g %.5f, match_z %.5f' % (kl_fake_g_, match_z_)
            print 'ex_mean %.5f, ex_var %.5f, egz_mean %.5f, egz_var %.5f' % (ex_mean_, ex_var_, egz_mean_, egz_var_)

            if j%1000 == 0:
                input_z = np.random.randn(args.batch,args.nz)
                input_z = input_z / np.sqrt((input_z*input_z).sum(axis=1))[:,np.newaxis]    
                # print "-------input z: min %.2f, max %.2f----------" % (input_z.min(), input_z.max())
                input_z = input_z.reshape(args.batch,1,1,args.nz)
                input_zcat = creat_z_cat(args)
                xgen = sess.run(g_z,
                    feed_dict={x:input_x,
                    z:input_z,
                    l:input_l,
                    z_cat: input_zcat.reshape(args.batch, 1, 1, args.cat_dim)})
                xgen = np.clip(xgen / 2. + 0.5 ,0., 1.)*255.
                realImg = np.clip(d / 2. + 0.5, 0., 1.)*255.
                img_xgen = visualization_color(xgen, np.argmax(input_zcat, axis=1), '_generated', np.argmax(np.squeeze(egzl_), axis=1))
                # img_real = visualization_color(realImg, np.argmax(labels, axis=1), '_real', np.argmax(labels, axis=1))
                k = cv2.waitKey(100)
                if k==1114083: # ctrl-c to exit
                    break
        pngfname = "%s-%d.png" % (tag, i)
        pathname = os.path.join(tag, pngfname)
        cv2.imwrite(pathname, img_xgen)

        # write model, redirect stderr to supress annoying messages
        with open(os.devnull, 'w') as sys.stdout:
            graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout'])
        sys.stdout=sys.__stdout__
        tf.train.write_graph(graph, '.', args.model, as_text=False)





