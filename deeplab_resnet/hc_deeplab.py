"""
@author: Tae-Hyun Oh (http://taehyunoh.com, taehyun@csail.mit.edu)
@date: Jul 29, 2018
@description: This is a part of the semantic feature extraction implementation used in 
[Semantic Soft Segmentation (Yağız et al., 2018)] (project page: http://people.inf.ethz.ch/aksoyy/sss/).
This code is for protyping research ideas; thus, please use this code only for non-commercial purpose only.  
"""


import os
import time
import tensorflow as tf
import numpy as np

from tensorflow.python.keras._impl.keras.initializers import he_normal
from tensorflow.python import debug as tf_debug

from .base import Model
from .image_reader import read_data_list, get_indicator_mat, get_batch_1chunk, get_batch, get_batch_1chunk

from .utils import inv_preprocess
from .model import DeepLabResNetModel
from .loader import load_single_image, load_batch_samplepts

# Loader
DIR_ANNOTATION = 'anno_png'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NINST = 3
NSAMPLEPTS = 500
DROPOUT_PROB = 1
FEAT_DIM = 128

#######################################################
'''
Helper functions
'''

def lowrank_linear(input_, dim_bottle, dim_out, name="lowrank_linear"):
	with tf.variable_scope(name):
		weights1 = tf.get_variable("fc_weights1", [input_.get_shape()[-1], dim_bottle], initializer=he_normal())
		weights2 = tf.get_variable("fc_weights2", [dim_bottle, dim_out], initializer=he_normal())
		biases = tf.get_variable("biases", [dim_out], initializer=tf.constant_initializer(0.01))

		activation = tf.add(tf.matmul(tf.matmul(input_, weights1), weights2), biases)
	return activation
	
def linear(input_, dim_out, name="linear"):
	with tf.variable_scope(name):
		weights1 = tf.get_variable("fc_weights1", [input_.get_shape()[-1], dim_out], initializer=he_normal())
		biases = tf.get_variable("fc_biases", [dim_out], initializer=tf.constant_initializer(0.01))

		activation = tf.add(tf.matmul(input_, weights1), biases)
	return activation
		

def conv2d(input_, output_dim, 
		   k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
		   name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
				  initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		# conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
		conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

		return conv

#######################################################
'''
HyperColumn architure class definition
'''

class HyperColumn_Deeplabv2(Model):
	"""HyperColumn_Deeplabv2."""
	def __init__(self, sess, args):
		"""Initialize the parameters.
			sess: tensorflow session
		"""
		self.sess = sess
		self.batch_size = args.batch_size
		self.args = args
		
		# parameters used to save a checkpoint
		self.dataset = "Hypcol"
		self.options = []
		self._attrs = ['batch_size', 'dataset']

		self.build_model()


	def fn_map2visualization(self, full_embedding, sz2d):
		randproj = tf.random_normal([FEAT_DIM,3])

		visualized = tf.matmul(full_embedding, randproj)
		tensorshape = tf.concat([tf.constant([-1]), sz2d, tf.constant([3])], 0)

		visualized = tf.reshape(visualized, tensorshape)
		maxval = tf.reduce_max(visualized)
		minval = tf.reduce_min(visualized)
		visimg = tf.truediv(visualized - minval, maxval-minval)*255.0
		return visimg

# Deprecated
	def lossfunction(self, tweightmat, tindicator, tembeddings):

		with tf.variable_scope('loss_computation') as scope:
			# tembeddings: #pts x 64
			sqrvals = tf.reduce_sum(tf.square(tembeddings), 1, keep_dims=True)
			# sqrvals: #pts x 1
			sqrvalsmat = tf.tile(sqrvals, [1, tf.shape(sqrvals)[0]])
			sqrvalsmat2 = tf.add(sqrvalsmat,tf.transpose(sqrvalsmat))
			distmat =  tf.add(sqrvalsmat2, tf.scalar_mul(-2.0, tf.matmul(tembeddings,  tf.transpose(tembeddings))))/64.0

			sigmamat = tf.scalar_mul(2.0, tf.reciprocal(1.0+tf.exp(distmat)))
			posnegmapping = tf.log(tf.add(tf.scalar_mul(0.5, 1.0-tindicator), tf.multiply(tindicator, sigmamat)))
			wcrossentropy = tf.multiply(tf.negative(tindicator+2.0), posnegmapping)
			lossval = tf.reduce_mean(wcrossentropy)
		return lossval

	def build_model(self):
		args = self.args

		npindicator = get_indicator_mat(NSAMPLEPTS, NINST)

		# TF part: Input feeding
		self.netcontainer = dict()
		tinput_img = tf.placeholder(tf.float32,shape=(None,None,3),name='feed_img')
		self.netcontainer['input_img'] = tinput_img

		sz_lossmat = (NSAMPLEPTS*NINST,NSAMPLEPTS*NINST)
		tlossweight = tf.placeholder(tf.float32, shape=sz_lossmat, name='const_weight')
		tlossindicator = tf.constant(npindicator, dtype=tf.float32, name='const_indicator')
		tsample_points = tf.placeholder(tf.float32, shape=(NSAMPLEPTS*NINST, 2),name='feed_x2d')
		self.netcontainer['input_weightmat'] = tlossweight
		self.netcontainer['input_samplepts'] = tsample_points

		input_img = tf.expand_dims(tinput_img, dim=0)
		
		# Create network.
		with tf.variable_scope('', reuse=False):
			net = DeepLabResNetModel({'data': input_img}, is_training=self.args.is_training, num_classes=self.args.num_classes)
		
		self.netcontainer['net'] = net

		t_imsz = tf.shape(input_img)[1:3]

		with tf.variable_scope('hypercolumn_layers') as scope:
			raw_color = conv2d(input_img/128.0, 4, k_h=1, k_w=1, d_h=1, d_w=1, name="hc_cv_0")
			raw_1 = tf.image.resize_bilinear(conv2d(net.layers['pool1'], 124, k_h=1, k_w=1, d_h=1, d_w=1, name="hc_cv_1"), t_imsz)
			raw_3 = tf.image.resize_bilinear(conv2d(net.layers['res3b3_relu'], 128, k_h=1, k_w=1, d_h=1, d_w=1, name="hc_cv_3"), t_imsz)
			raw_4 = tf.image.resize_bilinear(conv2d(net.layers['res4b22_relu'], 256, k_h=3, k_w=3, d_h=1, d_w=1, name="hc_cv_4"), t_imsz)
			raw_5 = tf.image.resize_bilinear(conv2d(net.layers['res5c'], 512, k_h=3, k_w=3, d_h=1, d_w=1, name="hc_cv_5"), t_imsz)

			raw_output = tf.nn.relu(tf.concat([raw_color, raw_1, raw_3, raw_4, raw_5], 3))
			
			nfeatdim = raw_output.get_shape()[-1]
			full_activation = tf.reshape(raw_output, [-1, nfeatdim])
			
			# FC layes
		with tf.variable_scope('fc1_matting') as scope:
			full_act1 = tf.nn.relu(lowrank_linear(full_activation, 256, 512, name="linear"))
			
		with tf.variable_scope('fc2_matting') as scope:
			full_act2 = tf.nn.relu(lowrank_linear(full_act1, 256, 512, name="linear"))

		with tf.variable_scope('fc3_matting') as scope:
			full_input3 = tf.concat([full_act1, full_act2], -1) 	# similar to DenseNet
			full_embedding = linear(full_input3, FEAT_DIM)
			

		# embeddings: #pts x FEAT_DIM
		visimg = self.fn_map2visualization(full_embedding, tf.shape(raw_output)[1:3])
		
		outshape = tf.concat([tf.shape(raw_output)[0:3], tf.constant([-1])], 0)
		self.netcontainer['out_hypcol'] = tf.reshape(full_embedding, outshape)
		self.netcontainer['out_visimg'] = visimg



# Deprecated
	def setup_optimizer(self):

		args = self.args
		# Which variables to load. Running means and variances are not trainable,
		# thus all_variables() should be restored.
		restore_var = [v for v in tf.global_variables()]

		all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
		fc_trainable = [v for v in all_trainable if 'fc' in v.name]
		conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
		fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
		fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
		assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
		assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
	 
		# Define loss and optimisation parameters.
		base_lr = tf.constant(args.learning_rate)
		step_ph = tf.placeholder(dtype=tf.float32, shape=(), name='ph_step')
		learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
		
		opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
		opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
		opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

		# Define a variable to accumulate gradients.
		accum_grads = [tf.Variable(tf.zeros_like(v.initialized_value()),
															 trainable=False) for v in conv_trainable + fc_w_trainable + fc_b_trainable]

		# Define an operation to clear the accumulated gradients for next batch.
		zero_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]

		# Compute gradients.
		grads = tf.gradients(self.loss, conv_trainable + fc_w_trainable + fc_b_trainable)
	 
		# Accumulate and normalise the gradients.
		accum_grads_op = [accum_grads[i].assign_add(tf.scalar_mul(1.0/np.float32(args.grad_update_every), grad)) for i, grad in
											 enumerate(grads) if grad is not None]

		grads_conv = accum_grads[:len(conv_trainable)]
		grads_fc_w = accum_grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
		grads_fc_b = accum_grads[(len(conv_trainable) + len(fc_w_trainable)):]

		# Apply the gradients.
		train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
		train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
		train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

		train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
		
		self.train_container = dict()
		self.train_container['train_op'] = train_op
		self.train_container['acc_grads_op'] = accum_grads_op
		self.train_container['zero_op'] = zero_op
		self.train_container['restore_var'] = restore_var
		self.train_container['step_ph'] = step_ph
		



# Deprecated
	def train(self):
		"""Training code.
		"""
		args = self.args
		self.max_iter = args.num_steps
		self.checkpoint_dir = args.snapshot_dir
		self.imgflist = read_data_list(os.path.join(args.data_dir, 'img'), args.data_list, '.jpg')
		self.labelmapflist = read_data_list(os.path.join(args.data_dir, DIR_ANNOTATION), args.data_list, '.png')

		## Image, Labelmap loader
		h, w = map(int, args.input_size.split(','))
		input_size = (h, w)
		loader_img = load_single_image(args, input_size)
		caller_imgloader = [loader_img['output_img'], loader_img['output_labelmap']]

		## Point sampler
		pt_sampler = load_batch_samplepts()
		caller_sampler = [pt_sampler['out_batchx2d'], pt_sampler['out_weightmat']]

		# Pixel-wise softmax loss.
		l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
		self.loss = tf.add_n(l2_losses)

		# Processed predictions: for visualisation.
		pred = tf.cast(tf.image.resize_bilinear(self.netcontainer['out_visimg'], input_size), tf.uint8)

		# Image summary.
		images_summary = tf.py_func(inv_preprocess, [tf.expand_dims(self.netcontainer['input_img'], dim=0), args.save_num_images, IMG_MEAN], tf.uint8)

		total_summary = tf.summary.image('images', tf.concat(axis=2, values=[images_summary, pred]), 
												 max_outputs=args.save_num_images) # Concatenate row-wise.
		summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())
		self.setup_optimizer()


		self.step = self.train_container['step_ph']

		tf.global_variables_initializer().run()
		self.load(self.checkpoint_dir)

		start_time = time.time()
		start_iter = self.step.eval()

		nimg = len(self.imgflist)    

		# Iterate over training steps.
		for step in range(0, args.num_steps):
			start_time = time.time()
			feed_dict = { self.step : step }
			loss_value = 0

			# Clear the accumulated gradients.
			sess.run(self.train_container['zero_op'], feed_dict=feed_dict)
		 
			# Image loading
			feed_dict_imgloader = {loader_img['input_img_name']: self.imgflist[step%nimg], 
														 loader_img['input_lbm_name']: self.labelmapflist[step%nimg]}
			cur_image, cur_labelmap = sess.run(caller_imgloader, feed_dict=feed_dict_imgloader)
			
			if len(np.unique(cur_labelmap)) < NINST:
				continue
			
			print('Loaded image: %s' % self.imgflist[step%nimg])

			# Accumulate gradients.
			for i in range(args.grad_update_every):
				# Point sampling
				feed_dict_sampler = {pt_sampler['input_labelmap']: cur_labelmap}
				batchx2d, weightmat = sess.run(caller_sampler, feed_dict=feed_dict_sampler)
				
				# print('Sampled %d' % i)

				feed_dict_backprob = {self.netcontainer['input_img']: cur_image, 
															self.netcontainer['input_weightmat']: weightmat,
															self.netcontainer['input_samplepts']: batchx2d,
															self.step : step}

				_, l_val = sess.run([self.train_container['acc_grads_op'], self.loss], feed_dict=feed_dict_backprob)
				loss_value += l_val

			# Normalise the loss.
			loss_value /= args.grad_update_every

			# Apply gradients.
			if step % args.save_pred_every == 0:
				print('Summary')
				feed_dict_summary = {self.netcontainer['input_img']: cur_image, 
									self.netcontainer['input_weightmat']: weightmat,
									self.netcontainer['input_samplepts']: batchx2d,
									self.step : step}
				summary, _ = sess.run([total_summary, self.train_container['train_op']], feed_dict=feed_dict_summary)

				summary_writer.add_summary(summary, step)

				self.save(self.checkpoint_dir, step)
			else:
				sess.run(self.train_container['train_op'], feed_dict=feed_dict)

			duration = time.time() - start_time
			print('step {:d} \t loss = {:.5f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

	def test(self, img):
		feed = {self.netcontainer['input_img']: img}
		embedmap = self.sess.run(self.netcontainer['out_hypcol'], feed_dict=feed)

		return embedmap
		
		
