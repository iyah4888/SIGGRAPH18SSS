'''
The code is borrowed from 
https://github.com/DrSleep/tensorflow-deeplab-resnet#using-your-dataset
Modified by Tae-Hyun Oh
'''

import os

import numpy as np
import tensorflow as tf

def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
   
    return img, label

def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label]) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop  

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(os.path.join(data_dir, image))
        masks.append(os.path.join(data_dir, mask))
    return images, masks

def read_data_list(data_dir, data_list, ext):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    dataflist = []
    
    for line in f:
        try:
            dataname = line.strip("\n")
        except ValueError: # Adhoc for test.
            dataname = line.strip("\n")
        dataflist.append(os.path.join(data_dir, dataname+ext))
    return dataflist


def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, ignore_label, img_mean): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label)

        # Randomly crops the images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    return img, label

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, 
                 random_scale, random_mirror, ignore_label, img_mean, coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        
        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=input_size is not None) # not shuffling if it is val
        self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror, ignore_label, img_mean) 

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  num_elements)
        return image_batch, label_batch





f_rperm = np.random.permutation

def read_inst_pts_data(matfilename):
  labelchunk = sio.loadmat(matfilename)['GTinst'] 
  ## parse the data structure
  labelst = dict()
  labelst['labelmap'] = labelchunk['Segmentation'][0,0]
  labelst['instID'] = np.unique(labelst['labelmap'])
  ## extract coordinates
  labelst['x2d'] = list()
  for i in range(len(labelst['instID'])):   # loop over #instances
    curx2d = np.where(np.equal(labelst['labelmap'], labelst['instID'][i]))
    labelst['x2d'].append(curx2d)
    
  return labelst




def parse_inst_pts_data(labelmap):
  ## parse the data structure
  labelst = dict()
  labelst['labelmap'] = np.squeeze(labelmap)
  labelst['instID'] = np.unique(labelst['labelmap'])
  ## extract coordinates
  parsed2d = list()
  for i in range(len(labelst['instID'])):   # loop over #instances
    curx2d = np.where(np.equal(labelst['labelmap'], labelst['instID'][i]))
    parsed2d.append(curx2d)
    
  return parsed2d


def get_indicator_mat(batch_size, ninst):
  indicator = np.zeros((ninst*batch_size,ninst), dtype=int)

  for i in range(ninst):
    indicator[(batch_size*i):(batch_size*(i+1)),i] = 1

  tmpcross = np.dot(indicator, indicator.T)

  indicator_mat = np.zeros_like(tmpcross) 
  indicator_mat[tmpcross.astype(bool)] = 1
  indicator_mat[np.logical_not(tmpcross)] = -1
  indicator_mat = indicator_mat.astype(np.float32)
  return indicator_mat

def _get_weight_mat(batch_size, ninst):
  weight_mat = np.ones((ninst*batch_size,ninst*batch_size), dtype=np.float32)
  # /np.float32(ninst*ninst*batch_size*batch_size)
  return weight_mat

def get_batch_1chunk(lst_x2d, batch_size, ninst):
  
  no_total_inst = len(lst_x2d);
  
  assert(no_total_inst >= ninst), 'No. instances in the image is less than the required number'

  curtarget = f_rperm(no_total_inst)[0:ninst] # select #ninst instances

  # sample batch no. pixels
  points = list()
  
  for i in range(ninst):
    candidate_inst_x2ds = lst_x2d[curtarget[i]] # list in 2-dim tuple
    sampleidx = np.random.randint(candidate_inst_x2ds[0].shape[0], size=batch_size)
    # cur_sampled2d: batch_size X 2
    cur_sampled2d = np.stack([candidate_inst_x2ds[0][sampleidx], candidate_inst_x2ds[1][sampleidx]], -1)
    points.append(cur_sampled2d)
  # points: ninst X batch_size X 2

  weightmat = _get_weight_mat(batch_size, ninst)

  batchx2d = np.vstack(points[:]).astype(np.float32)
  
  # batchx2d: ninst*batch_size X 2
  return batchx2d, weightmat


def get_batch(labelmap, batch_size, ninst):
  if len(labelmap.shape) == 3:
    labelmap = np.squeeze(labelmap)
  parsed2d = parse_inst_pts_data(labelmap)  # list
  batchx2d, weightmat = get_batch_1chunk(parsed2d, batch_size, ninst)
  return batchx2d, weightmat

def tf_wrap_get_patch(labelmap, batch_size, ninst):
  batchx2d, weightmat = tf.py_func(get_batch, [labelmap, batch_size, ninst], [tf.float32, tf.float32])
  return batchx2d, weightmat

def read_an_image_from_disk(t_imgfname, t_labelfname, input_size, random_scale, random_mirror, ignore_label, img_mean): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """


    img_contents = tf.read_file(t_imgfname)
    lbm_contents = tf.read_file(t_labelfname)
    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    label = tf.image.decode_png(lbm_contents, channels=1)

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label)

        # Randomly crops the images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)


    return img, label


