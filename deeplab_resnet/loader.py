import os
import time
import tensorflow as tf
import numpy as np

from .image_reader import read_an_image_from_disk, tf_wrap_get_patch

def load_single_image(args, input_size):
    loader_img = dict()
    tcurimgfname = tf.placeholder(tf.string, None, name='ph_fname_img')
    tcurlbmfname = tf.placeholder(tf.string, None, name='ph_fname_label')

    timg, tlabel = read_an_image_from_disk(tcurimgfname, tcurlbmfname, input_size, args.random_scale, args.random_mirror, 
                                    args.ignore_label, IMG_MEAN)
    
    loader_img['input_img_name'] = tcurimgfname
    loader_img['input_lbm_name'] = tcurlbmfname
    loader_img['output_img'] = timg
    loader_img['output_labelmap'] = tlabel
    return loader_img



def load_batch_samplepts():
    sampler = dict()
    tlabelholder = tf.placeholder(tf.float32, None, name='ph_labelmap')
    tbatchx2d, tweightmat = tf_wrap_get_patch(tlabelholder, NSAMPLEPTS, NINST)

    sampler['input_labelmap'] = tlabelholder
    sampler['out_batchx2d'] = tbatchx2d
    sampler['out_weightmat'] = tweightmat 
    return sampler