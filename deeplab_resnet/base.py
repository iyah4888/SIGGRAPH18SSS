import os
from glob import glob
import tensorflow as tf

class Model(object):
  """Abstract object representing an Reader model."""
  def __init__(self):
    self.vocab = None
    self.data = None

  def get_model_dir(self):
    model_dir = self.dataset
    for attr in self._attrs:
      if hasattr(self, attr):
        model_dir += "_%s:%s" % (attr, getattr(self, attr))
    return model_dir

  def save(self, checkpoint_dir, global_step=None):
    self.saver = tf.train.Saver(max_to_keep=10)

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__ or "Reader"
    model_dir = self.get_model_dir()

    checkpoint_dir = os.path.join(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess, 
        os.path.join(checkpoint_dir, model_name), global_step=global_step)

  def load(self, checkpoint_dir):
    self.saver = tf.train.Saver(max_to_keep=10)

    print(" [*] Loading checkpoints...")
    model_dir = self.get_model_dir()
    checkpoint_dir = os.path.join(checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      # self.saver.restore(self.sess, '/home/iyah4888/Drive/whitehole/Code/yagiz_semanticmatting/DeepLabbase/snapshotsv2/model.ckpt-20000')
      
      print(" [*] Load SUCCESS")
      return True
    else:
      print(" [!] Load failed...")
      return False
