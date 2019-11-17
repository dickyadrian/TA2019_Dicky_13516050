"""Train a voxel flow model on ucf101 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from CyclicGen_model_large import Voxel_flow_model
import scipy as sp
import imageio
import cv2
from vgg16 import Vgg16

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', None,
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_string('first', '',
                           """first image """)
tf.app.flags.DEFINE_string('second', '',
                           """second image """)
tf.app.flags.DEFINE_string('out', '',
                           """output image """)


def imread(filename):
    """Read image from file.
    Args:
    filename: .
    Returns:
    im_array: .
    """
    im = imageio.imread(filename, pilmode='RGB')
    return im / 127.5 - 1.0


def test(first, second, out):

    data_frame1 = np.expand_dims(imread(first), 0)
    data_frame3 = np.expand_dims(imread(second), 0)

    H = data_frame1.shape[1]
    W = data_frame1.shape[2]

    adatptive_H = int(np.ceil(H / 32.0) * 32.0)
    adatptive_W = int(np.ceil(W / 32.0) * 32.0)

    pad_up = int(np.ceil((adatptive_H - H) / 2.0))
    pad_bot = int(np.floor((adatptive_H - H) / 2.0))
    pad_left = int(np.ceil((adatptive_W - W) / 2.0))
    pad_right = int(np.floor((adatptive_W - W) / 2.0))

    print(str(H) + ', ' + str(W))
    print(str(adatptive_H) + ', ' + str(adatptive_W))

    """Perform test on a trained model."""
    with tf.Graph().as_default():
        # Create input and target placeholder.
        input_placeholder = tf.placeholder(tf.float32, shape=(None, H, W, 6))

        input_pad = tf.pad(input_placeholder, [[0, 0], [pad_up, pad_bot], [pad_left, pad_right], [0, 0]], 'SYMMETRIC')

        edge_vgg_1 = Vgg16(input_pad[:, :, :, :3], reuse=None)
        edge_vgg_3 = Vgg16(input_pad[:, :, :, 3:6], reuse=True)

        edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
        edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)

        edge_1 = tf.reshape(edge_1, [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2], 1])
        edge_3 = tf.reshape(edge_3, [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2], 1])

        with tf.variable_scope("Cycle_DVF"):
            # Prepare model.
            model = Voxel_flow_model(is_train=False)
            prediction = model.inference(tf.concat([input_pad, edge_1, edge_3], 3))[0]

        # Create a saver and load.
        sess = tf.Session()

        # Restore checkpoint from file.
        if FLAGS.pretrained_model_checkpoint_path:
            restorer = tf.train.Saver()
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        feed_dict = {input_placeholder: np.concatenate((data_frame1, data_frame3), 3)}
        # Run single step update.
        prediction_np = sess.run(prediction, feed_dict=feed_dict)

        output = prediction_np[-1, pad_up:adatptive_H - pad_bot, pad_left:adatptive_W - pad_right, :]
        output = np.round(((output + 1.0) * 255.0 / 2.0)).astype(np.uint8)
        output = np.dstack((output[:, :, 2], output[:, :, 1], output[:, :, 0]))
        cv2.imwrite(out, output)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    DATA_DIR = 'VideoData'
    RESULT_DIR = 'interpolated-cyclic'

    result_path = RESULT_DIR+datetime.today().strftime('%Y-%m-%d')
    os.mkdir(result_path)

    for dire in os.listdir(DATA_DIR):
        print(dire)
        images = sorted(os.listdir(os.path.join(DATA_DIR, dire)))
        first = os.path.join(DATA_DIR, dire, images[0])
        second = os.path.join(DATA_DIR, dire, images[1])
        first_number = int(images[0].split('frame')[-1].replace('.jpg', ''))
        out = os.path.join(result_path, 'frame'+str(first_number+1)+'.jpg')
        test(first, second, out)
