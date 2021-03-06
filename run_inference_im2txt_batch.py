#! /usr/bin/python
# -*- coding: utf8 -*-




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np
import scipy
import time
from PIL import Image

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
from model_im2txt import * #<--- Modify from Image-Captioning repo
from utils import *

# images_train_dir = '/home/haodong/Workspace/image_captioning/data/mscoco/raw-data/train2014/'
images_train_dir = './data/mscoco/train2014/'
images_train_list = tl.files.load_file_list(path=images_train_dir, regx='\\.jpg', printable=False)
# print(train_img_list)
images_train_list = [images_train_dir + s for s in images_train_list]
images_train_list = np.asarray(images_train_list)
# print(images_train_list)
n_images_train = len(images_train_list)
# exit()

# DIR = "/home/haodong/Workspace/image_captioning"

# Directory containing model checkpoints.
# CHECKPOINT_DIR = DIR + "/model/train"
# CHECKPOINT_DIR = "./data/mscoco/model.ckpt-2000000"#.data-00000-of-00001"
CHECKPOINT_DIR = '/home/lei/Documents/Workspace/models/research/im2txt/im2txt/model/train/model.ckpt-1000000'
# Vocabulary file generated by the preprocessing script.
# VOCAB_FILE = DIR + "/data/mscoco/word_counts.txt"
# VOCAB_FILE = "./data/mscoco/word_counts.txt"
VOCAB_FILE = '/home/lei/Documents/Workspace/models/research/im2txt/im2txt/data/mscoco/word_counts.txt'


tf.logging.set_verbosity(tf.logging.INFO) # Enable tf.logging



def prepro_img(x, mode='for_image_caption'):
    if mode=='for_image_caption':
        ## in model.py , image captioning preprocess the images as below.
        # image = tf.image.resize_images(image,
        #                                size=[resize_height, resize_width],
        #                                method=tf.image.ResizeMethod.BILINEAR)
        # image = tf.random_crop(image, [image_height, image_width, 3])
        # image = tf.sub(image, 0.5)
        # image = tf.mul(image, 2.0)
        x = imresize(x, size=[resize_height, resize_width], interp='bilinear', mode=None)
        x = crop(x, wrg=image_width, hrg=image_height, is_random=False)
        if x.shape[0] != image_width:
            x = imresize(x, size=[image_width, image_width], interp='bilinear', mode=None)
        x = x / (255. / 2.)
        x = x - 1.
    if mode=='read_image':
        # x : file name
        return scipy.misc.imread(x, mode='RGB')
    return x

def main(_):
    # Model checkpoint file or directory containing a model checkpoint file.
    checkpoint_path = CHECKPOINT_DIR
    # Text file containing the vocabulary.
    vocab_file = VOCAB_FILE
    # File pattern or comma-separated list of file patterns of image files.

    mode = 'inference'
    max_caption_length = 20
    top_k = 3
    # n_captions = 1
    print("n_images_train: %d" % n_images_train)

    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        # images, input_seqs, target_seqs, input_mask, input_feed = Build_Inputs(mode, input_file_pattern=None)
        images = tf.placeholder('float32', [batch_size, image_height, image_width, 3])
        input_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='input_seqs')
        net_image_embeddings = Build_Image_Embeddings(mode, images, train_inception=False)
        net_seq_embeddings = Build_Seq_Embeddings(input_seqs)
        softmax, net_img_rnn, net_seq_rnn, state_feed = Build_Model(mode, net_image_embeddings, net_seq_embeddings, target_seqs=None, input_mask=None)

        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if not checkpoint_path:
                raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

        saver = tf.train.Saver()
        def _restore_fn(sess):
            tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
            saver.restore(sess, checkpoint_path)
            # tf.train.NewCheckpointReader(checkpoint_path)
            tf.logging.info("Successfully loaded checkpoint: %s",
                          os.path.basename(checkpoint_path))

        restore_fn = _restore_fn
    g.finalize()

    # Create the vocabulary.
    vocab = tl.nlp.Vocabulary(vocab_file)

    # Generate captions
    with tf.Session(graph=g) as sess:
      # Load the model from checkpoint.
        restore_fn(sess)

        idexs = get_random_int(min=0, max=n_images_train-1, number=batch_size)
        b_image_file_name = images_train_list[idexs]

        start_time = time.time()
        ## read a batch of images for folder
        b_images = threading_data(b_image_file_name, prepro_img, mode='read_image')
        ## you may want to view the image
        # for i, img in enumerate(b_images):
        #     scipy.misc.imsave('image_%d.png' % i, img)
        # print("took %s seconds" % (time.time()-start_time))
        ## preprocess a batch of image
        b_images = threading_data(b_images, prepro_img, mode='for_image_caption')

        ## generate captions for a batch of images
        init_state = sess.run(net_img_rnn.final_state, feed_dict={images: b_images})
        state = np.hstack((init_state.c, init_state.h)) # (1, 1024)
        ids = [[vocab.start_id]] * batch_size
        print("took %s seconds" % (time.time()-start_time))
        sentences = [[] for _ in range(batch_size)]
        sentences_ids = [[] for _ in range(batch_size)]
        for _ in range(max_caption_length - 1):
            softmax_output, state = sess.run([softmax, net_seq_rnn.final_state],
                                    feed_dict={ input_seqs : ids,
                                                state_feed : state,
                                                })
            state = np.hstack((state.c, state.h))

            ids = []
            for i in range(batch_size):
                a_id = tl.nlp.sample_top(softmax_output[i], top_k=top_k)
                word = vocab.id_to_word(a_id)
                sentences[i].append(word)
                sentences_ids[i].append(a_id)
                ids = ids + [[a_id]]

        # before cleaning data
        for i, sentence in enumerate(sentences):
            print("%d : %s" % (i, " ".join(sentence)))

        for i, sentence_id in enumerate(sentences_ids):
            print("%d : %s" % (i, sentence_id))

        print("took %s seconds" % (time.time()-start_time))

        # after cleaning data
        sentences_ids = process_sequences(sentences_ids, end_id=vocab.end_id, pad_val=0)
        for i, sentence_id in enumerate(sentences_ids):
            print("%d : %s" % (i, sentence_id))

        for i, sentence_id in enumerate(sentences_ids):
            print("%d : %s" % (i, [vocab.id_to_word(id) for id in sentence_id]) )

        print("start_id:%d end_id:%d" % (vocab.start_id, vocab.end_id))

if __name__ == "__main__":
  tf.app.run()
