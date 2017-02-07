import tensorflow as tf
import os
import random
import scipy
import numpy as np

#files
def load_folder_list(path=""):
    """Return a folder list in a folder by given a folder path.

    Parameters
    ----------
    path : a string or None
        A folder path.
    """
    return [os.path.join(path,o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]

#utils
def print_dict(dictionary={}):
    """Print all keys and items in a dictionary.
    """
    for key, value in dictionary.iteritems():
        print("key: %s  value: %s" % (str(key), str(value)))

#prepro ?
def get_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.

    Examples
    ---------
    >>> r = get_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min,max) for p in range(0,number)]



## Save images
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def save_images(images, size, image_path):
    return imsave(images, size, image_path)




# Cost
def cosine_similarity(v1, v2):
    """Cosine similarity [-1, 1], `wiki <https://en.wikipedia.org/wiki/Cosine_similarity>`_.

    Parameters
    -----------
    v1, v2 : tensor of [batch_size, n_feature], with the same number of features.

    Returns
    ________
    a tensor of [batch_size, ]
    """
    return tf.reduce_sum(tf.mul(v1, v2), reduction_indices=1) / (tf.sqrt(tf.reduce_sum(tf.mul(v1, v1), reduction_indices=1)) * tf.sqrt(tf.reduce_sum(tf.mul(v2, v2), reduction_indices=1)))




# prepro
# def process_sequences(sequences, end_id=0, pad_val=0, is_shorten=True, remain_end_id=False):
#     """Set all tokens(ids) after END token to the padding value, and then shorten (option) it to the maximum sequence length in this batch.
#
#     Parameters
#     -----------
#     sequences : numpy array or list of list with token IDs.
#         e.g. [[4,3,5,3,2,2,2,2], [5,3,9,4,9,2,2,3]]
#     end_id : int, the special token for END.
#     pad_val : int, replace the end_id and the ids after end_id to this value.
#     is_shorten : boolean, default True.
#         Shorten the sequences.
#     remain_end_id : boolean, default False.
#         Keep an end_id in the end.
#
#     Examples
#     ---------
#     >>> sentences_ids = [[4, 3, 5, 3, 2, 2, 2, 2],  <-- end_id is 2
#     ...                  [5, 3, 9, 4, 9, 2, 2, 3]]  <-- end_id is 2
#     >>> sentences_ids = precess_sequences(sentences_ids, end_id=vocab.end_id, pad_val=0, is_shorten=True)
#     ... [[4, 3, 5, 3, 0], [5, 3, 9, 4, 9]]
#     """
#     max_length = 0
#     for i_s, seq in enumerate(sequences):
#         is_end = False
#         for i_w, n in enumerate(seq):
#             if n == end_id and is_end == False: # 1st time to see end_id
#                 is_end = True
#                 if max_length < i_w:
#                     max_length = i_w
#                 if remain_end_id is False:
#                     seq[i_w] = pad_val      # set end_id to pad_val
#             elif is_end == True:
#                 seq[i_w] = pad_val
#
#     if remain_end_id is True:
#         max_length += 1
#     if is_shorten:
#         for i, seq in enumerate(sequences):
#             sequences[i] = seq[:max_length]
#     return sequences
#
# def sequences_add_start_id(sequences, start_id=0, remove_last=False):
#     """Add special start token(id) in the beginning of each sequence.
#
#     Examples
#     ---------
#     >>> sentences_ids = [[4,3,5,3,2,2,2,2], [5,3,9,4,9,2,2,3]]
#     >>> sentences_ids = sequences_add_start_id(sentences_ids, start_id=2)
#     ... [[2, 4, 3, 5, 3, 2, 2, 2, 2], [2, 5, 3, 9, 4, 9, 2, 2, 3]]
#     >>> sentences_ids = sequences_add_start_id(sentences_ids, start_id=2, remove_last=True)
#     ... [[2, 4, 3, 5, 3, 2, 2, 2], [2, 5, 3, 9, 4, 9, 2, 2]]
#
#     - For Seq2seq
#     >>> input = [a, b, c]
#     >>> target = [x, y, z]
#     >>> decode_seq = [start_id, a, b] <-- sequences_add_start_id(input, start_id, True)
#     """
#     sequences_out = [[] for _ in range(len(sequences))]#[[]] * len(sequences)
#     for i in range(len(sequences)):
#         if remove_last:
#             sequences_out[i] = [start_id] + sequences[i][:-1]
#         else:
#             sequences_out[i] = [start_id] + sequences[i]
#     return sequences_out
#
# def sequences_get_mask(sequences, pad_val=0):
#     """Return mask for sequences.
#
#     Examples
#     ---------
#     >>> sentences_ids = [[4, 0, 5, 3, 0, 0],
#     ...                  [5, 3, 9, 4, 9, 0]]
#     >>> mask = sequences_get_mask(sentences_ids, pad_val=0)
#     ... [[1 1 1 1 0 0]
#     ...  [1 1 1 1 1 0]]
#     """
#     mask = np.ones_like(sequences)
#     for i, seq in enumerate(sequences):
#         for i_w in reversed(range(len(seq))):
#             if seq[i_w] == pad_val:
#                 mask[i, i_w] = 0
#             else:
#                 break   # <-- exit the for loop, prepcess next sequence
#     return mask

if __name__ == "__main__":
    sentences_ids = [[4,3,5,3,2,2,2,2], [5,3,9,4,9,2,2,3]]
    sentences_ids = precess_sequences(sentences_ids, end_id=2, pad_val=0, is_shorten=True)
    print(sentences_ids)#[[4, 3, 5, 3, 0], [5, 3, 9, 4, 9]]

    sentences_ids = [[4,3,5,3,2,2,2,2], [5,3,9,4,9,2,2,3]]
    sentences_ids = sequences_add_start_id(sentences_ids, start_id=2, remove_last=True)
    print(sentences_ids) #[[2, 4, 3, 5, 3, 2, 2, 2], [2, 5, 3, 9, 4, 9, 2, 2]]

    sentences_ids = [[4, 0, 5, 3, 0, 0], [5, 3, 9, 4, 9, 0]]
    mask = sequences_get_mask(sentences_ids, pad_val=0)
    print(mask)#[[1 1 1 1 0 0], [1 1 1 1 1 0]]


#
