#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model import *
from utils import *
import tensorflow as tf
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[2]:


tf.app.flags.DEFINE_string('f', '', 'kernel')

flags = tf.app.flags

flags.DEFINE_string('ASC', 'qb', 'restaurant or laptop or qb')
flags.DEFINE_string('DSC', 'yelp', '{yelp, twitter} for restaurant & {amazon, twitter} for laptop')
flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
#flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_integer('n_iter', 25, 'training iteration')
#flags.DEFINE_integer('n_iter', 2, 'training iteration')
# We slightly modify the training procedure. Feeding all DSC data in ONE epoch can get better results.
flags.DEFINE_float('gamma', 0.1, '{0.1, 0.1, 0.9, 0.2} for {res+yelp, res+twitter, laptop+amazon, laptop+twitter')
flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
flags.DEFINE_integer('position_dim', 100, 'dimension of position embedding')
#flags.DEFINE_integer('max_sentence_len', 85, 'max number of tokens per sentence')
flags.DEFINE_integer('max_sentence_len', 210, 'max number of tokens per sentence')
flags.DEFINE_integer('max_target_len', 25, 'max number of tokens per target')
flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob1')
flags.DEFINE_float('keep_prob2', 1.0, 'dropout keep prob2')
# Parameters for capsule layers.
flags.DEFINE_integer('filter_size', 3, 'filter_size')
flags.DEFINE_integer('sc_num', 16, 'sc_num')
flags.DEFINE_integer('sc_dim', 16, 'sc_dim')
flags.DEFINE_integer('cc_num',  3, 'cc_num')
flags.DEFINE_integer('cc_dim', 24, 'cc_dim')
flags.DEFINE_integer('iter_routing', 3, 'routing iteration')
flags.DEFINE_bool("reuse_embedding", True, "reuse word embedding & id, True or False")
FLAGS = flags.FLAGS


# In[3]:


def main(_):
    start_time = time.time()
    info = ''
    index = 0
    
    for name, value in FLAGS.__flags.items():
        value = value.value
        if index < 19:
            info += '{}:{}  '.format(name, value)
        if index in [5, 11]:
            info += '\n'
        index += 1
    print('\n{:-^80}'.format('Parameters'))
    print(info + '\n')
    
    print('---------')
    print(FLAGS.ASC)
    
    data_path = 'data/{}/'.format(FLAGS.ASC)
    
    if not FLAGS.reuse_embedding :
        print('Initialize Word Dictionary & Embedding')
        word_dict = data_init(data_path, FLAGS.DSC)
        w2v = init_word_embeddings(data_path, word_dict, FLAGS.DSC)
    else:
        print('Reuse Word Dictionary & Embedding')
        with open(data_path + FLAGS.DSC + '_word2id.txt', 'r', encoding='utf-8') as f:
            word_dict = eval(f.read())
        w2v = np.load(data_path + FLAGS.DSC + '_word_embedding.npy')
    
    #print(w2v)
    #print(word_dict)
    
    for i in range(15):
        model = MODEL(FLAGS, w2v, word_dict, data_path)
        model.run()
    
    end_time = time.time()
    print('Running Time: {:.0f}m {:.0f}s'.format((end_time-start_time) // 60, (end_time-start_time) % 60))


# In[ ]:


if __name__ == '__main__':
    tf.app.run()


# In[ ]:




