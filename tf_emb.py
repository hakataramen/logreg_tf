#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import numpy as np


def read_data(f, feature_size):
     l_list = []
     if_list = []

     Max_idx = 0
     

     lab_sents = f.read().strip().split("\n") #1文ごとに区切り [文1,文2,......],
     for line in lab_sents: #line:1行分のラベルと素性: ラベル,素性
          lab_idx_freq_list = read_instance(line) # lab_ = [[lab],[idx_freq]]
          l_list.extend(lab_idx_freq_list[0])
          #print(lab_idx_freq_list[1])

          if feature_size == -1:
               Max_idx = lab_idx_freq_list[1][-1] #インデックスの最大番号=Maxidx

               
          one_if_list = [] 
          for idx in lab_idx_freq_list[1]: #1文分のidx_freqリストからひとつづつidxを取り出す
               #print (idx_list)
               if (int(Max_idx) >= int(idx)) or (0 > feature_size): #Maxidxを超える素性は無視、0以下はすべて入れる
                    one_if_list.append(idx) #1文分のidx_freqをリストに追加(次の文を読み込む時点で初期化される)
                    #print(one_if_list)
          if_list.append(one_if_list) #1文分のidx_freqのリストをリスト毎if_listに追加
          #print(one_if_list)
     #print(if_list)
     y = ([l_list,if_list], int(Max_idx)+1)
     #print(y)
     return(y)



def read_instance(line):
     lab_list =[]
     idx_freq_list = []
     #max_idx = 0
     
     sents_split = line.strip().split(" ") #['ラベル', 'a:x', 'b:y', ...]
     label = sents_split[0] #ラベル
     sentence = sents_split[1:] #ラベル以外の部分

     lab_list.append(label)
     
     for idx_and_freq in sentence: #x:y
          split_idx_freq = idx_and_freq.split(":") #['x', 'y']
          idx = split_idx_freq[0]
          freq = split_idx_freq[1]

         # if int(idx) >int(feature_size):
              # break
          
         # max_idx = idx
          #     #print(max_idx)

          

          for i in range(int(freq)):#idxの繰り返し数分だけリストにidxを追加
               #print(idx)
               idx_freq_list.append(idx)

     x = (lab_list,idx_freq_list)
     #print(x)

     return x

def make_graph(vocab_size):
     #モデルパラメータ
     x = tf.placeholder(tf.int32, [None]) # shape=[None], train.txtのidxを受け取る(1文分?)
     one_hot = tf.one_hot(indices=x, depth=vocab_size) # shape=[None, vocab_size]　1分の語彙数分、idxの位置に1が立つ
     embedding = tf.Variable(tf.random_uniform(shape=[vocab_size,50], minval=-1.0, maxval=1.0, dtype=tf.float32)) # shape=[vocab_size, 50] 変数、全文の?語彙数×50次元のランダムの行列を用意(更新される)
     word_vecs = tf.matmul(one_hot, embedding) # shape=[None, 50]各単語に対応する成分をone_hotを掛けることでembeddingから取り出す(1文分?)
     # あるいは
     # word_vecs = tf.nn.embedding_lookup(embedding, x) # one_hotを経由しなくてよくなる
     sentence_vec = tf.reduce_mean(word_vecs, reduction_indices=0) # shape=[50]文を示すベクトル
     h = tf.reshape(sentence_vec, [1, 50]) # shape=[1,50] sentence_vecを計算できる形に
     w = tf.Variable(tf.random_uniform(shape=[50, 2], minval=-1.0, maxval=1.0, dtype=tf.float32))
     b = tf.Variable(tf.random_uniform(shape=[2], minval=-1.0, maxval=1.0, dtype=tf.float32))
     logit = tf.matmul(h, w) + b # shape=[1,2]
     logit = tf.reshape(logit, [2]) # shape=[2]
     y = tf.nn.softmax(logit) # shape=[2]
     label = tf.placeholder(tf.float32, [2]) # shape=[2]
     loss = -tf.reduce_sum(label*tf.log(y)) # shape=[]
     # あるいは
     #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(label, logit))

     accuracy, update_accuracy = tf.metrics.accuracy(tf.argmax(label, axis=0), tf.argmax(y, axis=0))
     def eval_accuracy(sess, corpus):
          sess.run(tf.local_variables_initializer())
          counter = 0
          for l,i in zip(corpus[0], corpus[1]):
               counter += 1
               if counter % 100 == 0: print(counter)
               sess.run(update_accuracy, feed_dict={x:i, label:to_onehot_label(l)})
          return sess.run(accuracy)

     return x, label, y, loss, eval_accuracy

     

def tf_embeddings(train_data, vocab_size):
     #sentence_list = train_data[1][0:]
     for sentence_list in train_data[1][0:]:
          print(sentence_list)
          #print(vector)
     return(sentence_list)
                                        

def to_onehot_label(label_string):
     if label_string == "1":
          return [1.0, 0.0]
     elif label_string == "-1":
          return [0.0, 1.0]

    
if __name__ == "__main__":

        tf.set_random_seed(0)

        if len(sys.argv) < 1:
             exit()

        train = sys.argv[1]
        #devel = sys.argv[2]
        #test = sys.argv[3]

        ft = open(train, "r")
        #fd = open(devel, "r")
        #fte = open(test, "r")

        train_data, vocab_size = read_data(ft, -1)
        #l_f_list_maxidx = read_data(fd, Max_idx)
        #l_f_list_maxidx = read_data(fte, Max_idx)

        #print(train_data)
        #print(l_f_list, maxidx)

        x, label, y, loss, eval_accuracy = make_graph(vocab_size)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        print(eval_accuracy(sess, train_data))
        train_data_size = len(train_data[0])
        for target in range(train_data_size):
             input_sentence = train_data[1][target]
             gold = train_data[0][target]
             onehot_gold = to_onehot_label(gold)
             sess.run(train_op, feed_dict={x:input_sentence, label:onehot_gold})

             if target % 100 == 0:
                  print(sess.run(loss, feed_dict={x:input_sentence, label:onehot_gold}))
        print(eval_accuracy(sess, train_data))


