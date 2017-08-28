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

def make_one_hot(corpus):
        #train_data前処理
        box = []
        for l, i in zip(corpus[0], corpus[1]):
             if l == "1":
                  box.append([0.0, 1.0])
             else:
                  box.append([1.0, 0.0])

        box2 = []
        box2.append(box)
        box2.append(corpus[1])

        return box2
    
if __name__ == "__main__":

        tf.set_random_seed(0)

        if len(sys.argv) < 1:
             exit()

        train = sys.argv[1]
        devel = sys.argv[2]
        #test = sys.argv[3]

        ft = open(train, "r")
        fd = open(devel, "r")
        #fte = open(test, "r")

        train_data, vocab_size = read_data(ft, -1)
        devel_data, _  = read_data(fd, vocab_size)
        #l_f_list_maxidx = read_data(fte, Max_idx)

        #print(train_data)
        #print(l_f_list, maxidx)

        #placeholderを定義
        x = tf.placeholder(tf.int32,[None]) #train_dataの一文のidx×回数分が入る、one_hot作成に使う
        label = tf.placeholder(tf.float32, [2]) #train_dataのラベルを入れる,[1]でOK?
        
        #labelを元にone_hotの作成
        #one_hot_label = tf.one_hot(indices=label, depth =2)

        #variablesの定義→w,b,embedding
        w = tf.Variable(tf.random_uniform(shape=[50,2], minval=-1.0, maxval=1.0, dtype=tf.float32))#重み
        b = tf.Variable(tf.random_uniform(shape=[2], minval=-1.0, maxval=1.0, dtype=tf.float32))#バイアス

        #embeddingsを作り出すための箱（vocab_sizeに対応し、one_hotを用いてここから抜き出しemneddingsを作成)
        embedding = tf.Variable(tf.random_uniform(shape=[vocab_size,50], minval=-1.0, maxval=1.0, dtype=tf.float32))

        embeddings = tf.nn.embedding_lookup(embedding, x)
        sentence_vec = tf.reduce_mean(embeddings, reduction_indices=0)

        h = tf.reshape(sentence_vec, [1,50])

        #ロジット
        logit = tf.matmul(h,w)+b
        logit = tf.reshape(logit, [2])

        #出力y
        y = tf.nn.softmax(logit)

        #誤差関数
        loss = -tf.reduce_sum(label*tf.log(y))

        #---------------------------------------

        #学習
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss)


        #評価
        accuracy, accuracy_update_op = tf.metrics.accuracy(tf.argmax(label, axis=0), tf.argmax(y, axis=0))
        precision, precision_update_op = tf.metrics.precision(tf.argmax(label, axis=0), tf.argmax(y, axis=0))
        recall, recall_update_op = tf.metrics.recall(tf.argmax(label, axis=0), tf.argmax(y, axis=0))#引数まだ決めてない

        
        sess = tf.Session()
        train_init_op = tf.global_variables_initializer()
        sess.run(train_init_op)
        eval_init_op = tf.local_variables_initializer()
        sess.run(eval_init_op)
        

        #学習開始---------------------------------------------------------

        tr_deform = make_one_hot(train_data)
        
        print("training...")
        counter = 0
        for l,i in zip(tr_deform[0], tr_deform[1]):
             counter +=1
             feed = {x:i, label:l}
             _,ls,ac_up_op,pre_up_op,rec_up_op = sess.run([train_step, loss, accuracy_update_op, precision_update_op, recall_update_op], feed_dict=feed)
             if counter % 200 ==0:
                  print("loss=",ls)
        acc = sess.run(accuracy)
        pre = sess.run(precision)
        rec = sess.run(recall)
        
        #print("accuracy=", sess.run(accuracy),"precision=", sess.run(precision), "recall=", sess.run(recall))
        #print("F=", (2*rec*pre)/(rec+pre))
        #print("training finished")

        #学習終了----------------------------------------------------------

        
        #予測開始----------------------------------------------------------
        
        print("predict start")
        dev_deform = make_one_hot(devel_data)
        print("predicting...")
        counter = 0
        for l,i in zip(dev_deform[0], dev_deform[1]):
             counter +=1
             feed = {x:i, label:l}
             ls,ac_up_op,pre_up_op,rec_up_op = sess.run([loss, accuracy_update_op, precision_update_op, recall_update_op], feed_dict=feed)
             if counter % 200 ==0:
                  print("loss=",ls)
        acc = sess.run(accuracy)
        pre = sess.run(precision)
        rec = sess.run(recall)
        
        print("accuracy=", sess.run(accuracy),"precision=", sess.run(precision), "recall=", sess.run(recall))
        print("F=", (2*rec*pre)/(rec+pre))
        print("predicting finished")
        
