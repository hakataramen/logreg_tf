#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys



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

     
    
if __name__ == "__main__":

        if len(sys.argv) < 1:
             exit()

        train = sys.argv[1]
        #devel = sys.argv[2]
        #test = sys.argv[3]

        ft = open(train, "r")
        #fd = open(devel, "r")
        #fte = open(test, "r")

        train_data, maxidx = read_data(ft, -1)
        #l_f_list_maxidx = read_data(fd, Max_idx)
        #l_f_list_maxidx = read_data(fte, Max_idx)

        print(train_data)
        #print(l_f_list, maxidx)
        
