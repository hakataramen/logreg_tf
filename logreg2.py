#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys


if len(sys.argv) < 1:
    exit()

#def line_split_by_label_vecgtor(line):
    
    
def line_split(line):
    sents_split = line.strip().split(" ") #['ラベル', 'a:x', 'b:y', ...]
    label = sents_split[0] #ラベル
    #print(label)
    sentence = sents_split[1:] #ラベル以外の部分

    return(label, sentence)

def sentence_to_idx_freq(sentence):
   dic = [] 
   for idx_and_freq in sentence: #x:y
       split_idx_freq = idx_and_freq.split(":") #['x', 'y']
       idx = split_idx_freq[0]
       freq = split_idx_freq[1]

       for i in range(int(freq)):#idxの繰り返し数分だけリストにidxを追加
           #print(idx)
           dic.append(idx)

   return dic
       #dic.appeend(split_idx_freq[0])
        #split_idx_freq[1])

    #return split_idx_freq
    #print(split_idx_time)

    #return split_idx_time
#def read_instance(label, sentence):
    

#def lab_sents_to_read_instance(lab_sents):
#    for sents in lab_sents: #sents:文のラベル+素性ベクトル1つ分(x:y)
#        dic = []
#        sents_split = sents.strip().split(" ")
#        label = sents_split[0] #ラベル
#        #print(label)
#        sents_delete_label = sents_split[1:] #ラベル削除
#        for idx_and_time in sents_delete_label: #x:yを一つずつ取り出し
#            #print(idx_and_time)
#            split_idx_time = idx_and_time.split(":") #:で区切ってリスト表示
#            #print(split_idx_time)
#            
#            idx = split_idx_time[0]
#            time = split_idx_time[1] #頻度回数の取り出し
#            #print(idx)
#            
#            for i in range(int(time)):#idxの繰り返し数分だけリストにidxを追加
#                #print(idx)
#                dic.append(idx)
#           tuple = (label,dic)
#            
#    return tuple
    

txt = sys.argv[1]

f = open(txt, "r")


lab_sents = f.read().strip().split("\n")

label_list=[]
idx_freq_list=[]

for line in lab_sents: #1行分のラベルと素性
    
    label, sentence = line_split(line) #ラベルと文に分ける
    #print(label,sentence)
    label_list.append(label)
    idx_freq_list.append(sentence_to_idx_freq(sentence))
"""
    for idx_freq in sentence_to_idx_freq(sentence):
        #print(idx_freq)
        idx_freq_list.append(idx_freq)
        #print(label_list,idx_freq_list)
"""
x = [label_list,idx_freq_list]
#print(x)
    #idx_list = sentence_to_idx_times(sentence)
#print(lab_sents)

#a = lab_sents_to_read_instance(lab_sents)

#print(a)
