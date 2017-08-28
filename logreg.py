#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys


if len(sys.argv) < 1:
    exit()

#def line_split_by_label_vecgtor(line):
    
    

def lab_sents_to_read_instance(lab_sents):
    for sents in lab_sents: #sents:文のラベル+素性ベクトル1つ分(x:y)
        dic = []
        sents_split = sents.strip().split(" ")
        label = sents_split[0] #ラベル
        #print(label)
        sents_delete_label = sents_split[1:] #ラベル削除
        for idx_and_time in sents_delete_label: #x:yを一つずつ取り出し
            #print(idx_and_time)
            split_idx_time = idx_and_time.split(":") #:で区切ってリスト表示
            #print(split_idx_time)
            
            idx = split_idx_time[0]
            time = split_idx_time[1] #頻度回数の取り出し
            #print(idx)
            
            for i in range(int(time)):#idxの繰り返し数分だけリストにidxを追加
                #print(idx)
                dic.append(idx)
            tuple = (label,dic)
            
    return tuple
    

txt = sys.argv[1]

f = open(txt, "r")

lab_sents = f.read().strip().split("\n")
#print(lab_sents)

a = lab_sents_to_read_instance(lab_sents)

print(a)

