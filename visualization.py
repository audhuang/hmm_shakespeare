from __future__ import print_function
from __future__ import division
from scipy.integrate import quad
import numpy as np
import random 
import codecs
import string
import re
import cPickle as cp
import sys
import os
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import cmudict

def pos_analysis(O, pos_dic, pos_to_words, index_dic): 
	parts_of_speech = []

	for key in pos_to_words: 
		parts_of_speech.append(key)

	dic = np.zeros([len(O), len(parts_of_speech)])
	

	for i in range(O.shape[0]): 
		for j in range(O.shape[1]): 
			pos_list = pos_dic[index_dic[j]]
			pos = max(set(pos_list), key=pos_list.count)
			index = parts_of_speech.index(pos)
			dic[i][index] += O[i][j]

	print(dic)

	for i in dic: 
		row = i.tolist()
		max_pos = row.index(max(row))
		print(max_pos, parts_of_speech[max_pos])
	return dic 


def word_category(O, index_dic): 
	indices = []
	for i in O: 
		row_words = []
		row = i
		values = row.argsort()[-10:][::-1]
		
		for j in values: 
			row_words.append(index_dic[j])

		indices.append(row_words)

	print(indices)
	return indices

def syllables(O, index_dic, syl_dic): 
	ave_syl = np.zeros([len(O)])
	for i in range(O.shape[0]): 
		total_prob = 0
		for j in range(O.shape[1]): 
			if O[i][j] != 0: 
				num_syl = sum(syl_dic[index_dic[j]]) / len(syl_dic[index_dic[j]])
				ave_syl[i] += (num_syl * O[i][j])
				total_prob += O[i][j]
		ave_syl[i] /= total_prob
	print(ave_syl)
	return ave_syl



def 



if __name__ == '__main__':
	transition_file = str('./pickles/transition.npy')
	observation_file = str('./pickles/observation.npy')
	index_dic_file = str('./pickles/index_to_word.p')
	count_dic_file = str('./pickles/count_dic.p')
	words_to_pos_file = str('./pickles/words_to_pos.p')
	pos_to_words_file = str('./pickles/pos_to_words.p')
	syl_dic = str('./pickles/syl_dic.p')

	A = np.load(transition_file, 'r')
	O = np.load(observation_file, 'r')
	index_dic = cp.load(open(index_dic_file, 'rb'))
	count_dic = cp.load(open(count_dic_file, 'rb'))
	pos_dic = cp.load(open(words_to_pos_file, 'rb'))
	pos_to_words = cp.load(open(pos_to_words_file, 'rb'))
	syl_dic = cp.load(open(syl_dic, 'rb'))


	norm = np.empty(O.shape)
	for i in range(O.shape[1]): 
		norm[:,i] = O[:,i] / count_dic[index_dic[i]]


	# pos_prob = pos_analysis(norm, pos_dic, pos_to_words, index_dic)
	# top_words = word_category(norm, index_dic)
	ave_syl = syllables(norm, index_dic, syl_dic)



