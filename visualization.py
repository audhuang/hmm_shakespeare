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
import matplotlib.pyplot as plt

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



def placement(word_list, O): 
	first = []
	last = []
	matrix = np.arange
	matrix = np.zeros([len(O), 2])	
	
	for line in word_list:
		first.append(line[-1])
		last.append(line[0]) 

	for i in range(len(O)): 
		for j in first: 
			matrix[i][0] += O[i][j]
		for k in last: 
			matrix[i][1] += O[i][j]

	return matrix 


def statistics(word_list, index_dic, syl_dic): 
	x_axis = np.arange(10)
	result = np.zeros(10)
	for line in word_list: 
		for word in line: 
			num_syl = syl_dic[index_dic[word]]
			result[num_syl] += 1

	fig = plt.figure() 
	ax = fig.add_subplot(111)
	ax.bar(x_axis, result)

	ax.set_xlabel('# syllables')
	ax.set_ylabel('number of words')
	plt.show()

# def num_lines(count_dic): 
	# for key in count_dic: 











if __name__ == '__main__':
	transition_file = str('./pickles/full_new_3203/transition_full.npy')
	observation_file = str('./pickles/full_new_3203/observation_full.npy')
	index_dic_file = str('./pickles/index_to_word.p')
	count_dic_file = str('./pickles/count_dic.p')
	words_to_pos_file = str('./pickles/words_to_pos.p')
	pos_to_words_file = str('./pickles/pos_to_words.p')
	syl_dic_file = str('./pickles/syl_dic.p')
	quatrains_file = str('./pickles/quatrains.p')
	volta_file = str('./pickles/volta.p')
	couplet_file = str('./pickles/couplet.p')
	word_list_file = str('./pickles/word_list.p')

	A = np.load(transition_file, 'r')
	O = np.load(observation_file, 'r')
	index_dic = cp.load(open(index_dic_file, 'rb'))
	count_dic = cp.load(open(count_dic_file, 'rb'))
	pos_dic = cp.load(open(words_to_pos_file, 'rb'))
	pos_to_words = cp.load(open(pos_to_words_file, 'rb'))
	syl_dic = cp.load(open(syl_dic_file, 'rb'))
	quatrains = cp.load(open(quatrains_file, 'rb'))
	volta = cp.load(open(volta_file, 'rb'))
	couplet = cp.load(open(couplet_file, 'rb'))
	word_list = cp.load(open(word_list_file, 'rb'))

	print(len(syl_dic), len(index_dic), len(count_dic), len(pos_dic))


	norm = np.empty(O.shape)
	for i in range(O.shape[1]): 
		norm[:,i] = O[:,i] / count_dic[index_dic[i]]


	# pos_prob = pos_analysis(norm, pos_dic, pos_to_words, index_dic)
	# top_words = word_category(norm, index_dic)
	# ave_syl = syllables(norm, index_dic, syl_dic)
	# quatrain_prob = placement(quatrains, norm)
	# volta_prob = placement(volta, norm)
	# couplet_prob = placement(couplet, norm)
	statistics(word_list, index_dic, syl_dic)





