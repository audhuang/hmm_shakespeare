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

def pos_analysis(A, O, pos_dic, index_dic): 
	dic = {}
	for key in pos_dic: 
		dic[key] = np.zeros(10)

	count = 0

	for i in range(len(O.shape[0])): 
		for j in range(len(i)): 
			pos = pos_dic[index_dic[j]]
			dic[pos][i] += O[i][j]





if __name__ == '__main__':
	transition_file = str('./pickles/' + sys.argv[1])
	observation_file = str('./pickles/' + sys.argv[2])
	index_dic_file = str('./pickles/' + sys.argv[3])
	count_dic_file = str('./pickles/' + sys.argv[4])
	words_to_pos_file = str('./pickles/' + sys.argv[5]), 'rb')

	A = np.load(transition_file, 'r')
	O = np.load(observation_file, 'r')
	index_dic = cp.load(open(index_dic_file, 'rb'))
	count_dic = cp.load(open(count_dic_file, 'rb'))
	pos_dic = cp.load(open(words_to_pos_file, 'rb'))
	
	norm = np.empty(O.shape)
	for i in range(O.shape[1]): 
		norm[:,i] = O[:,i] / count_dic[index_dic[i]]



