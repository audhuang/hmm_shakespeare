from __future__ import print_function
from __future__ import division
from scipy.integrate import quad
import random 
import numpy as np 
import codecs
import string
import re

from nltk.corpus import cmudict

d = cmudict.dict() # dicionary of syllables from cmudict
word_dic = {} # dictionary of unique word : unique index 

# parses the text file 'shakespeare.txt' and adds each unique word to a dictionary,
# WORD_DIC, with a unique index 
def parse(): 

	# open 'shakespeare.txt'
	inp = open('./project2data/shakespeare.txt', 'r')

	# use regex to parse unique words into list 'unique'
	text = inp.read().lower() # make everything lowercase
	text = re.sub('[^a-z\ \'\-]+', ' ', text) # replace punctuation with spaces
	words = list(text.split()) # split file into a list of words by spaces
	unique = list(set(words)) # get unique list of words

	# for each unique word, add as key to dictionary with unique index as value
	i = 0
	for word in unique: 
		word_dic[word] = i
		i += 1



def syllables(): 
	print(d['lour\'st'][0])


if __name__ == '__main__':
	parse()
