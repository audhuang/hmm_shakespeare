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
syl_dic = {} # dictionary of unique word : number of syllables if they're in cmudict
bad_dic = {} # dictionary of unique word : number of syllables if they're not in cmudict

# parses the text file 'shakespeare.txt' and adds each unique word to a dictionary,
# WORD_DIC, with a unique index 
def parse(): 

	# open 'shakespeare.txt'
	inp = open('./project2data/shakespeare.txt', 'r')

	# use regex to parse unique words into list 'unique'
	text = inp.read().lower() # make everything lowercase
	text = re.sub('[^a-z\ \'\-]+', ' ', text) # replace punctuation with spaces
	text = re.sub("(?=.*\w)^(\w|')+", ' ', text)

	words = list(text.split()) # split file into a list of words by spaces
	unique = list(set(words)) # get unique list of words

	print(unique)
	# for each unique word, add as key to dictionary with unique index as value
	i = 0
	for word in unique: 
		word_dic[word] = i
		i += 1


# there are a bunch of apostrophed, dashed, weirdly spelled, or archaic words that
# aren't in the cmu dictionary. we will have to manually determine the number of 
# syllables. 
# 
# though there are likely some errors, generally, we can state that the
# number of syllables in a word is equal to the number of vowels. however we
# need to take into account exceptions, for example diphthongs which are groups
# of vowels that only count as one syllable, or words ending in 'sm' which is
# two syllables, not one. all of these exceptions are listed below. 
# 
# stores word as key and number of syllables as value in "BAD_DICT"
def bad_syllables(not_in_dic): 

	# list of diphthongs, which all count as one syllable. i didn't include:
	# 'ia' as in 'variation'
	# 'io' as in 'violet', but i check for '-tion' later
	# 'ui' as in 'ruin'
	# 'oe' as in 'goest'
	thongs = ['aa', 'ae', 'ai', 'ao', 'au', 'ay', 'ea', 'ee', 'ei', 'eo', 'eu', 'ey']
	thongs += ['ii', 'ie', 'iu', 'oa', 'oi', 'oo', 'ou', 'oy', 'ua', 'ue', 'uo', 'uy']
	
	# list of past tense versions of diphthongs, which count as one syllable. didn't include: 
	# 'oed' as in 'forgoed'
	past_thongs = ['aed', 'eed', 'ied', 'ued']
	vowels = 'aeiouy'

	for i in not_in_dic: 
		# initiate number of syllables to zero
		num_syl = 0

		# if first word in hyphenated phrase ends with e, first subtract a syllable
		if 'e-' in i:  
			num_syl -= 1

		# remove all dashes and apostrophes
		word = i.translate(string.maketrans("",""), string.punctuation)
		
		# count number of vowels
		for char in word: 
			if char in vowels: 
				num_syl += 1

		# subtract number of diphthongs 
		for thong in thongs: 
			if thong in word: 
				num_syl -= 1

		# if ends in 'e', subtract one vowel/syllable
		if word[-1] == 'e' and word[-2] not in ['l', 'r']: 
			num_syl -= 1

		# if ends in 'ed' or 'es', subtract one vowel/syllable 
		# didn't include 'es'
		if word[-2:] == 'ed' or word[-3:] == 'eds': 
			num_syl -= 1

		# the exception to the above rule is if it ends in 'ded' or something 
		if word[-3:] in ['bed', 'ded', 'ted', 'bred', 'dred', 'tred']: 
			num_syl += 1

		# if ends in 'tion', subtract one vowel/syllable since we didn't
		# count 'io' as a diphthong
		if word[-4:] == 'tion' or word[-5:] == 'tions': 
			num_syl -= 1

		# if ends in 'oing' as in 'doing', add one vowel/syllable 
		if word[-4:] == 'oing' or word[-5:] == 'oings': 
			num_syl += 1

		# if y is a consonant (when it's the first letter), subtract a vowel
		if word[0] == 'y': 
			num_syl -= 1

		# if 'uie' is in the word, ie 'quiet', there should be two syllables 
		if 'uie' in word: 
			num_syl += 1

		# if ends in 'sm', that's two syllables not one
		if word[-2:] == 'sm': 
			num_syl += 1

		# if the word ends in 'ed' but also has a diphthong, we will undercount
		# based on the rules above. so we add a syllable to our count if 
		# there is a past-tense diphthong in our word
		for past_thong in past_thongs: 
			if past_thong in word: 
				num_syl += 1

		# there are a few words that are just a consonant and a dash, but this
		# this counts as one syllable even though there are no vowels
		if num_syl == 0:
			num_syl += 1


		# the number of syllables is equal to the number of vowels
		bad_dic[i] = num_syl

		print(bad_dic)


# fills in SYL_DICT and BAD_DICT with word as key and number of syllables as value
def syllables(): 
	not_in_dic = [] # list of words not in cmudict, so we have to manually do them
	
	# go through each unique word in our dictionary of indexed words, WORD_DIC
	for key in word_dic:
		# if word is in cmudict, look up the number of syllables and store in SYL_DIC
		if key in d: 
			syl_dic[key] = list(set(len([y for y in x if y[-1].isdigit()]) for x in d[key]))
		# if it's not in cmudict, append to the list of words not in cmudict
		else: 
			not_in_dic.append(key)

	# remove a string that's just a comma and not a word
	not_in_dic.remove("'")
	
	# call function to calculate number of syllables for each word not in cmudict
	bad_syllables(not_in_dic)
	

	

if __name__ == '__main__':
	parse() # parse file into word_dic, word : index
	syllables() # parse words into syl_dic and bad_dic, word : number of syllables
