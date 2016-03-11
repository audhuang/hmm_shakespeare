from __future__ import print_function
from __future__ import division
from scipy.integrate import quad
import random 
import numpy as np 
# import matplotlib.pyplot as plt


# just prints the line as a list of words for now because i'm not sure what to 
# do with it. have to deal with punctuation, capitalization, etc. 
def parse(): 
	inp = open('shakespeare.txt')

	# stores list of words in the line in "words"
	for line in inp: 
		words = line.split()
		print(words)

if __name__ == '__main__':
	parse()

# between poems, the lines are [], [], [number]