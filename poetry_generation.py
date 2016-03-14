
# coding: utf-8

# In[50]:

from __future__ import print_function
from __future__ import division
from scipy.integrate import quad
from scipy.linalg import norm
import random 
import os 
import numpy as np 
import pickle



def convert_syllables_dict(s_dict):

    # generate new dictionary that maps words to 
    # number of syllables (int) 
    new_dict = {} 
    for k, v in s_dict.iteritems():
        if type(v) is list: 
            new_dict[k] = v[0] 
        if type(v) is int: 
            new_dict[k] = v 

    # check types 
    for k, v in new_dict.iteritems():
        assert type(new_dict[k]) is int 
    return new_dict 

# def check_syllables_dict(s_dict):
#     for k, v in s_dict.iteritems():
#         if type(v) is list:
#             if len(v) != 1:
#                 print(k)
#                 print(v)
#         else:
#             assert type(v) is int 
#     return True 

def convert_to_words(n):
    words = []
    for num in n:
        words.append(states_to_words_dict[num])
        #words.append("niggerdly")
    
    return words

def check_syllables(words):
    syllables = 0
    for word in words:
        # print(type(syllables_dict[word]))
        # print(word)
        # print(syllables_dict['wherefore'])
        # print(type(syllables_dict[word]))
        syllables += syllables_dict[word]
    
    return syllables

def pick_start(S):
    
    pick = random.random()
    
    for i in range(S.shape[0]):
        pick = pick - S[i]
        if pick <= 0:
            return i
    
    
    return 0

def pick_rhyming_start(S, A, O, rhyme):
    
    rhyme_start = np.zeroes(S.shape)
    for i in range(O.shape[1]):
        for j in range(rhyme_start.shape[0]):
            rhyme_start[j] += O[i][rhyme]*A[j][i]
    
    return rhyme_start

def pick_next(A, current):
    
    pick = random.random()
    for i in range(A.shape[0]):
        pick = pick - A[i][current]
        if pick <= 0:
            return i
    
    
    return 0

def convert_to_observed(states, O):
    observed = []
    
    for state in states:
        pick = random.random()
        for i in range(O.shape[1]):
            pick = pick - O[state][i]
            if pick <= 0:
                observed.append(i)
                break
    
    
    return observed

def gen_line(A, O, start, length):
    
    syllables_right = False
    
    while (syllables_right == False):
        
        hidden_states = []
        current = start
        hidden_states.append(current)
        
        for i in range(length - 1):
            current = pick_next(A, current)
            hidden_states.append(current)
        
        print (hidden_states)
        observed_states = convert_to_observed(hidden_states, O)
        print (observed_states)
        current_line = convert_to_words(observed_states)
        
        # syllables_right = True
        syllables = check_syllables(current_line)
        
        if syllables == 10:
            syllables_right = True
            
    return current_line

def gen_couplets(A, O, S, avg_words, std):
    length = int(np.random.normal(avg_words, std))
    print (length)
    start = pick_start(S)
    line1 = gen_line(A, O, start, length)
    
    length = int(np.random.normal(avg_words, std))
    print (length)
    start = pick_start(S)
    line2 = gen_line(A, O, start, length)
    
    
    #length = np.random.normal(avg_words, std) - 1
    #rhyming = random.choice(rhyming_dict[line1[-1]])
    #S_rhyme = pick_rhyming_start(line1[-1])
    #start = pick_rhyming_start(S_rhyme)
    #line2 = gen_line(A, O, start, length) + rhyming
    
    return (line1, line2)

def poem_gen(S, A, O, avg_words, std):
    
    poem = []
    for i in range(14):
        poem.append([])
    
    for i in range(3):
        for j in range(2):
            (poem[4*i + j], poem[4*i + j + 2]) = gen_couplets(A, O, S, avg_words, std)
        
    (poem[12], poem[13]) = gen_couplets(A, O, S, avg_words, std)

    return poem
    
    
if __name__ == '__main__':
    L = 10
    M = 10

    S = np.random.uniform(size=L)    # initialize a start state distribution S for the HMM 
    S = np.divide(S, np.sum(S))      # normalize the vector to 1 


    # the rows of A are the target states and the columns of A are the start states. 
        # given a start state, one of the target states must be choosen so each column is normalized
    A = np.random.rand(L, L) 
    for i in range(L): 
        A[:,i] = np.divide(A[:,i], np.sum(A[:,i]))    

        # given some hidden state, there must be some observation, so every row of this matrix should
        # be normalized
    O = np.random.rand(L, M) 
    for i in range(L):
        O[i,:] = np.divide(O[i,:], np.sum(O[i,:])) 

    average_length = 8.15977443609
    std = 1.1474220639

    rhyming_dict = {}
    syllables_dict = {}
    states_to_words_dict = {}

    A = np.load(os.getcwd() + '/pickles/full_001/transition_full.npy', 'r') 
    O = np.load(os.getcwd() + '/pickles/full_001/observation_full.npy', 'r') 
    S = np.load(os.getcwd() + '/pickles/full_001/start_full.npy', 'r') 

    # A = pickle.load( open( "transition.npy", "rb" ) )
    # O = pickle.load( open( "observation.npy", "rb" ) )

    #S = pickle.load( open( "save.p", "rb" ) )
    #rhyming_dict = pickle.load( open( "rhyme_dic.p", "rb" ) )
    nonhomogenous_syllables_dict = pickle.load( open( "./pickles/syl_dic.p", "rb" ) )
    syllables_dict = convert_syllables_dict(nonhomogenous_syllables_dict)
    states_to_words_dict = pickle.load( open( "./pickles/index_to_word.p", "rb" ) )
    words_to_state_dict = {v: k for k, v in states_to_words_dict.iteritems()}
    # print(states_to_words_dict)
    # check_syllables_dict(syllables_dict)
    print(poem_gen(S, A, O, average_length, std))

# In[ ]:



