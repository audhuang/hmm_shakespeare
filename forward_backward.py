from __future__ import print_function
from __future__ import division
from scipy.integrate import quad
import random 
import numpy as np 




def forward(S, A, O, obs):

	""" Calculates the forward probability matrix F. This is a matrix where each (i, j) entry 
	    represents P(o_1, o_2, ... o_j, X_t = i| A, O). In other words, each (i, j) entry is the 
	    probability that the observed sequence is o_1, ... o_j and that at position j we are in 
	    hidden state i. We build F from the first observation o_1 up to the entire observed sequence 
	    o_1, ... o_M. Thus F has dimension L x M where L is the number of hidden states and M is the 
	    length of our input sample 'obs'. 

	    params:    S    np.array - state vector for starting distribution.

			       A    np.array - transition matrix, L x L for L hidden states, each (i, j) 
			            entry is P(X_i | X_j), or the probability of transitioning from start 
			            state X_j (column entry) to target state X_i (row entry).

			       O    np.array - observation matrix, L x M' for L hidden states and M' total
			            possible observations. each (i, j) entry is P(Y_j | X_i), or the 
			            probability of observing observation Y_j while in state X_i.

			       obs  np.array, list - the observations. these are assumed to be integers that
			            index correctly into A and O. 
	"""

	assert np.shape(A)[0] == np.shape(A)[1]    # transition matrix should be square 
	L = np.shape(A)[0]                         # L is the number of hidden states 
	M = len(obs)                               # M is the number of observations in our sample 'obs'  

	F = np.zeros(L, M)                         # the foward algorithm generates an L x M matrix
	F[:,0] = np.multiply(S, O[:,obs[0]])       # initialize the first column of F via S * (obs[0] column of B)
	F[:,0] = np.divide(F[:,0], np.sum(F[:,0])) # normalize the first column so the entries sum to 1 

	# begin the forward algorithm. generate each subsequent column of F via the previous one, 
	# normalizing at each step
	for j in range(1, M):
		F[:,j] = np.multiply(F[:,j - 1], np.multiply(A, O[:,obs[j]]))    # compute the new column j 
		F[:,j] = np.divide(F[:,j], np.sum(F[:,j]))                       # normalize column j 

	return F 

# currently unnormalized. we should probably use the same coefficients as in the forward algorithm
# so we can calculate gamma via: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm 
def backward(A, O, obs): 

	""" Calculates the backward probability matrix B. This is a matrix where each (i, j) entry 
	    represents P(o_(j + 1), o_(j + 1), ... o_M | X_t = i). Each (i, j) entry is the probability 
	    that the sequence ends in o_(j + 1), o_(j + 2), ... o_M where we are in hidden state i at 
	    position j. We build B from the last observation o_M up to the first o_1. Thus B has 
	    dimension L x M where L is the number of hidden states and M is the length of our sample 
	    'obs'. 

	    params:   
	""" 

	assert np.shape(A)[0] == np.shape(A)[1]    # transition matrix should be square 
	L = np.shape(A)[0]                         # L is the number of hidden states 
	M = len(obs)                               # M is the number of observations in our sample 'obs'

	B = np.zeros(L, M)                         # the backward algorithm generates an L x M matrix 
	B[:,M - 1] = np.ones(L)                    # initialize the last column of B. last column is not normalized. 

	# begin the backward algorithm. generate each previous column of B via the next one.
	# j starts at M - 2 and goes to 0 because B[:,M - 1] is the last column.  
	for j in reversed(range(M - 1)): 
		B[:,j] = np.multiply(np.multiply(A, O[:, obs[j + 1]]), B[:,j + 1])
		# not currently normalized  

	return B
