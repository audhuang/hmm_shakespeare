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

	C = []                                     # the list of coefficients used to normalize each column to 1 
	F = np.zeros(L, M)                         # the foward algorithm generates an L x M matrix
	F[:,0] = np.multiply(S, O[:,obs[0]])       # initialize the first column of F via S * (obs[0] column of B)
	c_0 = np.sum(F[:,0])                       # compute the first normalizing coefficient
	C.append(c_0)                              # record c_0 
	F[:,0] = np.divide(F[:,0], c_0)            # normalize the first column so the entries sum to 1 

	# begin the forward algorithm. generate each subsequent column of F via the previous one, 
	# normalizing at each step
	for j in range(1, M):
		F[:,j] = np.dot(np.multiply(A, O[:,obs[j]]), F[:,j - 1])         # compute the new column j 
		c_j = np.sum(F[:,j])                                             # compute the jth coeff.
		C.append(c_j)                                                    # record the jth coeff.  
		F[:,j] = np.divide(F[:,j], c_j)                                  # normalize column j 

	# return the foward matrix F and the list of normalizing coefficients C (these will be used
	# to normalize the backward probabilities in the backward step)
	return (F, C) 




def backward(A, O, C, obs): 

	""" Calculates the backward probability matrix B. This is a matrix where each (i, j) entry 
	    represents P(o_(j + 1), o_(j + 1), ... o_M | X_t = i). Each (i, j) entry is the probability 
	    that the sequence ends in o_(j + 1), o_(j + 2), ... o_M where we are in hidden state i at 
	    position j. We build B from the last observation o_M up to the first o_1. Thus B has 
	    dimension L x M where L is the number of hidden states and M is the length of our sample 
	    'obs'. 

	    params:    A    the transition matrix 

	               O    the observation matrix 

	               C   the list of forward coefficients 

	               obs  the list of observations 
	""" 

	assert np.shape(A)[0] == np.shape(A)[1]    # transition matrix should be square 
	L = np.shape(A)[0]                         # L is the number of hidden states 
	M = len(obs)                               # M is the number of observations in our sample 'obs'

	B = np.zeros(L, M)                         # the backward algorithm generates an L x M matrix
	B_MM = np.ones(L)                          # the backward algorithm is initialized with all ones 

	assert len(C) == M                         # number of coeff should equal length of sequence 


	# TODO: working on indexing below 

	# initialize the last column of B and then normalize with the last coefficient of C 
	B[:,M - 1] = np.dot(np.multiply(A, O[:,obs[-1]]), B_MM)    
	B[:,M - 1] = np.divide(B[:,M - 1], C[-1]) 
	
	for j in range(2,M):


	# populate B backwards. generate each previous column of B via the one in front.  
	for j in reversed(range(M - 1)): 
		# j starts at M - 2 and goes to 0 because B[:,M - 1] is the last column
		B[:, M - 2] = np.dot(np.multiply(A, O[:, obs[M - 2]]), B[:, M - 1])

		B[:,j] = np.multiply(np.multiply(A, O[:, obs[]]), B[:,j + 1])    # compute column
		B[:,j] = np.dot(np.multiply(A, O[:, ]), B[:, ]) 
		B[:,j] = np.divide(B[:,j], C[j])                                     # normalize 

	return B




def gamma(F, B): 
	""" Computes the gamma matrix B. This is a matrix where each (i, j) entry represents gamma_j(i) 
	    = P(X_j = i | o_1, ... o_M, S, A, O). This is the probability that at the jth part of our 
	    training sequence we are in hidden state i. 

	    params:    F    the forward matrix.

	               B    the backward matrix.   
	"""

	return 

