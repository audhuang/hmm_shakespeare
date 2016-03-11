from __future__ import print_function
from __future__ import division
from scipy.integrate import quad
import random 
import numpy as np 




def forward(S, T, B, obs):

	""" Calculates the forward probability matrix F. This is a matrix where each (i, j) entry 
	    represents P(o_1, o_2, ... o_j, X_t = i| T, B). In other words, each (i, j) entry is the 
	    probability that the observed sequence is o_1, ... o_j and that at o_j we are in hidden 
	    state i. We build F from the first observation o_1 up to the entire observed sequence o_1,
	    ... o_M. Thus F has dimension L x M where L is the number of hidden states and M is the 
	    length of our input sample obs. 

	    parameters:    S    np.array - state vector for starting distribution.

			           T    np.array - transition matrix, L x L for L hidden states, each (i, j) 
			                entry is P(X_i | X_j), or the probability of transitioning from start 
			                state X_j (column entry) to target state X_i (row entry).

			           B    np.array - observation matrix, L x M' for L hidden states and M' 
			                possible observations. each (i, j) entry is P(Y_j | X_i), or the 
			                probability of observing observation Y_j while in state X_i.

			           obs  np.array or list - the list of observations. these are assumed to index 
			                correctly into T and B (should be a list of integers). 
	"""

	assert np.shape(T)[0] == np.shape(T)[1]    # transition matrix should be square 
	L = np.shape(T)[0]                         # L is the number of hidden states 
	M = len(obs)                               # M is the number of observations in our sample 'obs'  

	F = np.zeros(L, M)                         # the foward algorithm generates an L x M matrix
	F[:,0] = np.multiply(S, B[:,obs[0]])       # initialize the first column of F via S * (obs[0] column of B)
	F[:,0] = np.divide(F[:,0], np.sum(F[:,0])) # normalize the first column so the entries sum to 1 

	# begin the forward algorithm. generate each subsequent column of F via the previous one, 
	# normalizing at each step
	for j in range(1, M):
		F[:,j] = np.multiply(F[:,j - 1], np.multiply(T, B[:,obs[j]]))    # compute the new column j 
		F[:,j] = np.divide(F[:,j], np.sum(F[:,j]))                       # normalize column j 

	return F 

def backward(T, B, obs): 

	""" Calculates the backward probability matrix B. 
	""" 
	
	return 
