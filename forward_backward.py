from __future__ import print_function
from __future__ import division
from scipy.integrate import quad
from scipy.linalg import norm
import random 
import pickle 
import sys 
import numpy as np 




# transition matrix has rows as target state and columns as start state 
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
	F = np.zeros((L, M))                         # the foward algorithm generates an L x M matrix
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

	B = np.zeros((L, M))                         # the backward algorithm generates an L x M matrix
	B_MM = np.ones(L)                          # the backward algorithm is initialized with all ones 

	assert len(C) == M                         # number of coeff should equal length of sequence 

	# initialize the last column of B and then normalize with the last coefficient of C 
	B[:,M - 1] = np.dot(np.multiply(A, O[:,obs[-1]]), B_MM)    
	B[:,M - 1] = np.divide(B[:,M - 1], C[-1]) 
	
	# goes from j = 2 to M, so M - j ranges from M - 2 to 0 (aka we work backwards starting with
	# the second to last column M - 2 to the first column 0)
	for j in range(2, M + 1):
		# compute the M - jth row via M - j + 1 and then normalize using C[M - j]
		B[:,M - j] = np.dot(np.multiply(A, O[:,obs[M - j]]), B[:, M - j + 1])
		B[:,M - j] = np.divide(B[:,M - j], C[M - j]) 

	return B




def gamma(S, F, B): 
	""" Computes the gamma matrix G. This is a matrix where each (i, j) entry represents gamma_j(i) 
	    = P(X_j = i | o_1, ... o_M, S, A, O). This is the probability that at the jth part of our 
	    training sequence we are in hidden state i. 

	    params:    S    the starting state distribution 

	    		   F    the forward matrix.

	               B    the backward matrix.   
	"""

	assert np.shape(F) == np.shape(B)    	# F & B should have shape L x M 
	L = np.shape(F)[0]                      # the number of hidden states is L 
	M = np.shape(F)[1] 

	B_MM = np.ones(L)                       # recreate B_MM from backward algorithm 

	F = np.hstack((S[:,np.newaxis], F))     # add to F the vector S as its first column 
	B = np.hstack((B, B_MM[:,np.newaxis]))  # add to B the vector B_MM as its last column

	assert np.shape(F) == np.shape(B)       # F and B should still be the same size 
	G = np.multiply(F, B)                   # multiply F and B entrywise to get G

	# renormalize 
	for i in range(M):
		G[:,i] = np.divide(G[:,i], np.sum(G[:,i]))

	# now remove the first column gamma_0 such that gamma is L x M 
	G = G[:,1:]
	assert np.shape(G) == (L, M)

	return G 




def xi(A, O, S, F, B):
	""" Computes the xi matrix E. This is a 3-dimensional matrix M x L x L 

		params:
	""" 
	
	assert np.shape(F) == np.shape(B)    	# F & B should have shape L x M 
	L = np.shape(F)[0]                      # the number of hidden states is L 
	M = np.shape(F)[1]                      # the length of the sequence is M 

	B_MM = np.ones(L)                       # recreate B_MM from backward algorithm 

	F = np.hstack((S[:,np.newaxis], F))     # add to F the vector S as its first column 
	B = np.hstack((B, B_MM[:,np.newaxis]))  # add to B the vector B_MM as its last column

	# now column F_1 correpsonds to B_1, etc.

	# initialize the 3D array 
	E = np.ones((M, L, L))

	# for every step in the M-length sequence, generate an L x L matrix as the t'th entry of our
	# M x L x L matrix 
	for t in range(M):
		t_matrix = np.ones((L, L))
		for i in range(L):
			for j in range(L): 
				t_matrix[i][j] = F[i][t] * A[i][j] * B[j][t + 1] * O[j][t + 1]
		# normalize the column ... or the row? 
		for j in range(L):
			t_matrix[:,j] = np.divide(t_matrix[:,j], np.sum(t_matrix[:,j]))
		E[t] = t_matrix 

	return E 
	



def difference(A, B):
	""" This function compututes the difference between matrices A and O (entrywise) and then 
	    returns the Frobenius norm of their difference. This acts as a tolerance for our convergence
	    condition.
	"""
	T = A - B
	return norm(T)

def indicator(a, b):
	if (a == b):
		return 1
	return 0 


def baum_welch(L, M, obs): 
	""" Runs the Baum-Welch algorithm on a list of training sequences X. Returns trained transition 
	    and observation matrices A and O. 

	    params:    L    int  - the number of hidden states to use for the HMM 
	               M    int  - the number of distinct observations possible in the training set 
	    	       X    list - the list of sequences used for the training. each sequence is assumed 
	    	                   to be a list of integers that correctly index into M
	"""


	# initialize start state S 
	S = np.random.uniform(size=L)    # initialize a start state distribution S for the HMM 
	S = np.divide(S, np.sum(S))      # normalize the vector to 1 

	# initialize transition and observation matrices A and O

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
	
	# for the moment, just do one step through the data 
	
	# TRAIN TRANSITION MATRIX 
	# for every (a,b) entry of A 
	print("TRAINING TRANSITION MATRIX...") 
	for a in range(L):
		for b in range(L): 

			print("Calculating A(a,b) for (a,b) = ", "(", a, ",", b, ")")

			# for every sample in the list of observations 
			for o in obs: 
				
				# perform forward and backward on the sample 
				(F, C) = forward(S, A, O, o)
				B = backward(A, O, C, o)

				# from forward and backward, compute gamma and xi 
				G = gamma(S, F, B)
				E = xi(A, O, S, F, B)

				# using gamma and xi, compute the (a, b) entry. sum xi(a,b) across the sequence 
				# to get the numerator, and sum gamma(a) across the sequence to get the denominator 
				numerator = 0.0
				denominator = 0.0
				assert (len(o) == np.shape(E)[0]) and (len(o) == np.shape(G)[1])
				for i in range(len(o)):
					numerator += E[i][a][b]
					denominator += G[a][i]

			A[a][b] = numerator / denominator 

	# TRAIN OBSERVATION MATRIX
	# for every (w, z) entry of O
	print("TRAINING OBSERVATION MATRIX...")
	sample_number = 0 
	for w in range(L):
		for z in range(M):

			print("Calculating Z(w,z) for (w,z) = ", "(", w, ",", z, ")")

			# for every sample in the list of observations 
			for o in obs:

				# perform forward and backward 
				(F, C) = forward(S, A, O, o)
				B = backward(A, O, C, o)

				# from forward and backward, compute gamma. xi is not needed here 
				G = gamma(S, F, B)

				# using gamma, compute the (w, z) entry
				numerator = 0.0
				denominator = 0.0 
				for i in range(len(o)):
					numerator += indicator(o[i], z) * G[w][i]
					denominator += G[w][i]

					# if (indicator(o[i],z) == 1):
					# 	print("INDICATOR") 

			O[w][z] = numerator / denominator 

	# return the trained transition and observation matrices (A, O)  
	return (A,O) 

def check_obs(idx, obs):
	""" Checks every term in every sequence of obs and sees if any term is >= idx or < 0. If true, 
	    returns false. Otherwise returns true.  
	"""
	for o in obs: 
		for term in o: 
			if (term >= idx) or (term < 0):  
				return False 
	return True 

# MAX_INDEX = 3232 
if __name__ == '__main__':

	# unpickle the list of observations 
	obs = pickle.load(open('sonnet_to_index.p', 'rb'))
	# for the moment only train on the first 100 samples 
	obs = obs[:2]

	print("Number of samples in dataset is: ", len(obs))

	MAX_OBS = 3232      # the total number of distinct observations in our dataset 
	MAX_STATES = 10    # start out with 100 hidden states and see where that takes us 

	# sanity check that no index in the dataset is >= MAX_OBS or < 0.
	assert check_obs(MAX_OBS, obs) == True     
	
	# attempt to perform training on the list of observations obs 
	(A, O) = baum_welch(MAX_STATES, MAX_OBS, obs) 
	
	# A = np.array([1,2,3])
	# O = np.array([3,4,5])
	print("FINAL TRANSITION MATRIX IS: \n", A)
	print("FINAL OBSERVATION MATRIX IS: \n", O)

	print("norm of O:", norm(O))    # O is mostly sparse if trained on few samples -- initialzied with zeros  