This is a group project on generating Shakespearean sonnets for CS 155 - Machine
Learning and Data Mining at Caltech. 

The team members are:
	Ritwik Anand
	Audrey Huang 
	Dryden Bouamalay

If you are participating in the competition, it is a violation of the honor code
to view this code without permission from one of the team members above. 

The final report is stored as a PDF as final_report.pdf 
Late hours used: 13 out of 17 remaining hours

GUIDE TO CODE:

	DIRECTORIES:
		pickles
			contains all the pickled numpy arrays, lists, and dictionaries used 
			throughout all programs during poem generation 
		images
			self explanatory 

	PYTHON:
		parse.py 
			contains functions related to parsing the poems and generating 
			dictionaries that map words to indices, etc. 
		baum_welch.py 
			this file implements forward-backward and the baum-welch algorithm 
			to train a hidden markov model on unsupervised data. outputs the 
			trained transition and observations matrices as .npy files into 
			./pickles 
		poetry_generation.py 
			generates a poem with rhyming, syllable considerations, etc. 
		visualization.py
			code for investigating the meaning of the hidden states, etc.  

	TEXT FILES:
		shakespeare.txt 
			the raw sonnet data
		poems.txt 
			some selected poems from the poetry generation step 

	BASH:
		train_hmms.sh
			a script used for training multiple hidden markov models with 
			various parameters 

