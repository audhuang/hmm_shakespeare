#!/bin/bash
echo "Enter number of hidden states: "
read NUM_HIDDEN
echo "Enter tolerance: "
read TOLERANCE  
python baum_welch.py ./pickles/sonnet_to_index.p 3204 $NUM_HIDDEN $TOLERANCE full
python baum_welch.py ./pickles/volta.p 3204 $NUM_HIDDEN $TOLERANCE volta
python baum_welch.py ./pickles/quatrains.p 3204 $NUM_HIDDEN $TOLERANCE quatrains
python baum_welch.py ./pickles/couplet.p 3204 $NUM_HIDDEN $TOLERANCE couplet
