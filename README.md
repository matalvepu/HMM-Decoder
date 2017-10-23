# HMM-Decoder

Author: Md Kamrul Hasan
Email: mhasan8@cs.rochester.edu

==================================================================================================
Description:
Implementation of the HMM decoder for Parts of Speech Tagging


==================================================================================================
Instruction to run:

python viterbi_final.py train test

It takes around 8 minutes to complete

==================================================================================================

Preprocessing:
I have preprocessed all the transition probability, emission probability and intitial state distribution in preprocessing step. I used log technique to avoid floating precision problem. I also used smoothing tehcnique to handle unseen tag or word in test file.


Accuray: 94 % 
