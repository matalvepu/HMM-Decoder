# HMM-Decoder

Author: Md Kamrul Hasan
Email:  mhasan8@cs.rochester.edu
Date:  9/7/2017

==================================================================================================
Description:
Implementation of the HMM decoder for Parts of Speech Tagging

I did it as a part of homework problem in the Statistical Speech and Language Processing class taught by Prof Daniel Gildea (https://www.cs.rochester.edu/~gildea/) in Spring 2014.


==================================================================================================
Instruction to run:

python viterbi_final.py train test

It takes around 8 minutes to complete

==================================================================================================

Preprocessing:
I have preprocessed all the transition probability, emission probability and intitial state distribution in preprocessing step. I used log technique to avoid floating precision problem. I also used smoothing tehcnique to handle unseen tag or word in test file.


Accuray: 94 % 
