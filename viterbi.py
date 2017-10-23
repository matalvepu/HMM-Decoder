import numpy as np
import sys

#this lemba parameter will be used for smoothing
lembda=0.00001

#It takes the train file and generate three map of count.
# a[tag1][tag2] : count of of transiotion from tag1 to tag2
#b[tag][word]: count of how many time (tag,word) pair is seen 
#pi[tag]: count how many time tag started the sentence. It is for initial distribution

def pre_process_data(f_train):
	#observation count (tag,word)
	b = {}
	#transition count
	a = {}
	#initial tag count
	pi = {}

	with open(f_train) as f:
	    for line in f:
	        line = line.split()
	        prev_tag = None
	        for i in range(1, len(line), 2):
	            w, t = line[i], line[i + 1]
	            if t in b:
	                if w in b[t]:
	                    b[t][w] += 1
	                else:
	                    b[t][w] = 1
	            else:
	                b[t] = {w: 1}

	            if i == 1:
	                prev_tag = t
	                if t in pi:
	                    pi[t] += 1
	                else:
	                    pi[t] = 1
	                continue

	            if prev_tag in a:
	                if t in a[prev_tag]:
	                    a[prev_tag][t] += 1
	                else:
	                    a[prev_tag][t] = 1
	            else:
	                a[prev_tag] = {t: 1}

	            prev_tag = t


	states = set()	
	for tag in a:
	    states.add(tag)
	    for next_tag in a[tag]:
	        states.add(next_tag)

	return [b,a,pi,list(states)]

# it can culate the transition, emission probability and 
# initial distribution from the count
# It also used smoothing paramter
def normalize(a,b,pi):
	for i in a.keys():
		s=float(sum(a[i].values()))
		for j in a[i].keys():
			a[i][j]=np.log((a[i][j]+lembda)/(s +len(a[i])*lembda))
		a[i]["c_s"]=s

	for i in b.keys():
		s=float(sum(b[i].values()))
		for j in b[i].keys():
			b[i][j]=np.log((b[i][j]+lembda)/(s+len(b[i])*lembda))
		b[i]["c_s"]=s

	s=float(sum(pi.values()))
	for i in pi.keys():
		pi[i]=np.log(pi[i]/s)

	return a,b,pi

#returns log of transition probability
def log_transition_prob(a,s,s_next):
	if s in a:
		if s_next in a[s]:			
			return a[s][s_next]		
		else:
			return np.log(lembda/(a[s]["c_s"] +len(a[s])*lembda))

	return - np.inf
#return the log of initial state probability
def log_pi_prob(pi,s):
	if s in pi:
		return pi[s]
	else:
		return - np.inf

#return the log of emission probability
def log_emission_prob(b,s,w):
	if s in b:
		if w in b[s]:			
			return b[s][w]
		else:
			return np.log(lembda/(b[s]["c_s"]+len(b[s])*lembda))

	return - np.inf

# it is the code for viterbi decoding
# I used log probability 
def viterbi_decoding(a,b,pi,states,observations):
	viterbi=np.zeros((len(states),len(observations)))
	back_pointer=np.zeros((len(states),len(observations)))

	for s in range(len(states)):
		viterbi[s][0]=log_pi_prob(pi,states[s])+log_emission_prob(b,states[s],observations[0])
		back_pointer[s][0]=0.

	for t in range(1,len(observations)):
		for s in range(len(states)):
			max_prob=-np.inf
			max_back_pointer_prob=-np.inf
			max_back_pointer_index=-1
			for s_prev in range(len(states)):
				val=viterbi[s_prev][t-1]+log_transition_prob(a,states[s_prev],states[s])
				if val>max_back_pointer_prob:
					max_back_pointer_prob=val
					max_back_pointer_index=s_prev
				val+=log_emission_prob(b,states[s],observations[t])					
				if val>max_prob:
					max_prob=val

			viterbi[s][t]=max_prob
			back_pointer[s][t]=max_back_pointer_index

	#this part back track and generate hidden sequence
	last_column=viterbi[:,len(observations)-1]
	s=np.argmax(last_column)
	hidden_seq=[]
	for t in range(len(observations)-1,-1,-1):
		hidden_seq.append(states[s])
		s=int(back_pointer[s][t])
	
	return hidden_seq[::-1]

#calculate relative accuracy
def get_sentence_accuracy(tag,hidden_seq):
	correct=0.
	for i in range(len(tag)):
		if tag[i]==hidden_seq[i]:
			correct+=1
	return correct/len(tag)

#It read all test sentences and do decoding and generate accuracy
def test_accuracy(a,b,pi,states,f_test):
	sen_count=0.
	accuracy_sum=0.
	with open(f_test) as f:
	    for line in f:
	    	if not line:
	    		continue
	        line = line.split()
	        sen=line[1::2]
	        tag=line[2::2]
	        for i in range(1,len(line),2):
	        	sen.append(line[i])
	        for i in range(2,len(line),2):
	        	tag.append(line[i])

	        hidden_seq=viterbi_decoding(a,b,pi,states,sen)
	        acc=get_sentence_accuracy(tag,hidden_seq)
	        accuracy_sum+=acc
	        sen_count+=1

	return accuracy_sum/sen_count


def main():
	f_train=sys.argv[1]
	f_test=sys.argv[2]

	# a[tag1][tag2] : count of of transiotion from tag1 to tag2
	#b[tag][word]: count of how many time (tag,word) pair is seen 
	#pi[tag]: count how many time tag started the sentence. It is for initial distribution
	b,a,pi,states=pre_process_data(f_train)
	#generate probability . Used smoothing technique too
	a,b,pi=normalize(a,b,pi)

	accuracy=test_accuracy(a,b,pi,states,f_test)
	print("Accuracy: ",accuracy)

if __name__ == '__main__':
    main()