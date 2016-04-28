# Python code to read suppliers text and convert into pickle file. 

import re 
import itertools 
import ast
import numpy as np
from scipy import sparse 

# Functions
# Training function
#Extract the information from the supplier file
def supp_extract(supp_filename):
    supp_text_file = open(supp_filename, "r")
    lines = supp_text_file.readlines()
    supp_text_file.close()
    totalwordlist = list()
    # Extracting words from the lines 
    for i in range(1, len(lines)):
        totalwordlist.append(re.sub("[^\w-]"," ",lines[i]).split())
    # Remove the first word from the list (as it is customer id)
    totalwordlist = [i[1:] for i in totalwordlist]
    wordlist1d = list((itertools.chain.from_iterable(totalwordlist)))
    wordlist1d = [x.lower() for x in wordlist1d] 
    # Remove any words that are repeated 
    set_wordlist = list(set(wordlist1d))
    ltw = len(totalwordlist)
    lsw = len(set_wordlist)
    prob_word = np.float64(np.zeros(lsw))
    # Create sparse matrix for joint probability matrix
    joint_prob_mat = sparse.csr_matrix((lsw, lsw))
    # Calculate word probability 
    for i in range(0,lsw):
        prob_word[i] = wordlist1d.count(set_wordlist[i])
    prob_word = 1/prob_word
    # Calculate joint probability matrix
    for i in range(0,ltw):
        for j in range(0,len(totalwordlist[i])-1):
            k = set_wordlist.index(totalwordlist[i][j].lower())
            l = set_wordlist.index(totalwordlist[i][j+1].lower())
            joint_prob_mat[k,l] = joint_prob_mat[k,l] + 1
    return joint_prob_mat, prob_word, set_wordlist

# Testing function
# Extract information from invoice
def invoice_extract(in_filename, joint_prob_mat, prob_word, set_wordlist):
    in_text_file = open(in_filename, "r")
    # Extract lines from invoice 
    lines = in_text_file.readlines()
    l_id = np.zeros(len(lines))
    w2 = list()
    invoice_dict = {}
    k=0
    # Analysing every word and creating matrix with identified values
    for i  in range(0, len(lines)):
        invoice_dict[i] = ast.literal_eval(lines[i])
        w1 = invoice_dict[i].get('word').lower()     
        if set_wordlist.count(w1)>0:
           l_id[i] =  set_wordlist.index(w1)
           if k==0:
               fmap = np.array([i,invoice_dict[i].get('line_id'),invoice_dict[i].get('word_id'),prob_word[l_id[i]],l_id[i]]) 
           else:
               nrw = np.array([i,invoice_dict[i].get('line_id'),invoice_dict[i].get('word_id'),prob_word[l_id[i]],l_id[i]])
               fmap = np.vstack([fmap, nrw])
           k=k+1
       
    # Find Connected components
    lbl = np.zeros(fmap.shape[0])
    fmap = fmap[np.argsort(fmap[:,2]),:]
    lbl = np.ones([fmap.shape[0]])
    lbl_val = fmap[:,3]+0.0
    for i in range(0, fmap.shape[0]-1):
        if np.logical_and(fmap[i,1]==fmap[i+1,1],fmap[i+1,2]-fmap[i,2]==1.0):
            if joint_prob_mat[np.uint16(fmap[i,0]),np.uint16(fmap[i+1,0])]>0:
                lbl[i+1]=lbl[i]
                lbl_val[i]= lbl_val[i] + joint_prob_mat[np.uint16(fmap[i,0]),np.uint16(fmap[i+1,0])]*fmap[i,3]
                lbl_val[i+1]= lbl_val[i+1] + joint_prob_mat[np.uint16(fmap[i,0]),np.uint16(fmap[i+1,0])]*fmap[i+1,3]              
        else:  
            lbl[i+1] = lbl[i]+1  
    lbl_sum = np.zeros(np.uint16(lbl.max()))
    # Find label with highest probability value 
    for i in range(1,np.uint16(lbl.max())):
       lbl_sum[i-1] =np.sum(lbl_val[lbl==i])
    ll = np.argmax(lbl_sum)+1
    # Extract the name of the largest label
    s_snm = len(lbl[lbl==ll])
    s_name = list()
    for i in range(0,fmap.shape[0]):
        if lbl[i]==ll:
            s_name.append(set_wordlist[np.uint16(fmap[i,4])].upper())
    return s_name


# Main Function 

# Enter the filename for supplier and invoice 
supp_filename = "suppliernames (1).txt"
in_filename = "invoice (1).txt"

joint_prob_mat, prob_word, set_wordlist = supp_extract(supp_filename)
s_name = invoice_extract(in_filename, joint_prob_mat, prob_word, set_wordlist)
print s_name



       
