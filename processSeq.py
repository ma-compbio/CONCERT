#-*- coding: utf-8 -*-  
import sys
import random
import numpy as np
import pandas as pd
import utility_1
import h5py
import json

eps=1e-12
def countCG(strs):
	strs = strs.upper()
	return float((strs.count("C")+strs.count("G")))/(len(strs))

def countCG_N(strs):
	strs = strs.upper()
	return float((strs.count("C")+strs.count("G")))/(len(strs)-strs.count("N")+eps)

def countCG_skew(strs):
	strs = strs.upper()
	num1, num2 = strs.count("G"), strs.count("C")
	return float((num1-num2))/(num1+num2+eps)

def one_hot_encoding(seq, seq_len):
	# seq_len = len(seq)
	vec1 = np.zeros((4,seq_len))
	cnt = 0

	for i in range(0,seq_len):
		print(i)
		if seq[i]=='A':
			vec1[0,i] = 1
		elif seq[i]=='G':
			vec1[1,i] = 1
		elif seq[i]=='C':
			vec1[2,i] = 1
		elif seq[i]=='T':
			vec1[3,i] = 1
		else:
			pass

	return np.int64(vec1)

def index_encoding(seq, seq_len, seq_dict):
	# seq_len = len(seq)
	vec1 = np.zeros(seq_len)

	for i in range(0,seq_len):
		vec1[i] = seq_dict[seq[i]]

	return np.int64(vec1)

# Read sequences as strings ("N" retained)
def getString(fileStr):
	file = open(fileStr, 'r')
	gen_seq = ""
	lines = file.readlines()
	for line in lines:
		line = line.strip()
		gen_seq += line

	gen_seq = gen_seq.upper()
	return gen_seq

# Read sequences of format fasta ("N" removed)
def getStringforUnlabel(fileStr):
	file = open(fileStr, 'r')
	gen_seq = ""
	lines = file.readlines()
	for line in lines:
		if(line[0] == ">"):
			continue
		else:
			line = line.strip()
			gen_seq += line

	gen_seq = gen_seq.upper()
	gen_seq = gen_seq.replace("N", "")

	return gen_seq

def get_reverse_str(str):
	str = str.upper()
	str_new=""
	for i in range(len(str)):
		if(str[i]=="T"):
			str_new+="A"
		elif(str[i]=="A"):
			str_new+="T"
		elif(str[i]=="G"):
			str_new+="C"
		elif(str[i]=="C"):
			str_new+="G"
		else:
			str_new+=str[i]
	return str_new

# Get sequence of 2K+1 centered at pos
def getSubSeq(str, pos, K):
	n = len(str)
	l = pos - K
	r = pos + K + 1
	if l > r or l < 0 or r > n - 1:
		return 0

	elif "N" in str[l:r]:
		return 0

	return str[l:r]

# Get sequence of 2K+1 centered at pos
def getSubSeq2(str, pos, K):
	n = len(str)
	l = max(0, pos - K)
	r = min(n - 1, pos + K + 1)
	if l > r:
		print(l, pos, r)
		print("left pointer is bigger than right one")
		return 0

	return str[l:pos]+" "+str[pos]+" "+str[pos+1:r]

# Convert DNA to sentences with overlapping window of size K
def DNA2Sentence(dna, K):

	sentence = ""
	length = len(dna)

	for i in range(length - K + 1):
		sentence += dna[i: i + K] + " "

	# remove spaces
	sentence = sentence[0 : len(sentence) - 1]
	return sentence

# Convert DNA to sentences with overlapping window of size K in reverse direction
def DNA2SentenceReverse(dna, K):

	sentence = ""
	length = len(dna)

	for i in range(length - K + 1):
		j = length - K - i
		sentence += dna[j: j + K] + " "

	# remove spaces
	sentence = sentence[0 : len(sentence) - 1]
	return sentence

def reverse(s): 
	str = "" 
	for i in s: 
		str = i + str
	return str
	
# Convert DNA to sentences with overlapping window of size K in reverse direction
def DNA2SentenceReverse_1(dna, K):

	sentence = ""
	length = len(dna)
	dna = reverse(dna)

	for i in range(length - K + 1):
		sentence += dna[i: i + K] + " "

	# remove spaces
	sentence = sentence[0 : len(sentence) - 1]
	return sentence

# Convert DNA to sentences with non-overlapping window of size K
def DNA2SentenceJump(dna, K,step):
	sentence = ""
	length = len(dna)

	i=0
	while i <= length - K:
		sentence += dna[i: i + K] + " "
		i += step
	return sentence

# Convert DNA to sentences with non-overlapping window of size K in reverse direction
def DNA2SentenceJumpReverse(dna, K,step):
	sentence = ""
	length = len(dna)

	i=0
	while j <= length - K:
		i = length - K - j
		sentence += dna[i: i + K] + " "
		j += step
	return sentence

def gen_Seq(Range):
	print ("Generating Seq...")
	table = pd.read_table(PATH1+"prep_data.txt",sep = "\t")
	print (len(table))
	table.drop_duplicates()
	print (len(table))
	label_file = open(PATH1+"LabelSeq", "w")

	total = len(table)

	list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", \
			"chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", \
			"chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY","chrM"]

	number_positive = 0
	dict_pos={}

	for i in range(total):
		
		if (number_positive % 100 == 0) and (number_positive != 0):
			print ("number of seq: %d of %d\r" %(number_positive,total),end = "")
			sys.stdout.flush()

		chromosome = table["chromosome"][i]
		if chromosome in dict_pos.keys():
			strs = dict_pos[chromosome]
		else:
			strs = processSeq.getString(ROOT_PATH1+"Chromosome_38/" + str(chromosome) + ".fa")
			dict_pos[chromosome] = strs

		bias = 7
		start = int(table["start"][i] - 1 - Range + bias)
		end = start + 23 + Range*2
		
		strand = table["strand"][i]
		
		edstrs1 = strs[start : end]

		if strand == "-":
			edstrs1 = edstrs1[::-1]
			edstrs1 = processSeq.get_reverse_str(edstrs1)
		
		if "N" in edstrs1:
			table = table.drop(i)
			continue

		outstr = "%s\n"%(edstrs1)
		label_file.write(outstr)
		number_positive += 1
	table.to_csv(PATH1+"prep_data.txt",sep = "\t",index = False)

def get_target():
	table = pd.read_table(PATH1+"prep_data.txt", sep="\t")
	print (len(table))
	table.drop_duplicates()
	print (len(table))
	target_file = open(PATH1+"TargetSeq", "w")
	for i in range(len(table)):
		target = table['target'][i].upper()
		target_file.write(target+"\n")
	target_file.close()

def prep_data():
	chrom_list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", \
		"chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", \
		"chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY","chrM"]
	tab = pd.read_table(PATH1+"casoffinder_CHANGEseq_joined.tsv",sep = '\t')
	tab  = tab[tab['chromosome'].isin(chrom_list)]
	tab['label'] = 1 - tab['reads'].isna()
	tab['end'] = tab['start'] + 23
	print (tab['chromosome'].unique())

	tab.to_csv(PATH1+"prep_data.txt",sep = "\t",index = False)

def load_file(f_name,length,vec_name):
	base_code = {
			'A': 0,
			'C': 1,
			'G': 2,
			'T': 3,
		}
	num_pairs = sum(1 for line in open(f_name))
	# number of sample pairs
	num_bases = 4

	with open(f_name, 'r') as f:
		line_num = 0 # number of lines (i.e., samples) read so far
		for line in f.read().splitlines():
			if (line_num % 100 == 0) and (line_num != 0):
				print ("number of input data: %d\r" %(line_num),end= "")
				sys.stdout.flush()

			if line_num == 0:
				# allocate space for output
				seg_length = length # number of bases per sample
				Xs_seq1 = np.zeros((num_pairs, num_bases, seg_length))
				

				for start in range(len(line)):
					if line[start] in base_code:
						print (start)
						break

			base_num = 0
			
			for x in line[start:start+length]:
				if x != "N":
					Xs_seq1[line_num, base_code[x], base_num] = 1
				base_num += 1
			line_num += 1
	X = Xs_seq1
	np.save("../%s" %(vec_name),X)

def kmer_dict(K):

	vec1 = ['A','G','C','T']
	vec2 = vec1.copy() # kmer dict
	vec3 = []
	num1 = len(vec1)
	for k1 in range(1,K):
		for character in vec1:
			for temp1 in vec2:
				seq1 = character+temp1
				vec3.append(seq1)
		vec2 = vec3.copy()
		vec3 = []

	return vec2

def kmer_counting(seq, K, kmer_dict1):

	len1 = len(kmer_dict1)
	vec = np.zeros((len1),dtype=np.float32)
	len2 = len(seq)-K+1
	cnt = 0
	for kmer in kmer_dict1:
		num1 = seq.count(kmer)
		vec[cnt] = num1
		cnt = cnt+1

	vec = vec*1.0/len2

	return vec

def align_region(species_id):

	col1, col2, col3 = '%s.chrom'%(species_id), '%s.start'%(species_id), '%s.stop'%(species_id)

def load_seq_kmer(species_id, file1, filename2, K, kmer_dict1):

	# file1 = pd.read_csv(filename1,sep='\t')
	col1, col2, col3 = '%s.chrom'%(species_id), '%s.start'%(species_id), '%s.stop'%(species_id)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])

	num1 = len(chrom)

	file = open(filename2, 'r')
	# serial_list, line_list = [], []
	serial_list = -np.ones((num1,2))
	f_list = np.zeros((num1,feature_dim))
	lines = file.readlines()
	num_line = len(lines)
	cnt = -1
	flag = 0
	print(num_line,num1)
	# temp1 = int(num_line/2)

	for line in lines:
		if(line[0]==">"):
			# continue
			# line: >chr1:5-10
			cnt = cnt + 1
			str1 = line[1:]
			temp1 = str1.split(':')
			t_chrom = temp1[0]
			temp2 = temp1[1].split('-')
			t_start, t_stop = int(temp2[0]), int(temp2[1])
			chrom1, start1, stop1, serial1 = chrom[cnt], start[cnt], stop[cnt], serial[cnt]
			if (chrom1==t_chrom) and (start1==t_start) and (stop1==t_stop):
				flag = 1
			else:
				b = np.where((chrom==t_chrom)&(start==t_start)&(stop==t_stop))[0]
				if len(b)>0:
					cnt = b[0]
					flag = 1
		else:
			if flag == 1:
				line = line.strip().upper()
				vec = kmer_counting(line,K,kmer_dict1)
				# line_list.append(line)
				# f_list.append(vec)
				# line_list.append(line)
				# N_list.append(line.count('N'))
				flag = 0
				serial_list[cnt,0], serial_list[cnt,1] = serial[cnt], line.count('N')
				f_list[cnt] = vec

	filename1 = '%s.vec'%(species_id)
	np.save(filename1,(serial_list,f_list))

	return serial_list, f_list

# load the annotation file and the sequence feature file
# return kmer feature: num_samples*feature_dim
# return one-hot encoding feature: num_samples*4*feature_dim
def load_seq_1_ori(species_id, file1, filename2, K, kmer_dict1):

	# file1 = pd.read_csv(filename1,sep='\t')
	col1, col2, col3 = '%s.chrom'%(species_id), '%s.start'%(species_id), '%s.stop'%(species_id)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])

	num1 = len(chrom)
	len1 = stop-start
	seq_len = int(np.median(len1))

	file = open(filename2, 'r')
	# serial_list, line_list = [], []
	serial_list = -np.ones((num1,2))
	feature_dim = len(kmer_dict1)
	f_list = np.zeros((num1,feature_dim))
	f_mtx = np.zeros((num1,4,seq_len))

	lines = file.readlines()
	num_line = len(lines)
	cnt = -1
	flag = 0
	print(num_line,num1)
	# temp1 = int(num_line/2)

	i = 0
	for line in lines:
		if(line[0]==">"):
			# continue
			# line: >chr1:5-10
			print(cnt)
			cnt = cnt + 1
			str1 = line[1:]
			temp1 = str1.split(':')
			t_chrom = temp1[0]
			temp2 = temp1[1].split('-')
			t_start, t_stop = int(temp2[0]), int(temp2[1])
			chrom1, start1, stop1, serial1 = chrom[cnt], start[cnt], stop[cnt], serial[cnt]
			if (chrom1==t_chrom) and (start1==t_start) and (stop1==t_stop):
				flag = 1
			else:
				b = np.where((chrom==t_chrom)&(start==t_start)&(stop==t_stop))[0]
				if len(b)>0:
					cnt = b[0]
					flag = 1
		else:
			if flag == 1:
				line = line.strip().upper()
				vec = kmer_counting(line,K,kmer_dict1)
				# line_list.append(line)
				# f_list.append(vec)
				flag = 0
				serial_list[cnt,0], serial_list[cnt,1] = serial[cnt], line.count('N')
				f_list[cnt] = vec
				f_mtx[cnt] = one_hot_encoding(line, seq_len)

				i += 1
				if i % 100 == 0:
					print("%d of %d\r" %(i,num1), end = "")
					sys.stdout.flush()

	b = np.where(serial_list[:,0]>=0)[0]
	serial_list, f_list, f_mtx, label, group_label = serial_list[b], f_list[b], f_mtx[b], label[b], group_label[b]
	# filename1 = '%s.vec'%(species_id)
	# np.save(filename1,(serial_list,f_list))

	return serial_list, f_list, f_mtx, label, group_label, signal

# load feature
def load_seq_altfeature_1(species_id, file1, filename2, output_filename):

	# file1 = pd.read_csv(filename1,sep='\t')
	col1, col2, col3 = '%s.chrom'%(species_id), '%s.start'%(species_id), '%s.stop'%(species_id)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])

	num1 = len(chrom)
	len1 = stop-start
	seq_len = int(np.median(len1))

	file = open(filename2, 'r')
	# serial_list, line_list = [], []
	serial_list = -np.ones((num1,3))
	feature_dim = 3
	# num1 = 2000
	f_list = np.zeros((num1,feature_dim))
	# f_mtx = np.zeros((num1,4,seq_len))

	lines = file.readlines()
	num_line = len(lines)
	cnt = -1
	flag = 0
	print(num_line,num1)
	# temp1 = int(num_line/2)

	i = 0
	serial_vec, seq_vec = [], []
	for line in lines:
		if(line[0]==">"):
			# continue
			# line: >chr1:5-10
			# print(cnt)
			cnt = cnt + 1
			str1 = line[1:]
			temp1 = str1.split(':')
			t_chrom = temp1[0]
			temp2 = temp1[1].split('-')
			t_start, t_stop = int(temp2[0]), int(temp2[1])
			chrom1, start1, stop1, serial1 = chrom[cnt], start[cnt], stop[cnt], serial[cnt]
			if (chrom1==t_chrom) and (start1==t_start) and (stop1==t_stop):
				flag = 1
			else:
				b = np.where((chrom==t_chrom)&(start==t_start)&(stop==t_stop))[0]
				if len(b)>0:
					cnt = b[0]
					flag = 1
		else:
			if flag == 1:
				line = line.strip().upper()
				# vec = kmer_counting(line,K,kmer_dict1)

				serial_vec.append([cnt,serial[cnt]])				
				seq_vec.append(line)
				GC_profile = countCG(line)
				GC_profile1 = countCG_N(line)
				GC_skew = countCG_skew(line)
				vec = [GC_profile,GC_profile1,GC_skew]

				# line_list.append(line)
				# f_list.append(vec)
				flag = 0
				serial_list[cnt,0], serial_list[cnt,1], serial_list[cnt,2] = serial[cnt], len(line), line.count('N')
				f_list[cnt] = vec
				# f_mtx[cnt] = one_hot_encoding(line, seq_len)

				i += 1
				if i % 1000 == 0:
					print("%d of %d\r" %(i,num1), end = "")
					sys.stdout.flush()

				# if cnt>1000:
				#  	break

	# b = np.where(serial_list[:,0]>=0)[0]
	# serial_list, f_list, f_mtx, label, group_label = serial_list[b], f_list[b], f_mtx[b], label[b], group_label[b]
	# filename1 = '%s.vec'%(species_id)
	# np.save(filename1,(serial_list,f_list))

	serial_vec = np.asarray(serial_vec)
	fields = ['index','serial','seq']
	data1 = pd.DataFrame(columns=fields)
	data1[fields[0]], data1[fields[1]] = serial_vec[:,0], serial_vec[:,1]
	data1[fields[2]] = seq_vec
	# data1.to_csv('test_seq.txt',index=False,sep='\t')
	data1.to_csv(output_filename,index=False,sep='\t')
	
	return serial_list, f_list, label, group_label, signal

# feature 1: GC profile
# feature 2: GC skew
# def load_seq_altfeature(filename2, K, kmer_dict1, sel_idx):
def load_seq_altfeature(filename2, sel_idx):

	file2 = pd.read_csv(filename2,sep='\t')
	seq = np.asarray(file2['seq'])

	if len(sel_idx)>0:
		seq = seq[sel_idx]
	num1 = len(seq)
	print("number of sequences %d"%(num1))

	feature_dim = 3
	f_list = np.zeros((num1,feature_dim))

	for i in range(0,num1):
		sequence = seq[i].strip()
		GC_profile = countCG(sequence)
		GC_profile1 = countCG_N(sequence)
		GC_skew = countCG_skew(sequence)
		f_list[i] = [GC_profile,GC_profile1,GC_skew]

		if i % 1000 == 0:
			print("%d of %d\r" %(i,num1), end = "")
			sys.stdout.flush()

	return f_list

def load_seq_1(species_id, file1, filename2, K, kmer_dict1, output_filename):

	# file1 = pd.read_csv(filename1,sep='\t')
	col1, col2, col3 = '%s.chrom'%(species_id), '%s.start'%(species_id), '%s.stop'%(species_id)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])

	num1 = len(chrom)
	len1 = stop-start
	seq_len = int(np.median(len1))

	file = open(filename2, 'r')
	# serial_list, line_list = [], []
	serial_list = -np.ones((num1,2))
	feature_dim = len(kmer_dict1)
	f_list = np.zeros((num1,feature_dim))
	f_mtx = np.zeros((num1,4,seq_len),dtype=np.float32)

	lines = file.readlines()
	num_line = len(lines)
	cnt = -1
	flag = 0
	print(num_line,num1)
	# temp1 = int(num_line/2)

	i = 0
	serial_vec, seq_vec = [], []
	for line in lines:
		if(line[0]==">"):
			# continue
			# line: >chr1:5-10
			# print(cnt)
			cnt = cnt + 1
			str1 = line[1:]
			temp1 = str1.split(':')
			t_chrom = temp1[0]
			temp2 = temp1[1].split('-')
			t_start, t_stop = int(temp2[0]), int(temp2[1])
			chrom1, start1, stop1, serial1 = chrom[cnt], start[cnt], stop[cnt], serial[cnt]
			if (chrom1==t_chrom) and (start1==t_start) and (stop1==t_stop):
				flag = 1
			else:
				b = np.where((chrom==t_chrom)&(start==t_start)&(stop==t_stop))[0]
				if len(b)>0:
					cnt = b[0]
					flag = 1
		else:
			if flag == 1:
				line = line.strip().upper()
				vec = kmer_counting(line,K,kmer_dict1)

				serial_vec.append([cnt,serial[cnt]])				
				seq_vec.append(line)

				# line_list.append(line)
				# f_list.append(vec)
				flag = 0
				serial_list[cnt,0], serial_list[cnt,1] = serial[cnt], line.count('N')
				f_list[cnt] = vec
				# f_mtx[cnt] = one_hot_encoding(line, seq_len)

				i += 1
				if i % 1000 == 0:
					print("%d of %d\r" %(i,num1), end = "")
					sys.stdout.flush()

				# if cnt>1000:
				#  	break

	# b = np.where(serial_list[:,0]>=0)[0]
	# serial_list, f_list, f_mtx, label, group_label = serial_list[b], f_list[b], f_mtx[b], label[b], group_label[b]
	# filename1 = '%s.vec'%(species_id)
	# np.save(filename1,(serial_list,f_list))

	serial_vec = np.asarray(serial_vec)
	fields = ['index','serial','seq']
	data1 = pd.DataFrame(columns=fields)
	data1[fields[0]], data1[fields[1]] = serial_vec[:,0], serial_vec[:,1]
	data1[fields[2]] = seq_vec
	# data1.to_csv('test_seq.txt',index=False,sep='\t')
	data1.to_csv(output_filename,index=False,sep='\t')
	
	return serial_list, f_list, f_mtx, label, group_label, signal

def load_seq_1_1(species_id, filename1, header, filename2, output_filename):

	file1 = pd.read_csv(filename1,sep='\t',header=header)
	colnames = list(file1)
	col1, col2, col3, col4 = colnames[0], colnames[1], colnames[2], colnames[3]
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1[col4])

	num1 = len(chrom)
	len1 = stop-start
	seq_len = int(np.median(len1))

	file = open(filename2, 'r')
	# serial_list, line_list = [], []
	serial_list = -np.ones((num1,2))
	# feature_dim = len(kmer_dict1)
	# f_list = np.zeros((num1,feature_dim))
	# f_mtx = np.zeros((num1,4,seq_len))

	lines = file.readlines()
	num_line = len(lines)
	cnt = -1
	flag = 0
	print(num_line,num1)

	i = 0
	serial_vec, seq_vec = [], []
	for line in lines:
		if(line[0]==">"):
			cnt = cnt + 1
			str1 = line[1:]
			temp1 = str1.split(':')
			t_chrom = temp1[0]
			temp2 = temp1[1].split('-')
			t_start, t_stop = int(temp2[0]), int(temp2[1])
			chrom1, start1, stop1, serial1 = chrom[cnt], start[cnt], stop[cnt], serial[cnt]
			if (chrom1==t_chrom) and (start1==t_start) and (stop1==t_stop):
				flag = 1
			else:
				b = np.where((chrom==t_chrom)&(start==t_start)&(stop==t_stop))[0]
				if len(b)>0:
					cnt = b[0]
					flag = 1
		else:
			if flag == 1:
				line = line.strip().upper()
				# vec = kmer_counting(line,K,kmer_dict1)

				serial_vec.append([cnt,serial[cnt]])				
				seq_vec.append(line)

				flag = 0
				serial_list[cnt,0], serial_list[cnt,1] = serial[cnt], line.count('N')
				# f_list[cnt] = vec

				i += 1
				if i % 1000 == 0:
					print("%d of %d\r" %(i,num1), end = "")
					sys.stdout.flush()

	serial_vec = np.asarray(serial_vec)
	fields = ['index','serial','seq']
	data1 = pd.DataFrame(columns=fields)
	data1[fields[0]], data1[fields[1]] = serial_vec[:,0], serial_vec[:,1]
	data1[fields[2]] = seq_vec
	# data1.to_csv('test_seq.txt',index=False,sep='\t')
	data1.to_csv(output_filename,index=False,sep='\t')
	
	# return serial_list, f_list
	return True

def load_seq_1a(species_id, file1, filename2, K, kmer_dict1, output_filename):

	# file1 = pd.read_csv(filename1,sep='\t')
	colnames = list(file1)
	col1, col2, col3, col4 = colnames[0], colnames[1], colnames[2], colnames[3]
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1[col4])

	num1 = len(chrom)
	len1 = stop-start
	seq_len = int(np.median(len1))

	file = open(filename2, 'r')
	# serial_list, line_list = [], []
	serial_list = -np.ones((num1,2))
	feature_dim = len(kmer_dict1)
	f_list = np.zeros((num1,feature_dim))
	f_mtx = np.zeros((num1,4,seq_len))

	lines = file.readlines()
	num_line = len(lines)
	cnt = -1
	flag = 0
	print(num_line,num1)
	# temp1 = int(num_line/2)

	i = 0
	serial_vec, seq_vec = [], []
	for line in lines:
		if(line[0]==">"):
			# continue
			# line: >chr1:5-10
			# print(cnt)
			cnt = cnt + 1
			str1 = line[1:]
			temp1 = str1.split(':')
			t_chrom = temp1[0]
			temp2 = temp1[1].split('-')
			t_start, t_stop = int(temp2[0]), int(temp2[1])
			chrom1, start1, stop1, serial1 = chrom[cnt], start[cnt], stop[cnt], serial[cnt]
			if (chrom1==t_chrom) and (start1==t_start) and (stop1==t_stop):
				flag = 1
			else:
				b = np.where((chrom==t_chrom)&(start==t_start)&(stop==t_stop))[0]
				if len(b)>0:
					cnt = b[0]
					flag = 1
		else:
			if flag == 1:
				line = line.strip().upper()
				vec = kmer_counting(line,K,kmer_dict1)

				serial_vec.append([cnt,serial[cnt]])				
				seq_vec.append(line)

				# line_list.append(line)
				# f_list.append(vec)
				flag = 0
				serial_list[cnt,0], serial_list[cnt,1] = serial[cnt], line.count('N')
				f_list[cnt] = vec
				# f_mtx[cnt] = one_hot_encoding(line, seq_len)

				i += 1
				if i % 1000 == 0:
					print("%d of %d\r" %(i,num1), end = "")
					sys.stdout.flush()

				# if cnt>1000:
				#  	break

	# b = np.where(serial_list[:,0]>=0)[0]
	# serial_list, f_list, f_mtx, label, group_label = serial_list[b], f_list[b], f_mtx[b], label[b], group_label[b]
	# filename1 = '%s.vec'%(species_id)
	# np.save(filename1,(serial_list,f_list))

	serial_vec = np.asarray(serial_vec)
	fields = ['index','serial','seq']
	data1 = pd.DataFrame(columns=fields)
	data1[fields[0]], data1[fields[1]] = serial_vec[:,0], serial_vec[:,1]
	data1[fields[2]] = seq_vec
	# data1.to_csv('test_seq.txt',index=False,sep='\t')
	data1.to_csv(output_filename,index=False,sep='\t')
	
	return serial_list, f_list

def generate_serial(chrom,start,stop):

	# chrom_vec = np.sort(np.unique(chrom))
	# print(chrom_vec)
	chrom_vec = []
	for i in range(1,23):
		chrom_vec.append('chr%d'%(i))
	chrom_vec += ['chrX']
	chrom_vec += ['chrY']
	print(chrom_vec)
	print(chrom)
	print(len(chrom))
	
	filename1 = '/volume01/yy3/seq_data/genome/hg38.chrom.sizes'
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	ref_chrom, chrom_size = np.asarray(data1[0]), np.asarray(data1[1])
	serial_start = 0
	serial_vec = []
	bin_size = stop[1]-start[1]
	print(bin_size)
	for chrom_id in chrom_vec:
		b1 = np.where(ref_chrom==chrom_id)[0]
		t_size = chrom_size[b1[0]]
		b2 = np.where(chrom==chrom_id)[0]
		if len(b1)>0:
			size1 = int(np.ceil(t_size*1.0/bin_size))
			serial = np.int64(start[b2]/bin_size)+serial_start
			serial_vec.extend(serial)
			print(chrom_id,b2,len(serial),serial_start,size1)
			serial_start = serial_start+size1
		else:
			print("error!")
			return

	return np.asarray(serial_vec)

def generate_serial_start():

	# chrom_vec = np.sort(np.unique(chrom))
	# print(chrom_vec)
	chrom_vec = []
	for i in range(1,23):
		chrom_vec.append('chr%d'%(i))
	chrom_vec += ['chrX']
	chrom_vec += ['chrY']
	print(chrom_vec)
	print(chrom)
	print(len(chrom))
	
	filename1 = '/volume01/yy3/seq_data/genome/hg38.chrom.sizes'
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	ref_chrom, chrom_size = np.asarray(data1[0]), np.asarray(data1[1])
	serial_start = 0
	serial_vec = []
	bin_size = stop[1]-start[1]
	print(bin_size)
	start_vec = dict()
	for chrom_id in chrom_vec:
		start_vec[chrom_id] = serial_start
		b1 = np.where(ref_chrom==chrom_id)[0]
		t_size = chrom_size[b1[0]]
		b2 = np.where(chrom==chrom_id)[0]
		if len(b1)>0:
			size1 = int(np.ceil(t_size*1.0/bin_size))
			serial = np.int64(start[b2]/bin_size)+serial_start
			serial_vec.extend(serial)
			print(chrom_id,b2,len(serial),serial_start,size1)
			serial_start = serial_start+size1
		else:
			print("error!")
			return

	return np.asarray(serial_vec), start_vec

def generate_serial_single(chrom,start,stop):

	# chrom_vec = np.sort(np.unique(chrom))
	# print(chrom_vec)
	serial_vec, start_vec = generate_serial_start()
	chrom_vec = []
	for i in range(1,23):
		chrom_vec.append('chr%d'%(i))
	chrom_vec += ['chrX']
	chrom_vec += ['chrY']
	print(chrom_vec)
	print(chrom)
	print(len(chrom))
	
	filename1 = '/volume01/yy3/seq_data/genome/hg38.chrom.sizes'
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	ref_chrom, chrom_size = np.asarray(data1[0]), np.asarray(data1[1])
	serial_start = 0
	serial_vec = []
	bin_size = stop[1]-start[1]
	print(bin_size)
	start_vec = dict()
	for chrom_id in chrom_vec:
		start_vec[chrom_id] = serial_start
		b1 = np.where(ref_chrom==chrom_id)[0]
		t_size = chrom_size[b1[0]]
		b2 = np.where(chrom==chrom_id)[0]
		if len(b1)>0:
			size1 = int(np.ceil(t_size*1.0/bin_size))
			serial = np.int64(start[b2]/bin_size)+serial_start
			serial_vec.extend(serial)
			print(chrom_id,b2,len(serial),serial_start,size1)
			serial_start = serial_start+size1
		else:
			print("error!")
			return

	return np.asarray(serial_vec), start_vec

def load_seq_2a(filename1, filename2, output_filename):

	file1 = pd.read_csv(filename1,sep='\t',header=None)
	colnames = list(file1)

	# col1, col2, col3, col4 = colnames[0], colnames[1], colnames[2], colnames[3]
	# chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1[col4])

	col1, col2, col3 = colnames[0], colnames[1], colnames[2]
	chrom, start, stop = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3])
	bin_size = stop[1]-start[1]
	# serial = generate_serial(chrom,start,stop)
	serial = utility_1.generate_serial(chrom,start,stop)
	
	num1 = len(chrom)
	len1 = stop-start
	seq_len = int(np.median(len1))
	print(num1, seq_len, len(serial))

	file = open(filename2, 'r')
	# serial_list, line_list = [], []
	serial_list = -np.ones((num1,2))
	# feature_dim = len(kmer_dict1)
	# f_list = np.zeros((num1,feature_dim))
	# f_mtx = np.zeros((num1,4,seq_len))

	lines = file.readlines()
	num_line = len(lines)
	cnt = -1
	flag = 0
	print(num_line,num1)
	# temp1 = int(num_line/2)

	i = 0
	serial_vec, seq_vec = [], []
	for line in lines:
		if(line[0]==">"):
			# continue
			# line: >chr1:5-10
			# print(cnt)
			cnt = cnt + 1
			str1 = line[1:]
			temp1 = str1.split(':')
			t_chrom = temp1[0]
			temp2 = temp1[1].split('-')
			t_start, t_stop = int(temp2[0]), int(temp2[1])
			chrom1, start1, stop1, serial1 = chrom[cnt], start[cnt], stop[cnt], serial[cnt]
			if (chrom1==t_chrom) and (start1==t_start) and (stop1==t_stop):
				flag = 1
			else:
				b = np.where((chrom==t_chrom)&(start==t_start)&(stop==t_stop))[0]
				if len(b)>0:
					cnt = b[0]
					flag = 1
		else:
			if flag == 1:
				line = line.strip().upper()
				# vec = kmer_counting(line,K,kmer_dict1)

				serial_vec.append([cnt,serial[cnt]])				
				seq_vec.append(line)
				print(chrom1,start1,stop1,t_chrom,t_start,t_stop,cnt,line[0:20])

				# line_list.append(line)
				# f_list.append(vec)
				flag = 0
				serial_list[cnt,0], serial_list[cnt,1] = serial[cnt], line.count('N')
				# f_list[cnt] = vec
				# f_mtx[cnt] = one_hot_encoding(line, seq_len)

				i += 1
				if i % 1000 == 0:
					print("%d of %d\r" %(i,num1), end = "")
					# sys.stdout.flush()

				# if cnt>10:
				#  	break

	# b = np.where(serial_list[:,0]>=0)[0]
	# serial_list, f_list, f_mtx, label, group_label = serial_list[b], f_list[b], f_mtx[b], label[b], group_label[b]
	# filename1 = '%s.vec'%(species_id)
	# np.save(filename1,(serial_list,f_list))

	serial_vec = np.asarray(serial_vec)
	fields = ['index','serial','seq']
	data1 = pd.DataFrame(columns=fields)
	data1[fields[0]], data1[fields[1]] = serial_vec[:,0], serial_vec[:,1]
	data1[fields[2]] = seq_vec
	# data1.to_csv('test_seq.txt',index=False,sep='\t')
	data1.to_csv(output_filename,index=False,sep='\t')

	# np.savetxt('serial_list.txt', serial_list, fmt='%d', delimiter='\t')
	# np.savetxt('serial1.txt', serial, fmt='%d', delimiter='\t')
	
	return serial_list, serial

def load_signal_1(species_id, filename1):

	file1 = pd.read_csv(filename1,sep='\t')
	col1, col2, col3 = '%s.chrom'%(species_id), '%s.start'%(species_id), '%s.stop'%(species_id)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])

	return chrom, label, group_label, signal

def load_seq_2(filename1, K, kmer_dict1, sel_idx=[]):

	file1 = pd.read_csv(filename1,sep='\t')
	seq = np.asarray(file1['seq'])
	serial = np.asarray(file1['serial'])

	if len(sel_idx)>0:
		seq = seq[sel_idx]
	num1 = len(seq)
	print("number of sequences %d"%(num1))

	feature_dim = len(kmer_dict1)
	seq_len = len(seq[0])

	f_list = np.zeros((num1,feature_dim))
	# f_mtx = np.zeros((num1,4,seq_len))
	f_mtx = []

	for i in range(0,num1):
		sequence = seq[i]
		vec = kmer_counting(sequence,K,kmer_dict1)
		f_list[i] = vec
		# f_mtx[i] = one_hot_encoding(sequence, seq_len)

		if i % 1000 == 0:
			print("%d of %d\r" %(i,num1), end = "")
			sys.stdout.flush()
		# if i>=1000:
		# 	break

	return f_list, f_mtx, serial

def load_seq_2_kmer(seq1, K, kmer_dict1, chrom_id, sel_idx):

	# file2 = pd.read_csv(filename2,sep='\t')
	# seq = np.asarray(file2['seq'])

	print(len(seq1))
	if len(sel_idx)>0:
		seq = seq1[sel_idx]
	else:
		seq = seq1
	num1 = len(seq)
	print("number of sequences %d"%(num1))

	feature_dim = len(kmer_dict1)
	# seq_len = len(seq[0])

	f_list = np.zeros((num1,feature_dim))
	# f_mtx = np.zeros((num1,4,seq_len))

	for i in range(0,num1):
		sequence = seq[i]
		vec = kmer_counting(sequence,K,kmer_dict1)
		f_list[i] = vec
		# f_mtx[i] = one_hot_encoding(sequence, seq_len)

		if i % 1000 == 0:
			print("%s %d of %d\r" %(chrom_id,i,num1), end = "")
			sys.stdout.flush()

	return f_list

def load_seq_2_kmer1(seq1, serial1, K, kmer_dict1, chrom_id, sel_idx, queue1, filename_prefix, save_mode=1):

	# file2 = pd.read_csv(filename2,sep='\t')
	# seq = np.asarray(file2['seq'])

	print(len(seq1))
	if len(sel_idx)>0:
		seq = seq1[sel_idx]
		serial2 = serial1[sel_idx]
	else:
		seq = seq1
		serial2 = serial1

	num1 = len(serial2)
	print("number of sequences %d"%(num1))

	feature_dim = len(kmer_dict1)
	# seq_len = len(seq[0])
	f_list = np.zeros((len(serial2),feature_dim))
	# f_mtx = np.zeros((num1,4,seq_len))

	for i in range(0,num1):
		sequence = seq[i].upper()
		vec = kmer_counting(sequence,K,kmer_dict1)
		f_list[i] = vec
		# f_mtx[i] = one_hot_encoding(sequence, seq_len)

		if i % 1000 == 0:
			print("%s %d of %d\r" %(chrom_id,i,num1), end = "")
			sys.stdout.flush()
		# if i>100:
		#	break

	if save_mode==1:
		output_filename = '%s_kmer%d_%s.h5'%(filename_prefix,K,chrom_id)
		with h5py.File(output_filename,'w') as fid:
			fid.create_dataset("serial", data=serial2, compression="gzip")
			fid.create_dataset("vec", data=f_list, compression="gzip")
		queue1.put((chrom_id, sel_idx, output_filename))
	else:
		queue1.put((chrom_id, sel_idx, serial2, f_list))

	return True

def load_seq_2_kmer1_subregion(seq1, serial1, K, kmer_dict1, chrom_id, sel_idx, region_size,
								queue1, filename_prefix, save_mode=1):

	# file2 = pd.read_csv(filename2,sep='\t')
	# seq = np.asarray(file2['seq'])

	print(len(seq1))
	if len(sel_idx)>0:
		seq = seq1[sel_idx]
		serial2 = serial1[sel_idx]
	else:
		seq = seq1
		serial2 = serial1

	num1 = len(serial2)
	print("number of sequences %d"%(num1))

	feature_dim = len(kmer_dict1)
	region_num = int(np.ceil(len(seq[0])/region_size))
	# seq_len = len(seq[0])
	f_list = np.zeros((len(serial2),region_num,feature_dim),dtype=np.float32)
	# f_mtx = np.zeros((num1,4,seq_len))

	for i in range(0,num1):
		sequence = seq[i].upper()
		vec = kmer_counting_subregion(sequence,K,kmer_dict1,feature_dim,region_size,region_num)
		f_list[i] = vec
		# f_mtx[i] = one_hot_encoding(sequence, seq_len)

		if i % 1000 == 0:
			print("%s %d of %d\r" %(chrom_id,i,num1), end = "")
			sys.stdout.flush()
		# if i>100:
		#	break

	if save_mode==1:
		output_filename = '%s_kmer%d_%s.h5'%(filename_prefix,K,chrom_id)
		with h5py.File(output_filename,'w') as fid:
			fid.create_dataset("serial", data=serial2, compression="gzip")
			fid.create_dataset("vec", data=f_list, compression="gzip")
		queue1.put((chrom_id, sel_idx, output_filename))
	else:
		queue1.put((chrom_id, sel_idx, serial2, f_list))

	return True
	
def load_seq_list(species_vec,filename1,K):

	file1 = pd.read_csv(filename1,sep='\t')
	col1, col2, col3 = '%s.chrom'%(species_id), '%s.start'%(species_id), '%s.stop'%(species_id)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['state'])
	group_label = np.asarray(file1['group'])
	num1 = len(chrom)

	kmer_dict1 = kmer_dict(K)
	filename1 = 'test1.txt'
	cnt = 0
	t_serial_vec, t_serial_vec1 = [], []
	t_feature_vec = []
	for species_id in species_vec:
		print(species_id)
		filename2 = '%s.seq.txt'%(species_id)
		t_serial, t_feature = load_seq(species_id, file1, filename2, K, kmer_dict1)
		temp1 = t_serial[:,0]
		temp2 = np.zeros(len(temp1))
		b = np.where(temp1>=0)[0]
		temp2[b] = 1
		t_serial_vec.append(t_serial[:,0])
		t_serial_vec1.append(temp2)
		if cnt==0:
			t_feature_mtx = t_feature
		else:
			t_feature_mtx = np.hstack((t_feature_mtx,t_feature))
		cnt = cnt+1
	
	species_num = len(species_vec)
	t_serial_vec = np.asarray(t_serial_vec).T
	t_serial_vec1 = np.asarray(t_serial_vec1).T
	t1 = np.sum(t_serial_vec1,1)
	b = np.where(t1==species_num)[0]	# the species all have features
	t_feature_mtx = t_feature_mtx[b]
	t_serial_mtx = t_serial_vec[b,0]
	t_label = label[b]

	return t_label, t_serial_mtx, t_feature_mtx

# generate bed file
def load_bedfile(species_id,ref_filename,match_filename,signal_filename):

	data1 = pd.read_csv(ref_filename,sep='\t')
	data2 = pd.read_csv(match_filename,header=None,sep='\t')
	data3 = pd.read_csv(signal_filename,header=None,sep='\t')
	
	colnames = list(data1)
	print(colnames)
	ref_chrom, ref_start, ref_stop, serial = data1[colnames[0]], data1[colnames[1]], data1[colnames[2]], np.asarray(data1[colnames[3]])
	num1 = len(ref_chrom)

	t_chrom, t_start, t_stop, t_serial = np.asarray(data2[0]), np.asarray(data2[1]), np.asarray(data2[2]), np.asarray(data2[3])
	t_chrom1, t_start1, t_stop1, signal1 = np.asarray(data3[0]), np.asarray(data3[1]), np.asarray(data3[2]), np.asarray(data3[3])

	flag_vec = np.zeros(num1)
	for i in range(0,num1):
		if (ref_chrom[i]==t_chrom1[i]) and (ref_start[i]==t_start1[i]) and (ref_stop[i]==t_stop1[i]):
			flag_vec[i] = 1

	if np.sum(flag_vec)<num1:
		print("serial not matched!")
		return False

	idx1 = mapping_Idx(t_serial,serial)
	print(idx1.shape,signal1.shape)
	fields = ['%s.chrom'%(species_id),'%s.start'%(species_id),'%s.stop'%(species_id),'serial','label','group_label','signal']
	data_2 = pd.DataFrame(columns = fields)
	data_2[fields[0]] = t_chrom[idx1]
	data_2[fields[1]] = t_start[idx1]
	data_2[fields[2]] = t_stop[idx1]
	data_2[fields[-1]] = signal1[0:num1]
	for i in range(3,6):
		data_2[fields[i]] = data1[colnames[i]]

	print("writing to file...")
	filename1 = 'estiamte_rt_%s.txt'%(species_id)
	data_2.to_csv(filename1,index=False,sep='\t')

	return True

def mapping_Idx(serial1,serial2):

	ref_serial = np.sort(serial1)
	ref_sortedIdx = np.argsort(serial1)
	ref_serial = np.int64(ref_serial)
	
	map_serial = np.sort(serial2)
	map_sortedIdx = np.argsort(serial2)
	map_serial = np.int64(map_serial)

	num1 = np.max((ref_serial[-1],map_serial[-1]))+1
	vec1 = np.zeros((num1,2))
	vec1[map_serial,0] = 1
	b = np.where(vec1[ref_serial,0]>0)[0]
	vec1[ref_serial,1] = 1
	b1 = np.where(vec1[map_serial,1]>0)[0]

	idx = ref_sortedIdx[b]
	idx1 = -np.ones(len(map_serial))
	idx1[map_sortedIdx[b1]] = idx

	return np.int64(idx1)

