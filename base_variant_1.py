
from __future__ import print_function

import string
import sys
import os
from collections import deque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import tensorflow as tf
import keras
keras.backend.image_data_format()
from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense, Reshape, Lambda, Conv1D, Flatten, MaxPooling1D, UpSampling1D, GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional, BatchNormalization, Dropout, Concatenate, Embedding, Activation, Dot, dot
from keras.models import Model, clone_model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.constraints import unitnorm
from keras_layer_normalization import LayerNormalization
tf.keras.backend.set_floatx('float32')

import sklearn as sk
from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, TSNE
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD, FastICA, NMF, MiniBatchDictionaryLearning
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import roc_curve, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score,accuracy_score,matthews_corrcoef
from scipy import stats
from scipy.stats import multivariate_normal, kurtosis, skew, pearsonr, spearmanr
import processSeq
from processSeq import load_seq_1, kmer_dict, load_signal_1, load_seq_2, load_seq_2_kmer, load_seq_altfeature
import xgboost
import pickle

import os.path
from optparse import OptionParser

import time
from timeit import default_timer as timer

import utility_1
from utility_1 import mapping_Idx

import h5py
import json

# generate sequences
# idx_sel_list: chrom, serial
# seq_list: relative positions
def generate_sequences(idx_sel_list, gap_tol=5, region_list=[]):

	chrom = idx_sel_list[:,0]
	chrom_vec = np.unique(chrom)
	chrom_vec = np.sort(chrom_vec)
	seq_list = []
	print(len(chrom),chrom_vec)
	for chrom_id in chrom_vec:
		b1 = np.where(chrom==chrom_id)[0]
		t_serial = idx_sel_list[b1,1]
		prev_serial = t_serial[0:-1]
		next_serial = t_serial[1:]
		distance = next_serial-prev_serial
		b2 = np.where(distance>gap_tol)[0]

		if len(b2)>0:
			if len(region_list)>0:
				# print('region_list',region_list,len(b2))
				b_1 = np.where(region_list[:,0]==chrom_id)[0]
				# print(b2)
				t_serial = idx_sel_list[b2,1]

				if len(b_1)>0:
					# b2 = np.setdiff1d(b2,region_list[b_1,1])
					# print(region_list,region_list[b_1,1],len(b2))

					t_id1 = utility_1.mapping_Idx(t_serial,region_list[b_1,1])
					t_id1 = t_id1[t_id1>=0]
					t_id2 = b2[t_id1]
					b2 = np.setdiff1d(b2,t_id2)

					# print(len(b2))
					# print(idx_sel_list[b2])

					# return

		# print('gap',len(b2))
		if len(b2)>0:
			t_seq = list(np.vstack((b2[0:-1]+1,b2[1:])).T)
			t_seq.insert(0,np.asarray([0,b2[0]]))
			t_seq.append(np.asarray([b2[-1]+1,len(b1)-1]))
		else:
			t_seq = [np.asarray([0,len(b1)-1])]
		# print(t_seq)
		# print(chrom_id,len(t_seq),max(distance))
		seq_list.extend(b1[np.asarray(t_seq)])

	return np.asarray(seq_list)

# select sample
def sample_select2a1(x_mtx, y, idx_sel_list, seq_list, tol=5, L=5):

	num_sample = len(idx_sel_list)
	num1 = len(seq_list)
	size1 = 2*L+1
	print(num_sample,num1,size1)
	feature_dim = x_mtx.shape[1]

	vec1_local = np.zeros((num_sample,size1),dtype=int)
	vec1_serial = np.zeros((num_sample,size1),dtype=int)
	feature_mtx = np.zeros((num_sample,size1,feature_dim),dtype=np.float32)
	signal_mtx = np.zeros((num_sample,size1))
	ref_serial = idx_sel_list[:,1]
	id_vec = np.zeros(num_sample,dtype=np.int8)

	for i in range(0,num1):
		s1, s2 = seq_list[i][0], seq_list[i][1]+1
		serial = ref_serial[s1:s2]
		id_vec[s1:s2] = 1
		# print('start stop',s1,s2,serial)
		num2 = len(serial)
		t1 = np.outer(list(range(s1,s2)),np.ones(size1))
		t2 = t1 + np.outer(np.ones(num2),list(range(-L,L+1)))
		t2[t2<s1] = s1
		t2[t2>=s2] = s2-1
		idx = np.int64(t2)
		# print(idx)
		vec1_local[s1:s2] = idx
		vec1_serial[s1:s2] = ref_serial[idx]
		feature_mtx[s1:s2] = x_mtx[idx]
		signal_mtx[s1:s2] = y[idx]

		# if i%10000==0:
		# 	print(i,num2,vec1_local[s1],vec1_serial[s1])
	
	id1 = np.where(id_vec>0)[0]
	num2 = len(id1)

	if num2<num_sample:
		feature_mtx, signal_mtx = feature_mtx[id1], signal_mtx[id1]
		# vec1_serial, vec1_local = vec1_serial[id1], vec1_local[id1]
		vec1_serial = vec1_serial[id1]

		id_1 = -np.ones(sample_num,dtype=np.int64)
		id_1[id1] = np.arange(num2)
		vec1_local = id_1[vec1_local]
		b1 = np.where(vec1_local<0)[0]
		if len(b1)>0:
			print('error!',b1)
			return -1
		
	# signal_mtx = signal_mtx[:,np.newaxis]
	signal_mtx = np.expand_dims(signal_mtx, axis=-1)

	# signal_mtx = np.expand_dims(signal_ntx, axis=-1)

	return feature_mtx, signal_mtx, vec1_serial, vec1_local

def score_2a(y, y_predicted):

	score1 = mean_squared_error(y, y_predicted)
	score2 = pearsonr(y, y_predicted)
	score3 = explained_variance_score(y, y_predicted)
	score4 = mean_absolute_error(y, y_predicted)
	score5 = median_absolute_error(y, y_predicted)
	score6 = r2_score(y, y_predicted)
	score7, pvalue = spearmanr(y,y_predicted)
	# vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6]
	vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6, score7, pvalue]

	return vec1

def read_phyloP(species_name):

	path1 = './'
	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path1,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	num_sample = len(chrom_ori)
	chrom_vec = np.unique(chrom_ori)
	chrom_vec = ['chr22']

	for chrom_id in chrom_vec:
		filename1 = '%s/phyloP/hg19.phyloP100way.%s.bedGraph'%(path1,chrom_id)
		data1 = pd.read_csv(filename1,header=None,sep='\t')
		chrom, start, stop, score = data1[0], data1[1], data1[2], data1[3]
		len1 = stop-start
		b = np.where(chrom_ori==chrom_id)[0]
		num_sample1 = len(b)
		vec1 = np.zeros((num_sample1,16))
		print(chrom_id,len(chrom),len(b))
		cnt = 0
		b1 = [-1]
		for i in b:
			t1 = b1[-1]+1
			b1 = np.where((start[t1:]>=start_ori[i])&(stop[t1:]<stop_ori[i]))[0]+t1
			if len(b1)==0:
				b1 = [-1]
				continue

			t_len1, t_score = np.asarray(len1[b1]), np.asarray(score[b1])
			s1 = 0
			s2 = np.sum(t_len1)
			i1 = cnt
			for j in range(0,12):
				temp1 = (j-8)*2.5
				b2 = np.where((t_score<temp1+2.5)&(t_score>=temp1))[0]
				print(b2)
				vec1[i1,j] = np.sum(t_len1[b2])*1.0/s2
				s1 = s1+temp1*vec1[i1,j]

			vec1[i1,12] = s1   # average
			vec1[i1,13] = np.median(t_score)
			vec1[i1,14] = np.max(t_score)
			vec1[i1,15] = np.min(t_score)
			
			cnt += 1
			if cnt%1000==0:
				print(cnt,len(b1),s2,vec1[i1,12:16])
				break
		
		# dict1 = dict()
		# dict1['vec'], dict1['index'] = vec1,b
		# np.save('phyloP_%s'%(chrom_id),dict1,allow_pickle=True)
		fields = ['index']
		for j in range(0,12):
			temp1 = (j-8)*2.5
			fields.append('%s-%s'%(temp1,temp1+2.5))
		fields.extend(range(0,4))
		data1 = pd.DataFrame(data = np.hstack((b[:,np.newaxis],vec1)),columns=fields)
		data1.to_csv('phyloP_%s.txt'%(chrom_id),sep='\t',index=False)

	return vec1

def read_phyloP_1(ref_filename,header,file_path,chrom_vec,n_level=15,offset=10,magnitude=2):

	file1 = pd.read_csv(ref_filename,header=header,sep='\t')
	
	# col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	colnames = list(file1)
	col1, col2, col3, col4 = colnames[0], colnames[1], colnames[2], colnames[3]
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1[col4])
	num_sample = len(chrom_ori)
	# chrom_vec = np.unique(chrom_ori)
	# chrom_vec = [chrom_id]
	# n_level, offset, magnitude = 15, 10, 2
	score_max = (n_level-offset)*magnitude

	for chrom_id in chrom_vec:
		# filename1 = '%s/hg19.phyloP100way.%s.bedGraph'%(file_path,chrom_id)
		filename1 = '%s/chr%s.phyloP100way.bedGraph'%(file_path,chrom_id)
		data1 = pd.read_csv(filename1,header=None,sep='\t')
		chrom, start, stop, score = data1[0], data1[1], data1[2], data1[3]
		len1 = stop-start
		chrom_id1 = 'chr%s'%(chrom_id)
		b = np.where(chrom_ori==chrom_id1)[0]
		num_sample1 = len(b)
		vec1 = np.zeros((num_sample1,n_level+4))
		print(chrom_id,len(chrom),len(b))
		cnt = 0
		m_idx = len(start)-1
		start_idx = 0
		print("number of regions", len(b))
		for i in b:
			t_start, t_stop = start_ori[i], stop_ori[i] # position of zero region
			position = [t_start,t_stop]
			if start_idx<=m_idx:
				b1, start_idx = utility_1.search_region_include(position, start, stop, m_idx, start_idx)

			# print(count,t_start,t_stop,t_stop-t_start,start_idx,len(id3))
			if len(b1)==0:
				continue

			t_len1, t_score = np.asarray(len1[b1]), np.asarray(score[b1])
			t_score[t_score>score_max] = score_max-1e-04
			s1 = 0
			s2 = np.sum(t_len1)
			for j in range(0,n_level):
				temp1 = (j-offset)*magnitude
				b2 = np.where((t_score<temp1+magnitude)&(t_score>=temp1))[0]
				# print(b2)
				vec1[cnt,j] = np.sum(t_len1[b2])*1.0/s2
				s1 = s1+temp1*vec1[cnt,j]

			vec1[cnt,n_level:n_level+4] = [s1,np.median(t_score),np.max(t_score),np.min(t_score)]
			
			cnt += 1
			pre_b1 = b1
			if cnt%1000==0:
				print(chrom_id,cnt,len(b1),s2,vec1[cnt,-4:])
				# break
		
		# dict1 = dict()
		# dict1['vec'], dict1['index'] = vec1,b
		# np.save('phyloP_%s'%(chrom_id),dict1,allow_pickle=True)
		fields = ['index']
		for j in range(0,n_level):
			temp1 = (j-offset)*magnitude
			fields.append('%s-%s'%(temp1,temp1+magnitude))
		fields.extend(range(0,4))
		idx = serial_ori[b]
		data1 = pd.DataFrame(data = np.hstack((idx[:,np.newaxis],vec1)),columns=fields)
		data1.to_csv('phyloP_%s.txt'%(chrom_id),sep='\t',index=False)

	return vec1

def read_motif_1(filename,output_filename=-1):

	data1 = pd.read_csv(filename,sep='\t')
	colnames = list(data1)
	col1, col2, col3 = colnames[0], colnames[1], colnames[2]
	chrom, start, stop = np.asarray(data1[col1]), np.asarray(data1[col2]), np.asarray(data1[col3])
	region_len = stop-start
	m1, m2, median_len = np.max(region_len), np.min(region_len), np.median(region_len)
	b1 = np.where(region_len!=median_len)[0]
	print(m1,m2,median_len,len(b1))

	bin_size = median_len

	motif_name = colnames[3:]
	mtx1 = np.asarray(data1.loc[:,motif_name])
	mtx1 = mtx1*1000.0/np.outer(region_len,np.ones(mtx1.shape[1]))
	print('motif',len(motif_name))

	print(mtx1.shape)
	print(np.max(mtx1),np.min(mtx1),np.median(mtx1))

	if output_filename!=-1:
		fields = colnames
		data1 = pd.DataFrame(columns=fields)
		data1[colnames[0]], data1[colnames[1]], data1[colnames[2]] = chrom, start, stop
		num1 = len(fields)-3
		for i in range(0,num1):
			data1[colnames[i+3]] = mtx1[:,i]
		
		data1.to_csv(output_filename,header=True,index=False,sep='\t')
		print(output_filename, data1.shape)

	return mtx1, chrom, start, stop, colnames

def read_gc_1(ref_filename,header,filename,output_filename):

	# path1 = '/volume01/yy3/seq_data/dl/replication_timing'
	# filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path1,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)

	sel_idx = []
	file1 = pd.read_csv(ref_filename,header=header,sep='\t')
	f_list = load_seq_altfeature(filename,sel_idx)

	# col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	colnames = list(file1)
	col1, col2, col3, col4 = colnames[0], colnames[1], colnames[2], colnames[3]
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1[col4])
	num_sample = len(chrom_ori)

	if num_sample!=f_list.shape[0]:
		print('error!',num_sample,f_list.shape[0])

	fields = ['chrom','start','stop','serial','GC','GC_N','GC_skew']
	file2 = pd.DataFrame(columns=fields)
	file2['chrom'], file2['start'], file2['stop'], file2['serial'] = chrom_ori, start_ori, stop_ori, serial_ori
	for i in range(0,3):
		file2[fields[i+4]] = f_list[:,i]

	file2.to_csv(output_filename,index=False,sep='\t')
	
	return f_list

def generate_serial(filename1,chrom,start,stop):

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
	
	# filename1 = '/volume01/yy3/seq_data/genome/hg38.chrom.sizes'
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	ref_chrom, chrom_size = np.asarray(data1[0]), np.asarray(data1[1])
	serial_start = 0
	serial_vec = np.zeros(len(chrom))
	bin_size = stop[1]-start[1]
	print(bin_size)
	for chrom_id in chrom_vec:
		b1 = np.where(ref_chrom==chrom_id)[0]
		t_size = chrom_size[b1[0]]
		b2 = np.where(chrom==chrom_id)[0]
		if len(b1)>0:
			size1 = int(np.ceil(t_size*1.0/bin_size))
			serial = np.int64(start[b2]/bin_size)+serial_start
			serial_vec[b2] = serial
			print(chrom_id,b2,len(serial),serial_start,size1)
			serial_start = serial_start+size1
		else:
			print("error!")
			return

	return np.int64(serial_vec)

def generate_serial_local(filename1,chrom,start,stop,chrom_num):

	# chrom_vec = np.sort(np.unique(chrom))
	# print(chrom_vec)
	chrom_vec = []
	for i in range(1,chrom_num+1):
		chrom_vec.append('chr%d'%(i))
	chrom_vec += ['chrX']
	chrom_vec += ['chrY']
	chrom_vec += ['chrM']
	print(chrom_vec)
	print(chrom)
	print(len(chrom))

	t_chrom = np.unique(chrom)
	
	# filename1 = '/volume01/yy3/seq_data/genome/hg38.chrom.sizes'
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	ref_chrom, chrom_size = np.asarray(data1[0]), np.asarray(data1[1])
	# serial_start = np.zeros(len(chrom))
	serial_start = 0
	serial_start_1 = dict()
	serial_vec = np.zeros(len(chrom))
	bin_size = stop[1]-start[1]
	print(bin_size)
	for chrom_id in chrom_vec:
		b1 = np.where(ref_chrom==chrom_id)[0]
		t_size = chrom_size[b1[0]]
		serial_start_1[chrom_id] = serial_start
		size1 = int(np.ceil(t_size*1.0/bin_size))
		serial_start = serial_start+size1

	for chrom_id in t_chrom:
		b2 = np.where(chrom==chrom_id)
		serial = np.int64(start[b2]/bin_size)+serial_start_1[chrom_id]
		serial_vec[b2] = serial

	return np.int64(serial_vec)

def generate_serial_start(filename1,chrom,start,stop,chrom_num=19):

	# chrom_vec = np.sort(np.unique(chrom))
	# print(chrom_vec)
	chrom_vec = []
	for i in range(1,chrom_num+1):
		chrom_vec.append('chr%d'%(i))
	chrom_vec += ['chrX']
	chrom_vec += ['chrY']
	print(chrom_vec)
	print(chrom)
	print(len(chrom))
	
	# filename1 = '/volume01/yy3/seq_data/genome/hg38.chrom.sizes'
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	ref_chrom, chrom_size = np.asarray(data1[0]), np.asarray(data1[1])
	serial_start = 0
	serial_vec = -np.ones(len(chrom))
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
			serial_vec[b2] = serial
			print(chrom_id,b2,len(serial),serial_start,size1)
			serial_start = serial_start+size1
		else:
			print("error!")
			return

	return np.int64(serial_vec), start_vec

def shuffle_array(vec):
	num1 = len(vec)
	idx = np.random.permutation(num1)
	vec = vec[idx]

	return vec, idx

# input: estimated attention, type_id: training, validation, or test data
# output: ranking of attention
def select_region1_sub(filename,type_id):

	data1 = pd.read_csv(filename,sep='\t')
	colnames = list(data1)

	# chrom	start	stop	serial	signal	predicted_signal	predicted_attention
	chrom, start, serial = data1['chrom'], data1['start'], data1['serial']
	chrom, start, serial = np.asarray(chrom), np.asarray(start), np.asarray(serial)
	predicted_attention = data1['predicted_attention']
	predicted_attention = np.asarray(predicted_attention)

	ranking = stats.rankdata(predicted_attention,'average')/len(predicted_attention)
	rank1 = np.zeros((len(predicted_attention),2))
	rank1[:,0] = ranking

	chrom_vec = np.unique(chrom)
	for t_chrom in chrom_vec:
		b1 = np.where(chrom==t_chrom)[0]
		t_attention = predicted_attention[b1]
		t_ranking = stats.rankdata(t_attention,'average')/len(t_attention)
		rank1[b1,1] = t_ranking

	data1['Q1'] = rank1[:,0]	# rank across all the included chromosomes
	data1['Q2'] = rank1[:,1]	# rank by each chromosome
	data1['typeId'] = np.int8(type_id*np.ones(len(rank1)))

	return data1,chrom_vec

# merge estimated attention from different training/test splits
# type_id1: chromosome order; type_id2: training: 0, test: 1, valid: 2
def select_region1_merge(filename_list,output_filename,type_id1=0,type_id2=1):

	list1 = []
	chrom_numList = []
	# b1 = np.where((self.chrom!='chrX')&(self.chrom!='chrY'))[0]
	# ref_chrom, ref_start, ref_serial = self.chrom[b1], self.start[b1], self.serial[b1]
	# num_sameple = len(ref_chrom)
	i = 0
	serial1 = []
	num1 = len(filename_list)
	vec1 = list(range(num1))
	if type_id1==1:
		vec1 = list(range(num1-1,-1,-1))
	for i in vec1:
		filename1 = filename_list[i]
		# data1: chrom, start, stop, serial, signal, predicted_signal, predicted_attention, Q1, Q2, typeId
		# typeId: training: 0, test: 1, valid: 2
		data1, chrom_vec = select_region1_sub(filename1,type_id2)
		print(filename1,len(data1))
		# list1.append(data1)
		# if i==0:
		# 	serial1 = np.asarray(data1['serial'])
		t_serial = np.asarray(data1['serial'],dtype=np.int64)
		t_serial2 = np.setdiff1d(t_serial,serial1)
		serial1 = np.union1d(serial1,t_serial)
		id1 = mapping_Idx(t_serial,t_serial2)
		colnames = list(data1)
		data1 = data1.loc[id1,colnames]
		list1.append(data1)
		chrom_numList.append(chrom_vec)

	data2 = pd.concat(list1, axis=0, join='outer', ignore_index=True, 
				keys=None, levels=None, names=None, verify_integrity=False, copy=True)
	print('sort')
	data2 = data2.sort_values(by=['serial'])
	data2.to_csv(output_filename,index=False,sep='\t')

	return data2, chrom_numList

class Reader(object):

	def __init__(self, ref_filename, feature_idvec = [1,1,1,1]):
		# Initializes RepliSeq
		self.ref_filename = ref_filename
		self.feature_idvec = feature_idvec
		
	def generate_serial(self,filename1,filename2,output_filename,header=None):

		data1 = pd.read_csv(filename2, header=header, sep='\t')
		colnames = list(data1)
		chrom, start, stop = np.asarray(data1[colnames[0]]), np.asarray(data1[colnames[1]]), np.asarray(data1[colnames[2]])
		serial_vec, start_vec = generate_serial_start(filename1,chrom,start,stop)

		if output_filename!=None:
			colnames2 = colnames[0:3]+['serial']+colnames[3:]
			data2 = pd.DataFrame(columns=colnames2)
			data2['serial'] = serial_vec
			for colname1 in colnames:
				data2[colname1] = data1[colname1]
			flag = False
			if header!=None:
				flag = True
			data2.to_csv(output_filename,header=flag,index=False,sep='\t')

		return serial_vec, start_vec

	def load_motif(self,filename1,motif_filename,output_filename):

		# output_filename = None
		# ref_filename = 'hg38.5k.serial.bed'
		# motif_filename = 'hg38.motif.count.txt'
		# output_filename1 = None
		mtx1, chrom, start, stop, colnames = read_motif_1(motif_filename)

		serial_vec, start_vec = generate_serial_start(filename1,chrom,start,stop)

		if output_filename!=None:
			colnames2 = ['chrom','start','stop','serial']
			data2 = pd.DataFrame(columns=colnames2)
			data2['chrom'], data2['start'], data2['stop'], data2['serial'] = chrom, start, stop, serial_vec

			data3 = pd.DataFrame(columns=colnames[3:],data=mtx1)

			data1 = pd.concat([data2,data3], axis=1, join='outer', ignore_index=True, 
								keys=None, levels=None, names=None, verify_integrity=False, copy=True)

			data1.to_csv(output_filename,header=True,index=False,sep='\t')
			print('data1',data1.shape)

		return True

class ConvergenceMonitor(object):

	_template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

	def __init__(self, tol, n_iter, verbose):
		self.tol = tol
		self.n_iter = n_iter
		self.verbose = verbose
		self.history = deque(maxlen=2)
		self.iter = 0

	def __repr__(self):
		class_name = self.__class__.__name__
		params = dict(vars(self), history=list(self.history))
		return "{0}({1})".format(
			class_name, _pprint(params, offset=len(class_name)))

	def report(self, logprob):
		if self.verbose:
			delta = logprob - self.history[-1] if self.history else np.nan
			message = self._template.format(
				iter=self.iter + 1, logprob=logprob, delta=delta)
			print(message, file=sys.stderr)

		self.history.append(logprob)
		self.iter += 1

	@property
	def converged(self):
		return (self.iter == self.n_iter or
				(len(self.history) == 2 and
				 self.history[1] - self.history[0] < self.tol))

class _Base1(BaseEstimator):

	def __init__(self, file_path, species_id, resolution, run_id, generate,
					chromvec,test_chromvec,
					featureid,type_id,cell,method,ftype,ftrans,tlist,
					flanking,normalize,
					config,
					attention=1,feature_dim_motif=1,
					kmer_size=[6,5]):
		# Initializes RepliSeq
		self.run_id = run_id
		self.cell = cell
		self.generate = generate
		self.train_chromvec = chromvec
		self.chromosome = chromvec[0]
		print('train_chromvec',train_chromvec)
		print('test_chromvec',test_chromvec)
		self.test_chromvec = test_chromvec
		self.config = config
		self.n_epochs = config['n_epochs']
		self.species_id = species_id
		self.type_id = type_id
		self.cell_type = cell
		self.cell_type1 = config['celltype_id']
		self.method = method
		self.ftype = ftype
		self.ftrans = ftrans[0]
		self.ftrans1 = ftrans[1]
		self.t_list = tlist
		self.flanking = flanking
		self.flanking1 = 3
		self.normalize = normalize
		self.batch_size = config['batch_size']
		# config = dict(output_dim=hidden_unit,fc1_output_dim=fc1,fc2_output_dim=fc2,units1=units1[0],
		# 				units2=units1[1],n_epochs=n_epochs,batch_size=batch_size)
		# config['feature_dim_vec'] = units1[2:]
		self.tol = config['tol']
		self.attention = attention
		self.attention_vec = [12,17,22,32,51,52,58,60]
		self.attention_vec1 = [1]
		self.lr = config['lr']
		self.step = config['step']
		self.feature_type = -1
		self.kmer_size = kmer_size
		self.activation = config['activation']
		self.min_delta = config['min_delta']
		self.chromvec_sel = chromvec
		self.feature_dim_transform = config['feature_dim_transform']
		feature_idvec = [1,1,1,1]
		# ref_filename = 'hg38_5k_serial.bed'
		if 'ref_filename' in config:
			ref_filename = config['ref_filename']
		else:
			ref_filename = 'hg38_5k_serial.bed'
		self.reader = Reader(ref_filename, feature_idvec)
		self.predict_type_id = 0
		self.method = method
		self.train = self.config['train_mode']
		# self.path = '/mnt/yy3'
		self.path = file_path
		self.model_path = '%s/test_%d.h5'%(self.path,run_id)
		self.pos_code = config['pos_code']
		self.feature_dim_select1 = config['feature_dim_select']
		self.method_vec = [[11,31],[22,32,52,17,51,58,60],[56,62]]
		self.resolution = resolution
		# if self.species_id=='mm10':
		# 	self.cell_type1 = config['cell_type1']

		if 'cell_type1' in self.config:
			self.cell_type1 = config['cell_type1']

		if ('load_type' in self.config) and (self.config['load_type']==1):
			self.load_type = 1
		else:
			self.load_type = 0
		
		if (method>10) and not(method in [56]) :
			self.predict_context = 1
		else:
			self.predict_context = 0

		if ftype[0]==-5:
			self.feature_idx1= -5 # full dimensions
		elif ftype[0]==-6:
			self.feature_idx1 = -6	# frequency dimensions
		else:
			self.feature_idx1 = ftype

		if 'est_attention_type1' in self.config:
			self.est_attention_type1 = self.config['est_attention_type1']
		else:
			self.est_attention_type1 = 1

		if 'est_attention_sel1' in self.config:
			self.est_attention_sel1 = self.config['est_attention_sel1']
		else:
			self.est_attention_sel1 = 0

		# self.feature_idx = [0,2]
		self.feature_idx = featureid
		self.x, self.y = dict(), dict() # feature matrix and signals
		self.vec = dict()	# serial
		self.vec_local = dict()

		if self.species_id.find('hg')>=0:
			self.chrom_num = 22
		elif self.species_id.find('mm')>=0:
			self.chrom_num = 19
		else:
			self.chrom_num = -1

		self.region_list_test, self.region_list_train, self.region_list_valid = [],[],[]
		if 'region_list_test' in config:
			self.region_list_test = config['region_list_test']
		
		if 'region_list_train' in config:
			self.region_list_train = config['region_list_train']

		if 'region_list_valid' in config:
			self.region_list_valid = config['region_list_valid']

		flag = False
		if 'scale' in config:
			flag = True
			self.scale = config['scale']
		else:
			self.scale = [0,1]

		if ('activation_basic' in config) and (config['activation_basic']=='tanh'):
			if (flag==True) and (self.scale[0]>=0):
				flag = False
			if flag==False:
				self.scale = [-1,1]

		self.region_boundary = []
		self.serial_vec = []
		self.f_mtx = []
		
		print('scale',self.scale)
		print(self.test_chromvec)
		filename1 = '%s_chr%s-chr%s_chr%s-chr%s'%(self.cell_type, self.train_chromvec[0], self.train_chromvec[-1], self.test_chromvec[0], self.test_chromvec[-1])
		self.filename_load = filename1
		print(self.filename_load,self.method,self.predict_context,self.attention)
		self.set_generate(generate,filename1)

	def load_ref_serial(self, ref_filename, header=None):

		# path2 = '/volume01/yy3/seq_data/dl/replication_timing'
		# filename1 = '%s/estimate_rt/estimate_rt_%s.1.txt'%(path2,species_name)
		# filename2a = 'test_seq_%s.1.txt'%(species_name)
		if header==None:
			file1 = pd.read_csv(ref_filename,header=header,sep='\t')
		else:
			file1 = pd.read_csv(ref_filename,sep='\t')
		colnames = list(file1)		
		# col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
		col1, col2, col3, col_serial = colnames[0], colnames[1], colnames[2], colnames[3]
		self.chrom_ori, self.start_ori, self.stop_ori, self.serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1[col_serial])
		print('load ref serial', self.serial_ori.shape)
		
		return self.serial_ori

	# load local serial and signal
	def load_local_serial(self, filename1, header=None, region_list=[], type_id2=1, signal_normalize=1,region_list_1=[]):

		if header==None:
			file2 = pd.read_csv(filename1,header=header,sep='\t')
		else:
			file2 = pd.read_csv(filename1,sep='\t')

		colnames = list(file2)
		col1, col2, col3, col_serial = colnames[0], colnames[1], colnames[2], colnames[3]
		# sort the table by serial
		file2 = file2.sort_values(by=[col_serial])

		self.chrom, self.start, self.stop, self.serial = np.asarray(file2[col1]), np.asarray(file2[col2]), np.asarray(file2[col3]), np.asarray(file2[col_serial])

		b = np.where((self.chrom!='chrX')&(self.chrom!='chrY')&(self.chrom!='chrM'))[0]
		self.chrom, self.start, self.stop, self.serial = self.chrom[b], self.start[b], self.stop[b], self.serial[b]

		if self.chrom_num>0:
			chrom_num = self.chrom_num
		else:
			chrom_num = len(np.unique(self.chrom))
		chrom_vec = [str(i) for i in range(1,chrom_num+1)]
		print('chrom_vec', chrom_vec)

		self.bin_size = self.stop[1]-self.start[1]
		scale = self.scale
		if len(colnames)>=5:
			col_signal = colnames[4]
			self.signal = np.asarray(file2[col_signal])
			self.signal = self.signal[b]
			self.signal_pre = self.signal.copy()

			if signal_normalize==1:
				if self.run_id>10:
					# self.signal = signal_normalize(self.signal,[0,1]) # normalize signals
					self.signal_pre1, id1, signal_vec1 = self.signal_normalize_chrom(self.chrom,self.signal,chrom_vec,scale)

					if not('train_signal_update' in self.config) or (self.config['train_signal_update']==1):
						train_signal, id2, signal_vec2 = self.signal_normalize_chrom(self.chrom,self.signal,self.train_chromvec,scale)
						id_1 = mapping_Idx(id1,id2)
						self.signal = self.signal_pre1.copy()
						self.signal[id_1] = train_signal
					else:
						self.signal = self.signal_pre1.copy()
				else:
					print('signal_normalize_bychrom')
					self.signal, id1, signal_vec = self.signal_normalize_bychrom(self.chrom,self.signal,chrom_vec,scale)
		else:
			self.signal = np.ones(len(b))
		# print(self.signal.shape)
		print('load local serial', self.serial.shape, self.signal.shape, np.max(self.signal), np.min(self.signal))

		if 'tol_region_search' in self.config:
			tol = self.config['tol_region_search']
		else:
			tol = 2

		# only train or predict on some regions
		print('load_local_serial',len(self.chrom))
		if len(region_list_1)>0:
			num1 = len(region_list_1)
			list1 = []
			for i in range(num1):
				t_region = region_list_1[i]
				t_chrom, t_start, t_stop = 'chr%d'%(t_region[0]), t_region[1], t_region[2]
				t_id1 = np.where((self.chrom==t_chrom)&(self.start<t_stop)&(self.stop>t_start))[0]
				list1.extend(t_id1)

			b1 = np.asarray(list1)
			self.chrom, self.start, self.stop, self.serial = self.chrom[b1], self.start[b1], self.stop[b1], self.serial[b1]
			print('load_local_serial',num1,len(self.chrom))
			print(region_list_1)

		if len(region_list)>0:
			# print('load_local_serial',region_list)
			# id1, region_list = self.region_search_1(chrom,start,stop,serial,region_list)
			id1, region_list = self.region_search_1(self.chrom,self.start,self.stop,self.serial,region_list,type_id2,tol)
			self.chrom, self.start, self.stop, self.serial, self.signal = self.chrom[id1], self.start[id1], self.stop[id1], self.serial[id1], self.signal[id1]

			id2 = self.region_search_boundary(self.chrom,self.start,self.stop,self.serial,region_list)
			# print('region_search_boundary', id2[:,0], self.start[id2[:,1:3]],self.stop[id2[:,1:3]])
			self.region_boundary = id2
			# print(self.serial[id2[:,1:3]])
			print('region_boundary',id2)

			# return

		else:
			print('load_local_serial',region_list)
			# assert len(region_list)>0
			# return

		return self.serial, self.signal

	# training, validation and test data index
	def prep_training_test(self,train_sel_list_ori):

		train_id1, test_id1, y_signal_train1, y_signal_test, train1_sel_list, test_sel_list = self.generate_train_test_1(train_sel_list_ori)

		self.idx_list = {'test':test_id1}
		self.y_signal = {'test':y_signal_test}

		if len(y_signal_test)>0:
			print('y_signal_test',np.max(y_signal_test),np.min(y_signal_test))

		if len(y_signal_train1)>0:
			print('y_signal_train',np.max(y_signal_train1),np.min(y_signal_train1))
			self.idx_list.update({'train':[],'valid':[]})
		else:
			return

		# y_signal_test_ori = signal_normalize(y_signal_test,[0,1])
		# shuffle array
		# x_test_trans, shuffle_id2 = shuffle_array(x_test_trans)
		# test_sel_list = test_sel_list[shuffle_id2]
		# x_train1_trans, shuffle_id1 = shuffle_array(x_train1_trans)
		# train_sel_list = train_sel_list[shuffle_id1]

		print(train1_sel_list[0:5])

		# split training and validation data
		if 'ratio1' in self.config:
			ratio = self.config['ratio1']
		else:
			ratio = 0.95
		if 'type_id1' in self.config:
			type_id_1 = self.config['type_id1']
		else:
			type_id_1 = 0

		idx_train, idx_valid, idx_test = self.generate_index_1(train1_sel_list, test_sel_list, ratio, type_id_1)
		print('idx_train,idx_valid,idx_test', len(idx_train), len(idx_valid), len(idx_test))

		if (len(self.region_list_train)>0) or (len(self.region_list_valid)>0):
			idx_train, idx_valid = self.generate_train_test_2(train1_sel_list,idx_train,idx_valid)
			print('idx_train,idx_valid', len(idx_train), len(idx_valid))

		train_sel_list, val_sel_list = train1_sel_list[idx_train], train1_sel_list[idx_valid]

		self.idx_list.update({'train':train_id1[idx_train],'valid':train_id1[idx_valid]})
		self.idx_train_val = {'train':idx_train,'valid':idx_valid}
		self.y_signal.update({'train':y_signal_train1[idx_train],'valid':y_signal_train1[idx_valid]})

		return train_sel_list, val_sel_list, test_sel_list

	# prepare data from predefined features: kmer frequency feature and motif feature
	def prep_data_sub2(self,path1,file_prefix,type_id2,feature_dim1,feature_dim2,flag_1):

		species_id = self.species_id
		celltype_id = self.cell_type1
		if species_id=='mm10':
			kmer_dim_ori, motif_dim_ori = 100, 50
			filename1 = '%s/%s_%d_%d_%d.npy'%(path1,file_prefix,type_id2,kmer_dim_ori,motif_dim_ori)
			# filename2 = 'test_%s_genome%d_kmer7.h5'%(species_id,celltype_id)
			filename2 = '%s_%d_kmer7_0_200_trans.h5'%(species_id,celltype_id)
		else:
			kmer_dim_ori, motif_dim_ori = 50, 50
			filename1 = '%s/%s_%d_%d_%d.npy'%(path1,file_prefix,type_id2,kmer_dim_ori,motif_dim_ori)
			# filename2 = 'test_%s_kmer7.h5'%(species_id)
			filename2 = '%s_kmer7_0_200_trans.h5'%(species_id)

		kmer_size1, kmer_size2, kmer_size3 = 5,6,7
		x_train1_trans, train_sel_list_ori = [], []
		flag1, flag2 = 0, 0
		flag3 = True
		# if kmer_size2 in self.kmer_size:
		if flag3==True:
			if os.path.exists(filename1)==True:
				print("loading data...")
				data1 = np.load(filename1,allow_pickle=True)
				data_1 = data1[()]
				x_train1_trans_ori, train_sel_list_ori = np.asarray(data_1['x1']), np.asarray(data_1['idx'])

				print('train_sel_list',train_sel_list_ori.shape)
				print('x_train1_trans',x_train1_trans_ori.shape)
				if kmer_size2 in self.kmer_size:
					flag1 = 1
				serial1 = train_sel_list_ori[:,1]
				dim1 = x_train1_trans_ori.shape[1]
				if (self.feature_dim_motif==0) or (flag_1==True):
					x_train1_trans = x_train1_trans_ori[:,0:-motif_dim_ori]
				else:
					# d1 = np.min((dim1-motif_dim_ori+feature_dim2,d1))
					# d2 = dim1-motif_dim_ori
					# sel_id1 = list(range(21))+list(range(21,21+feature_dim1))
					# x_train1_trans_1 = x_train1_trans[:,sel_id1]
					# x_train1_trans_2 = x_train1_trans[:,d2:d1]
					x_train1_trans_1 = x_train1_trans_ori[:,0:dim1-motif_dim_ori]
					x_train1_trans_2 = x_train1_trans_ori[:,dim1-motif_dim_ori:]

			else:
				print('data not found!')
				print(filename1)
				return x_train1_trans, trans_sel_list_ori

		if kmer_size3 in self.kmer_size:
			with h5py.File(filename2,'r') as fid:
				serial2 = fid["serial"][:]
				feature_mtx = fid["vec"][:]
				# feature_mtx = feature_mtx[:,0:kmer_dim_ori]
				print(serial2)
				print(len(serial2),feature_mtx.shape)
				flag2 = 1

		if flag1==1:
			if flag2==1:
				t_serial = np.intersect1d(serial1,serial2)
				id1 = mapping_Idx(serial1,t_serial)
				id2 = mapping_Idx(serial2,t_serial)

				if 'feature_dim_transform_1' in self.config:
					sel_idx = self.config['feature_dim_transform_1']
					sel_id1, sel_id2 = list(0,21)+list(range(sel_idx[0])), range(sel_idx[1])
				else:
					sel_id1 = list(0,21)+list(range(10))
					sel_id2 = range(feature_dim1-sel_idx1)

				if (self.feature_dim_motif==0) or (flag_1==True):
					x_train1_trans = np.hstack((x_train1_trans[id1,sel_id1],feature_mtx[id2,sel_id2]))
				else:
					x_train1_trans = np.hstack((x_train1_trans_1[id1,sel_id1],feature_mtx[id2,sel_id2],x_train1_trans_2[id1,0:feature_dim2]))

				train_sel_list_ori = train_sel_list_ori[id1]
			else:
				pass

		elif flag2==1:
			t_serial = np.intersect1d(serial1,serial2)
			id1 = mapping_Idx(serial1,t_serial)
			id2 = mapping_Idx(serial2,t_serial)

			x_train1_trans = np.hstack((x_train1_trans_ori[id1,0:2],feature_mtx[id2,0:feature_dim1]))
			train_sel_list_ori = train_sel_list_ori[id1]
			self.feature_dim_select1 = -1

			if (self.feature_dim_motif==1) and (flag_1==False):
				x_train1_trans = np.hstack((x_train1_trans,x_train1_trans_2[id1,0:feature_dim2]))

			# id1 = mapping_Idx(self.serial_ori,serial2)
			# b1 = (id1>=0)
			# id1 = id1[b1]
			# serial2, feature_mtx = serial2[b1], feature_mtx[b1]

			# chrom1 = self.chrom_ori[id1]
			# chrom2 = np.zeros(len(serial2),dtype=np.int32)
			# chrom_vec = np.unique(chrom1)
			# for chrom_id in chrom_vec:
			# 	b2 = np.where(chrom1==chrom_id)[0]
			# 	chrom_id1 = int(chrom_id[3:])
			# 	chrom2[b2] = chrom_id1

			# x_train1_trans = feature_mtx[:,0:feature_dim1]
			# trans_sel_list_ori = np.vstack((chrom2,serial2)).T

		else:
			print('data not found!')

		return x_train1_trans, train_sel_list_ori

	# prepare data from predefined features
	def prep_data_sub1(self,path1,file_prefix,type_id2,feature_dim_transform,load_type=0):

		self.feature_dim_transform = feature_dim_transform
		# map_idx = mapping_Idx(serial_ori,serial)

		sub_sample_ratio = 1
		shuffle = 0
		normalize, flanking, attention, run_id = self.normalize, self.flanking, self.attention, self.run_id
		config = self.config
		vec2 = dict()
		tol = self.tol
		L = flanking

		# np.save(filename1)
		print("feature transform")
		# filename1 = '%s/%s_%d_%d_%d.npy'%(path1,file_prefix,type_id2,feature_dim_transform[0],feature_dim_transform[1])
		
		print(self.species_id)

		t_featuredim1, t_featuredim2 = feature_dim_transform[0], feature_dim_transform[1]

		flag1 = False
		if self.species_id=='hg38':
			if 'motif_trans_typeid' in self.config:
				flag1 = True

		if (self.species_id=='mm10'):
			flag1 = True

		if (t_featuredim1>0) or (flag1==False):
			x_train1_trans, train_sel_list_ori = self.prep_data_sub2(path1,file_prefix,type_id2,t_featuredim1,t_featuredim2,flag1)
			if len(x_train1_trans)==0:
				print('data not found!')
				return -1

		if t_featuredim2>0:
			print('train_sel_list',train_sel_list_ori.shape)
			print('x_train1_trans',x_train1_trans.shape)

			if (self.feature_dim_motif>=1) and (flag1==True):
				if self.species_id=='mm10':
					annot1 = '%s_%d_motif'%(self.species_id,self.cell_type1)
				else:
					annot1 = '%s_motif'%(self.species_id)

				motif_trans_typeid = self.config['motif_trans_typeid']
				motif_featuredim = self.config['motif_featuredim']
				motif_filename = '%s_%d_%d_trans.h5'%(annot1,motif_trans_typeid,motif_featuredim)
				if motif_featuredim<t_featuredim2:
					print('error! %d %d',motif_featuredim,t_featuredim2)
					t_featuredim2 = motif_featuredim

				with h5py.File(motif_filename,'r') as fid:
					serial_1 = fid["serial"][:]
					motif_data = fid["vec"][:]
					print(len(serial_1),motif_data.shape)

				serial1 = train_sel_list_ori[:,1]
				serial2 = serial_1

				t_serial = np.intersect1d(serial1,serial2)
				id1 = mapping_Idx(serial1,t_serial)
				id2 = mapping_Idx(serial2,t_serial)

				x_train1_trans = np.hstack((x_train1_trans[id1],motif_data[id2,0:t_featuredim2]))
				train_sel_list_ori = train_sel_list_ori[id1]
				# train_sel_list_ori2 = serial_1[id2]

		else:
			print("data not found!")
			return

		x_train1_trans = self.feature_dim_select(x_train1_trans,feature_dim_transform)

		# feature loaded not specific to cell type
		if load_type==1:
			return x_train1_trans, train_sel_list_ori

		list1 = ['motif_feature','feature2']
		for t_feature in list1:
			if (t_feature in self.config) and (self.config[t_feature]==1):
				if t_feature=='feature2':
					pre_config = self.config['pre_config']
					if self.chrom_num>0:
						chrom_num = self.chrom_num
					else:
						chrom_num = len(np.unique(self.chrom))

					chrom_vec = list(range(1,chrom_num+1))
					feature_mtx2, serial_2 = self.prep_data_sequence_3(pre_config,chrom_vec)
				else:
					x = 1

				x_train1_trans_ori1 = x_train1_trans.copy()
				train_sel_list_ori1 = train_sel_list_ori.copy()
				serial1 = train_sel_list_ori[:,1]
				serial2 = serial_2[:,1]

				t_serial = np.intersect1d(serial1,serial2)
				id1 = mapping_Idx(serial1,t_serial)[0]
				id2 = mapping_Idx(serial2,t_serial)[0]

				x_train1_trans = np.hstack((x_train1_trans[id1],feature_mtx2[id2]))
				train_sel_list_ori = train_sel_list_ori[id1]
				train_sel_list_ori2 = serial_2[id2]
				b1 = np.where(train_sel_list_ori[:,0]!=train_sel_list_ori2[:,0])[0]
				if len(b1)>0:
					print('error! train_sel_list_ori',len(b1))

		if ('centromere' in self.config) and (self.config['centromere']==1):
			regionlist_filename = 'hg38.centromere.bed'
			serial1 = train_sel_list_ori[:,1]
			serial_list1, centromere_serial = self.select_region(serial1, regionlist_filename)
			id1 = mapping_Idx(serial1,serial_list1)
			id1 = id1[id1>=0]
			x_train1_trans = x_train1_trans[id1]
			train_sel_list_ori = train_sel_list_ori[id1]
			print(x_train1_trans.shape,train_sel_list_ori.shape)

		print('positional encoding', self.pos_code)
		print('feature dim',x_train1_trans.shape)
		self.feature_dim = x_train1_trans.shape[1]
		start = time.time()
		if self.pos_code ==1:
			x_train1_trans = self.positional_encoding1(x_train1_trans,train_sel_list_ori,self.feature_dim)
			print(x_train1_trans.shape)
		stop = time.time()
		print('positional encoding', stop-start)

		## shuffle array
		if ('shuffle' in self.config) and (self.config['shuffle']==1):
			x_train1_trans, shuffle_id1 = shuffle_array(x_train1_trans)
			print('array shuffled')
			# np.random.shuffle(x_tran1_trans)
			# train_sel_list = train_sel_list[shuffle_id1]
		elif ('noise' in self.config) and (self.config['noise']>0):
			if self.config['noise']==1:
				x_train1_trans = np.zeros_like(x_train1_trans)
				print('x_train1_trans, noise 1', x_train1_trans[0:5])
			elif self.config['noise']==2:
				x_train1_trans = np.random.uniform(0,1,x_train1_trans.shape)
			else:
				x_train1_trans = np.random.normal(0,1,x_train1_trans.shape)
		else:
			pass

		if 'sub_sample_ratio' in self.config:
			sub_sample_ratio = self.config['sub_sample_ratio']
			num_sample = len(train_sel_list_ori)
			sub_sample = int(num_sample*sub_sample_ratio)
			train_sel_list_ori = train_sel_list_ori[0:sub_sample]
			x_train1_trans = x_train1_trans[0:sub_sample]

		# align train_sel_list_ori and serial
		print(train_sel_list_ori.shape,len(self.serial))
		id1 = mapping_Idx(train_sel_list_ori[:,1],self.serial)
		id2 = (id1>=0)
		print('mapping',len(self.serial),np.sum(id2),len(self.serial),len(id2))
		# self.chrom, self.start, self.stop, self.serial, self.signal = self.chrom[id2], self.start[id2], self.stop[id2], self.serial[id2], self.signal[id2]
		self.local_serial_1(id2)

		id1 = id1[id2]
		train_sel_list_ori = train_sel_list_ori[id1]
		x_train1_trans = x_train1_trans[id1]

		self.x_train1_trans = x_train1_trans
		self.train_sel_list = train_sel_list_ori

		return x_train1_trans, train_sel_list_ori

	def output_generate_sequences(self,idx_sel_list,seq_list):

		num1 = len(seq_list)
		t_serial1 = idx_sel_list[:,1]
		seq_list = np.asarray(seq_list)
		t_serial = t_serial1[seq_list]
		id1 = mapping_Idx(self.serial,t_serial[:,0])
		chrom1, start1, stop1 = self.chrom[id1], self.start[id1], self.stop[id1]

		id2 = mapping_Idx(self.serial,t_serial[:,1])
		chrom2, start2, stop2 = self.chrom[id2], self.start[id2], self.stop[id2]

		fields = ['chrom','start','stop','serial1','serial2']
		data1 = pd.DataFrame(columns=fields)
		data1['chrom'], data1['start'], data1['stop'] = chrom1, start1, stop2
		data1['serial1'], data1['serial2'] = t_serial[:,0], t_serial[:,1]
		data1['region_len'] = t_serial[:,1]-t_serial[:,0]+1

		output_filename = 'test_seqList_%d_%d.txt'%(idx_sel_list[0][0],idx_sel_list[0][1])
		data1.to_csv(output_filename,index=False,sep='\t')

		return True

	# prepare data from predefined features
	def prep_data(self,path1,file_prefix,type_id2,feature_dim_transform):

		x_train1_trans, train_sel_list_ori = self.prep_data_sub1(path1,file_prefix,type_id2,feature_dim_transform)

		train_sel_list, val_sel_list, test_sel_list = self.prep_training_test(train_sel_list_ori)
		
		# keys = ['train','valid','test']
		keys = ['train','valid']
		#  self.idx_sel_list = {'train':train1_sel_list,'valid':val_sel_list,'test':test_sel_list}
		idx_sel_list = {'train':train_sel_list,'valid':val_sel_list,'test':test_sel_list}
		# self.idx_sel_list = idx_sel_list

		# seq_list_train, seq_list_valid: both locally calculated
		self.seq_list = dict()
		start = time.time()
		for i in keys:
			self.seq_list[i] = generate_sequences(idx_sel_list[i],region_list=self.region_boundary)
			print(len(self.seq_list[i]))
			self.output_generate_sequences(idx_sel_list[i],self.seq_list[i])

		stop = time.time()
		print('generate_sequences', stop-start)

		# generate initial state index
		self.init_id = dict()
		self.init_index(keys)

		# training and validation data
		# x_train1_trans = self.x_train1_trans
		for i in keys:
			idx = self.idx_list[i]
			if self.method<5 or self.method in [56]:
				self.x[i] = x_train1_trans[idx]
				self.y[i] = self.y_signal[i]
				print(self.x[i].shape, self.y[i].shape)
			else:
				idx_sel_list = self.train_sel_list[idx]
				start = time.time()
				x, y, self.vec[i], self.vec_local[i] = sample_select2a1(x_train1_trans[idx],self.y_signal[i],
																idx_sel_list, self.seq_list[i], self.tol, self.flanking)
				stop = time.time()
				print('sample_select2a1',stop-start)

				# concate context for baseline methods
				if self.method<=10:
					# x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.2, random_state=42)
					x = x.reshape(x.shape[0],x.shape[1]*x.shape[-1])
					y = y[:,self.flanking]

				self.x[i], self.y[i] = x, y
				print(self.x[i].shape, self.y[i].shape)

		return True

	# prepare data from predefined features
	def prep_data_1(self,path1,file_prefix,type_id2,feature_dim_transform,
						n_fold=5, ratio=0.9, type_id=1):
		
		x_train1_trans, train_sel_list_ori = self.prep_data_sub1(path1,file_prefix,type_id2,feature_dim_transform)
		print(train_sel_list_ori)

		id1 = mapping_Idx(train_sel_list_ori[:,1],self.serial)
		id2 = (id1>=0)
		print('mapping',len(self.serial),np.sum(id2))

		self.chrom, self.start, self.stop, self.serial, self.signal = self.chrom[id2], self.start[id2], self.stop[id2], self.serial[id2], self.signal[id2]
		
		id1 = id1[id2]
		train_sel_list_ori = train_sel_list_ori[id1]
		self.x_train1_trans = self.x_train1_trans[id1]

		print(train_sel_list_ori.shape,self.x_train1_trans.shape)
		id_vec = self.generate_index_2(train_sel_list_ori, n_fold=n_fold, ratio=ratio, type_id=type_id)

		return id_vec

	def find_serial_ori_1_local(self,chrom_vec,type_id2=1):

		# filename1 = 'mm10_%d_%s_encoded1.h5'%(self.config['cell_type1'],chrom_id1)
		self.species_id = 'mm10'
		self.cell_type1 = self.config['cell_type1']
		file_path1 = '/work/magroup/yy3/data1/replication_timing3/mouse'
		# filename1 = '%s/mm10_5k_seq_genome%d_1.txt'%(file_path1,self.config['cell_type1'])

		chrom_id1 = 'chr1'
		filename1 = '%s_%d_%s_encoded1.h5'%(self.species_id,self.cell_type1,chrom_id1)

		list1, list2 = [], []
		serial_vec = []

		print(filename1)
		if os.path.exists(filename1)==False:
			# prepare data from predefined features
			# one hot encoded feature vectors for each chromosome
			self.prep_data_sequence_ori()
			print('prep_data_sequence_ori',filename1)

		for chrom_id in chrom_vec:
			# if chrom_id<22:
			# 	continue
			chrom_id1 = 'chr%s'%(chrom_id)

			# if self.config['species_id']==0:
			# 	filename2 = 'mm10_%d_%s_encoded1.h5'%(self.config['cell_type1'],chrom_id1)
			# else:
			# 	filename2 = '%s_%s_encoded1.h5'%(self.species_id,chrom_id1)
						
			filename2 = '%s_%d_%s_encoded1.h5'%(self.species_id,self.cell_type1,chrom_id1)
			with h5py.File(filename2,'r') as fid:
				serial1 = fid["serial"][:]
				if type_id2==1:
					seq1 = fid["vec"][:]
					list2.extend(seq1)
				list1.extend([chrom_id]*len(serial1))
				serial_vec.extend(serial1)
				print(chrom_id,len(serial1))

		list1, serial_vec = np.asarray(list1), np.asarray(serial_vec)
		serial_vec = np.hstack((list1[:,np.newaxis],serial_vec))
		f_mtx = np.asarray(list2)

		# data_1 = pd.read_csv(filename1,sep='\t')
		# colnames = list(data_1)
		# local_serial = np.asarray(data_1['serial'])
		# local_seq = np.asarray(data_1['seq'])	
		# print('local_seq', local_seq.shape)

		# serial_vec = local_serial
		# f_mtx = local_seq

		# filename2 = '%s/mm10_5k_serial.bed'%(file_path1)
		# file2 = pd.read_csv(filename2,header=None,sep='\t')
		# ref_chrom, ref_start, ref_stop, ref_serial = np.asarray(file2[0]), np.asarray(file2[1]), np.asarray(file2[2]), np.asarray(file2[3])

		# # assert list(local_serial==list(ref_serial))

		# id_vec1 = []
		# for chrom_id in chrom_vec:
		# 	# if chrom_id<22:
		# 	# 	continue
		# 	# chrom_id1 = 'chr%s'%(chrom_id)
		# 	id1 = np.where(ref_chrom=='chr%d'%(chrom_id))[0]
		# 	id_vec1.extend(id1)
		# 	print(chrom_id,len(id1))

		# id_vec1 = np.asarray(id_vec1)
		# ref_chrom_1, ref_serial_1 = ref_chrom[id_vec1], ref_serial[id_vec1]
		# print('ref chrom local', len(ref_chrom_1), len(ref_serial_1))

		# id1 = utility_1.mapping_Idx(ref_serial_1,local_serial)
		# id2 = np.where(id1>=0)[0]
		# id1 = id1[id2]
		# # assert len(id2)==len(id1)

		# chrom1 = ref_chrom_1[id1]
		# local_chrom = [int(chrom1[3:]) for chrom1 in ref_chrom_1]
		# local_chrom = np.asarray(local_chrom)
		# local_serial, local_seq = local_serial[id2], local_seq[id2]

		# serial_vec = np.column_stack((local_chrom,local_serial))
		# f_mtx = np.asarray(local_seq)

		return serial_vec, f_mtx

	# find serial and feature vectors
	# input: type_id1: load sequence feature or kmer frequency feature, motif feature
	#		 type_id2: load serial or feature vectors
	def find_serial_ori_1(self,file_path,file_prefix,chrom_vec,type_id1=0,type_id2=0,select_config={}):

		# load the sequences
		if type_id1==0:
			# list2 = np.zeros((interval,region_unit_size,4),dtype=np.int8)
			filename1 = '%s_serial_2.txt'%(self.species_id)
			list1, list2 = [], []
			serial_vec = []
				
			if (os.path.exists(filename1)==False) or (type_id2==1):

				if self.config['species_id']==0:
					serial_vec, list2 = self.find_serial_ori_1_local(chrom_vec)

				else:
					for chrom_id in chrom_vec:
						# if chrom_id<22:
						# 	continue
						chrom_id1 = 'chr%s'%(chrom_id)
						
						filename2 = '%s_%s_encoded1.h5'%(self.species_id,chrom_id1)
						with h5py.File(filename2,'r') as fid:
							serial1 = fid["serial"][:]
							if type_id2==1:
								seq1 = fid["vec"][:]
								list2.extend(seq1)
							list1.extend([chrom_id]*len(serial1))
							serial_vec.extend(serial1)
							print(chrom_id,len(serial1))
					
					list1, serial_vec = np.asarray(list1), np.asarray(serial_vec)
					serial_vec = np.hstack((list1[:,np.newaxis],serial_vec))
					
				np.savetxt(filename1,serial_vec,fmt='%d',delimiter='\t')
			else:
				serial_vec = np.loadtxt(filename1,dtype=np.int64)

			if serial_vec.shape[-1]>2:
				cnt1 = serial_vec[:,-1]
				b1 = np.where(cnt1>0)[0]
				ratio1 = len(b1)/len(serial_vec)
				print('sequence with N', len(b1),len(serial_vec),ratio1)
			# serial_vec = serial_vec[:,0]
			f_mtx = np.asarray(list2)

		elif type_id1==2:

			filename1 = select_config['input_filename1']
			layer_name = select_config['layer_name']

			with h5py.File(filename1,'r') as fid:
				f_mtx = np.asarray(fid[layer_name][:],dtype=np.float32)
				print(f_mtx.shape)
				
				serial_vec = fid["serial"][:]
				assert len(serial_vec )==f_mtx.shape[0]
				print(serial_vec[0:5])

		else:
			# load kmer frequency features and motif features
			load_type_id2 = 0
			x_train1_trans, train_sel_list_ori = self.prep_data_sub1(file_path,file_prefix,load_type_id2,self.feature_dim_transform,load_type=1)
			# serial_vec = train_sel_list_ori[:,1]
			serial_vec = np.asarray(train_sel_list_ori)
			f_mtx = np.asarray(x_train1_trans)

		return serial_vec, f_mtx

	def find_serial_ori(self,file_path,file_prefix,type_id1=0,type_id2=0,select_config={}):

		chrom_vec = np.unique(self.chrom)

		chrom_vec1 = []
		for chrom_id in chrom_vec:
			try:
				id1 = chrom_id.find('chr')
				if id1>=0:
					chrom_id1 = int(chrom_id[3:])
					chrom_vec1.append(chrom_id1)
			except:
				continue

		chrom_vec1 = np.sort(chrom_vec1)
		serial_vec, f_mtx = self.find_serial_ori_1(file_path,file_prefix,chrom_vec1,
													type_id1=type_id1,type_id2=type_id2,
													select_config=select_config)
		self.serial_vec = serial_vec
		self.f_mtx = f_mtx

		# list2 = np.zeros((interval,region_unit_size,4),dtype=np.int8)
		print(len(self.chrom),len(self.serial))
		# cnt1 = serial_vec[:,1]
		# b1 = np.where(cnt1>0)[0]
		# ratio1 = len(b1)/len(serial_vec)
		# print(len(b1),len(serial_vec),ratio1)
		id1 = mapping_Idx(serial_vec[:,1],self.serial)
		b1 = np.where(id1>=0)[0]

		self.local_serial_1(b1,type_id=0)

		print(len(self.chrom),len(self.serial))

		return True

	def prep_data_2(self,file_path,file_prefix,seq_len_thresh=50):

		self.find_serial_ori(file_path,file_prefix)

		chrom_vec = np.unique(self.chrom)
		chrom_vec1 = []
		for chrom_id in chrom_vec:
			try:
				id1 = chrom_id.find('chr')
				if id1>=0:
					chrom_id1 = int(chrom_id[3:])
					chrom_vec1.append(chrom_id1)
			except:
				continue

		chrom_vec1 = np.sort(chrom_vec1)
		sample_num = len(self.chrom)
		idx_sel_list = -np.ones((sample_num,2),dtype=np.int64)

		for chrom_id in chrom_vec1:
			chrom_id1 = 'chr%d'%(chrom_id)
			b1 = np.where(self.chrom==chrom_id1)[0]
			idx_sel_list[b1,0] = [chrom_id]*len(b1)
			idx_sel_list[b1,1] = self.serial[b1]

		id1 = idx_sel_list[:,0]>=0
		idx_sel_list = idx_sel_list[id1]
		sample_num = len(id1)
		y = self.signal[id1]
		x_mtx = idx_sel_list[id1]

		seq_list = generate_sequences(idx_sel_list, gap_tol=5, region_list=[])
		seq_len = seq_list[:,1]-seq_list[:,0]+1
		thresh1 = seq_len_thresh
		b1 = np.where(seq_len>thresh1)[0]
		print(len(seq_list),len(b1))
		seq_list = seq_list[b1]
		seq_len1 = seq_list[:,1]-seq_list[:,0]+1
		print(sample_num,np.sum(seq_len1),seq_list.shape,np.max(seq_len),np.min(seq_len),np.median(seq_len),np.max(seq_len1),np.min(seq_len1),np.median(seq_len1))
		self.output_generate_sequences(idx_sel_list,seq_list)

		t_mtx, signal_mtx, vec1_serial, vec1_local = sample_select2a1(x_mtx, y, idx_sel_list, seq_list, tol=self.tol, L=self.flanking)

		t_serial = vec1_serial[:,self.flanking]
		context_size = vec1_serial.shape[1]
		id1 = mapping_Idx(idx_sel_list[:,1],t_serial)
		b1 = np.where(id1>=0)[0]
		if len(b1)!=len(vec1_serial):
			print('error!',len(b1),len(vec1_serial))
			return -1
		sel_id1 = id1[b1]
		# idx_sel_list1 = idx_sel_list[sel_id1]
		# label1 = y[sel_id1]
		t_chrom = idx_sel_list[sel_id1,0]
		print(t_chrom,t_serial)
		print(t_chrom.shape,t_serial.shape)
		print(vec1_serial.shape)

		list_ID = []
		cnt1 = 0
		interval = 200
		list1, list2 = [],[]
		list3 = []
		# region_unit_size = 5000
		# list2 = np.zeros((interval,region_unit_size,4),dtype=np.int8)
		for chrom_id in chrom_vec1:
			# if chrom_id<22:
			# 	continue
			chrom_id1 = 'chr%s'%(chrom_id)
			filename1 = '%s_%s_encoded1.h5'%(self.species_id,chrom_id1)
			t_id1 = np.where(t_chrom==chrom_id)[0]
			t_serial1 = t_serial[t_id1]	# serial by chromosome
			sample_num1 = len(t_serial1)

			num_segment = np.int(np.ceil(sample_num1/interval))
			print(chrom_id1,num_segment,interval,sample_num1)

			with h5py.File(filename1,'r') as fid:
				serial1 = fid["serial"][:]
				seq1 = fid["vec"][:]
				serial1 = serial1[:,0]
				print(serial1.shape, seq1.shape)
				id1 = utility_1.mapping_Idx(serial1,t_serial1)
				id2 = np.where(id1>=0)[0]
				num1 = len(id2)
				segment_id = 0

				t_signal_mtx = signal_mtx[t_id1[id2]]
				list3.extend(t_signal_mtx)

				for i in range(num1):
					cnt2 = i+1
					t_id2 = id2[i]
					label_serial = t_serial1[t_id2]
					t_vec1_serial = vec1_serial[t_id1[t_id2]]
					id_1 = mapping_Idx(serial1,t_vec1_serial)
					b1 = np.where(id_1>=0)[0]
					if len(b1)!=context_size:
						b2 = np.where(id_1<0)[0]
						print('error!',chrom_id1,label_serial,t_vec1_serial[b2],len(b1),context_size)
						np.savetxt('temp1.txt',serial1,fmt='%d',delimiter='\t')
						np.savetxt('temp2.txt',t_vec1_serial,fmt='%d',delimiter='\t')
						return -1
					t_mtx = seq1[id_1[b1]]
					list1.append(t_vec1_serial)
					list2.append(t_mtx)

					local_id = cnt2%interval
					label_id = cnt1
					output_filename = 'test1_%s_%s_%d.h5'%(self.cell,chrom_id1,segment_id)
					if (cnt2%interval==0) or (cnt2==num1):
						output_filename1 = '%s/%s'%(file_path,output_filename)
						list1 = np.asarray(list1)
						list2 = np.asarray(list2,dtype=np.int8)
						print(chrom_id1,segment_id,local_id,label_id,label_serial,list1.shape,list2.shape)
						# with h5py.File(output_filename1,'w') as fid:
						# 	fid.create_dataset("serial", data=list1, compression="gzip")
						# 	fid.create_dataset("vec", data=list2, compression="gzip")
						# dict1 = {'serial':list1.tolist(),'vec':list2.tolist()}
						# np.save(output_filename,dict1,allow_pickle=True)
						# with open(output_filename, "w") as fid: 
						# 	json.dump(dict1,fid)
						# with open(output_filename,"w",encoding='utf-8') as fid:
						# 	json.dump(dict1,fid,separators=(',', ':'), sort_keys=True, indent=4)
						list1, list2 = [], []
						segment_id += 1

					cnt1 = cnt1+1
					list_ID.append([label_id,label_serial,output_filename,local_id])
					# if cnt2%interval==0:
					# 	break

		# with open(output_filename, "r") as fid: 
		# 	dict1 = json.load(fid)
		# 	serial1, vec1 = np.asarray(dict1['serial']), np.asarray(dict1['vec'])
		# 	print(serial1.shape,vec1.shape)
		
		# with h5py.File(output_filename1,'r') as fid:
		# 	serial1 = fid["serial"][:]
		# 	vec1 = fid["vec"][:]
		# 	print(serial1.shape,vec1.shape)
			
		fields = ['label_id','label_serial','filename','local_id']
		list_ID = np.asarray(list_ID)
		data1 = pd.DataFrame(columns=fields,data=list_ID)
		output_filename = '%s/%s_label_ID_1'%(file_path,self.cell)
		data1.to_csv(output_filename+'.txt',index=False,sep='\t')
		# np.save(output_filename,data1,allow_pickle=True)
		output_filename = '%s/%s_label.h5'%(file_path,self.cell)
		list3 = np.asarray(list3)
		print(list3.shape)
		with h5py.File(output_filename,'w') as fid:
			fid.create_dataset("vec", data=np.asarray(list3), compression="gzip")

		return list_ID

	# find serial for training and validation data
	def prep_data_2_sub1(self,file_path,file_prefix,type_id1=0,type_id2=0,gap_tol=5,seq_len_thresh=5,select_config={}):

		if type_id1>=0:
			self.find_serial_ori(file_path,file_prefix,
									type_id1=type_id1,type_id2=type_id2,
									select_config=select_config)

		chrom_vec = np.unique(self.chrom)
		chrom_vec1 = []
		for chrom_id in chrom_vec:
			try:
				id1 = chrom_id.find('chr')
				if id1>=0:
					chrom_id1 = int(chrom_id[3:])
					chrom_vec1.append(chrom_id1)
			except:
				continue

		chrom_vec1 = np.sort(chrom_vec1)
		sample_num = len(self.chrom)
		idx_sel_list = -np.ones((sample_num,2),dtype=np.int64)
		if 'gap_thresh' in self.config:
			gap_tol = self.config['gap_thresh']

		if 'seq_len_thresh' in self.config:
			seq_len_thresh = self.config['seq_len_thresh']

		for chrom_id in chrom_vec1:
			chrom_id1 = 'chr%d'%(chrom_id)
			b1 = np.where(self.chrom==chrom_id1)[0]
			idx_sel_list[b1,0] = [chrom_id]*len(b1)
			idx_sel_list[b1,1] = self.serial[b1]

		id1 = idx_sel_list[:,0]>=0
		idx_sel_list = idx_sel_list[id1]
		sample_num = len(id1)
		y = self.signal[id1]
		x_mtx = idx_sel_list[id1]

		self.train_sel_list_ori = idx_sel_list
		self.y_signal_1 = self.signal[id1]
		ref_serial = idx_sel_list[:,1]

		# train_sel_list, val_sel_list = train1_sel_list[idx_train], train1_sel_list[idx_valid]
		# self.idx_list.update({'train':train_id1[idx_train],'valid':train_id1[idx_valid]})
		# self.idx_train_val = {'train':idx_train,'valid':idx_valid}
		# self.y_signal.update({'train':y_signal_train1[idx_train],'valid':y_signal_train1[idx_valid]})
		train_sel_list, val_sel_list, test_sel_list = self.prep_training_test(idx_sel_list)
		print(len(train_sel_list),len(val_sel_list),len(test_sel_list))
		
		keys = ['train','valid','test']
		# keys = ['train','valid']
		#  self.idx_sel_list = {'train':train1_sel_list,'valid':val_sel_list,'test':test_sel_list}
		self.idx_sel_list_ori = {'train':train_sel_list,'valid':val_sel_list,'test':test_sel_list}
		# self.idx_sel_list = idx_sel_list

		# seq_list_train, seq_list_valid: both locally calculated
		self.seq_list = dict()
		start = time.time()
		# seq_len_thresh = 20
		self.local_serial_dict = dict()
		for i in keys:
			# self.seq_list[i] = generate_sequences(idx_sel_list1[i],region_list=self.region_boundary)
			# print(len(self.seq_list[i]))
			# self.output_generate_sequences(idx_sel_list[i],self.seq_list[i])
			idx_sel_list1 = self.idx_sel_list_ori[i]
			# region_list_id = 'region_list_%s'%(i)
			# if region_list_id in self.config:
			# 	region_list = self.config[region_list_id]
			# else:
			# 	region_list = []
			# region_list = np.asarray(region_list)
			# print(region_list_id,region_list)

			# if i=='test':
			# 	region_boundary = self.region_boundary
			# else:
			# 	region_boundary = []

			region_boundary = self.region_boundary
			print('region_boundary',region_boundary)
			# assert len(region_boundary)==0
			seq_list = generate_sequences(idx_sel_list1, gap_tol=gap_tol, region_list=region_boundary)
			# seq_len = seq_list[:,1]-seq_list[:,0]+1
			# thresh1 = seq_len_thresh
			# b1 = np.where(seq_len>thresh1)[0]
			# print(len(seq_list),len(b1))
			# seq_list = seq_list[b1]
			# seq_len1 = seq_list[:,1]-seq_list[:,0]+1
			# print(sample_num,np.sum(seq_len1),len(seq_list),np.max(seq_len),np.min(seq_len),np.median(seq_len),np.max(seq_len1),np.min(seq_len1),np.median(seq_len1))

			# reselect the regions according to the subsequence length
			# recalculate seq_list
			idx_sel_list1, seq_list = self.select_region_local_1(idx_sel_list1,seq_list, 
																gap_tol=gap_tol, 
																seq_len_thresh=seq_len_thresh, 
																region_list=[])

			self.idx_sel_list_ori[i] = idx_sel_list1
			self.seq_list[i] = seq_list

			x1 = idx_sel_list1
			sel_id = utility_1.mapping_Idx(ref_serial,idx_sel_list1[:,1])
			y1 = self.y_signal_1[sel_id]
			x, y, t_vec_serial, t_vec_local = sample_select2a1(x1,y1,
															idx_sel_list1, seq_list, self.tol, self.flanking)

			t_serial1 = t_vec_serial[:,self.flanking]
			# if np.sum(t_serial1!=sel_idx_list1[:,1])>0:
			# 	print('error!',i)
			# 	return
			id1 = utility_1.mapping_Idx(idx_sel_list1[:,1],t_serial1)
			b1 = np.where(id1>=0)[0]
			if len(b1)!=len(t_serial1):
				print('error!',i)
				return

			idx_sel_list1 = idx_sel_list1[id1[b1]]
			self.local_serial_dict[i] = [idx_sel_list1,y1,y,t_vec_serial,t_vec_local]
			print(i,t_serial1.shape,y.shape)

		stop = time.time()
		print('generate_sequences', stop-start)

		return self.local_serial_dict

	# load feature
	def load_feature_local(self,chrom_vec,type_id=0,select_config={}):

		# load sequences
		if type_id==0:
			serial_vec = []
			list1, list2 = [],[]
			# list2 = np.zeros((interval,region_unit_size,4),dtype=np.int8)

			if self.config['species_id']==0:
				serial_vec, f_mtx = self.find_serial_ori_1_local(chrom_vec)

			else:
				for chrom_id in chrom_vec:
					# if chrom_id<22:
					# 	continue
					chrom_id1 = 'chr%s'%(chrom_id)
					filename1 = '%s_%s_encoded1.h5'%(self.species_id,chrom_id1)
					with h5py.File(filename1,'r') as fid:
						serial1 = fid["serial"][:]
						seq1 = fid["vec"][:]
						serial_vec.extend(serial1)
						list1.extend([chrom_id]*len(serial1))
						list2.extend(seq1)
						print(len(serial1),seq1.shape)

				list1 = np.asarray(list1)
				serial_vec = np.hstack((list1[:,np.newaxis],serial_vec))
				f_mtx = np.asarray(list2)

		# kmer frequency and motif feature
		elif type_id==1:
			if len(self.serial_vec)>0 and (len(self.f_mtx)>0):
				serial_vec = self.serial_vec
				f_mtx = self.f_mtx

			else:
				type_id2 = 0
				x_train1_trans, train_sel_list_ori = self.prep_data_sub1(self.file_path,self.file_prefix,type_id2,self.feature_dim_transform,load_type=1)
				# serial_vec = train_sel_list_ori[:,1]
				serial_vec = np.asarray(train_sel_list_ori)
				f_mtx = np.asarray(x_train1_trans)

		else:

			filename1 = select_config['input_filename1']
			layer_name = select_config['layer_name']

			with h5py.File(filename1,'r') as fid:
				f_mtx = np.asarray(fid[layer_name][:],dtype=np.float32)
				print(f_mtx.shape)
				
				serial_vec = fid["serial"][:]
				assert len(serial_vec )==f_mtx.shape[0]
				print(serial_vec[0:5])

		return serial_vec, f_mtx

	# find serial
	def find_serial_local(self,ref_serial,vec_serial_ori,sel_id):

		serial_1 = vec_serial_ori[:,self.flanking]
		# print(len(ref_serial),ref_serial)
		# print(len(serial_1),serial_1)

		assert np.max(np.abs(ref_serial-serial_1))==0
		
		t_vec_serial = np.ravel(vec_serial_ori[sel_id])
		serial1 = np.unique(t_vec_serial)
		id1 = mapping_Idx(ref_serial,serial1)

		b1 = np.where(id1<0)[0]
		if len(b1)>0:
			print('error!',len(b1))
			print(serial1[b1])

		b_1 = np.where(id1>=0)[0]
		id1 = id1[b_1]
		sample_num = len(ref_serial)
		id2 = np.setdiff1d(np.arange(sample_num),id1)
		if len(id2)>0:
			t_serial2 = ref_serial[id2]
			id_2 = mapping_Idx(serial_1,t_serial2)
			sel_id = list(sel_id)+list(id_2)

		sel_id = np.unique(sel_id)

		print('find serial local',len(sel_id),len(id_2))

		return sel_id

	# load training and validation data
	def prep_data_2_sub2(self,type_id1=0,keys=['train','valid'],stride=1,type_id=0,select_config={}):

		chrom1 = []
		for i in range(0,len(keys)):
			key1 = keys[i]
			idx_sel_list, y_ori, y, vec_serial, vec_local = self.local_serial_dict[key1]
			chrom1.extend(idx_sel_list[:,0])
			
		chrom_vec1 = np.sort(np.unique(chrom1))

		serial_vec, f_mtx = self.load_feature_local(chrom_vec1,type_id=type_id1,select_config=select_config)
		
		print('load feature local', serial_vec.shape, f_mtx.shape)

		if serial_vec.shape[1]>2:
			cnt1 = serial_vec[:,-1]
			b1 = np.where(cnt1>0)[0]
			ratio1 = len(b1)/len(serial_vec)
			print(len(b1),len(serial_vec),ratio1)

		ref_serial = serial_vec[:,1]

		for i in range(0,len(keys)):
			key1 = keys[i]
			idx_sel_list, y_ori, y, vec_serial, vec_local = self.local_serial_dict[key1]
			num1 = len(idx_sel_list)

			if stride>1:
				id1 = list(range(0,num1,stride))

				# the windows cover the positions
				print(num1,stride)
				if type_id==1:
					id1 = self.find_serial_local(idx_sel_list[:,1],vec_serial,id1)
				
				y, vec_serial, vec_local = y[id1], vec_serial[id1], vec_local[id1]
				self.local_serial_dict[key1] = [idx_sel_list, y_ori, y, vec_serial, vec_local]

			id2 = mapping_Idx(ref_serial,idx_sel_list[:,1])
			print(key1,len(ref_serial),len(idx_sel_list))
			print(ref_serial[0:5])
			print(idx_sel_list[0:5,1])

			b1 = np.where(id2<0)[0]
			if len(b1)>0:
				print('error!',len(b1),key1)
				# return 
			print('mapping',len(id2))

			# update
			b_1 = np.where(id2>=0)[0]
			id2 = id2[b_1]
			idx_sel_list, y_ori = idx_sel_list[b_1], y_ori[b_1]
			y, vec_serial, vec_local = y[b_1], vec_serial[b_1], vec_local[b_1]
			self.local_serial_dict[key1] = [idx_sel_list, y_ori, y, vec_serial, vec_local]

			self.x[key1] = f_mtx[id2]
			self.idx[key1] = id2

		return True

	# training and predition with sequences
	def control_pre_test1_repeat(self,path1,file_prefix,run_id_load=-1):

		self.prep_data_2_sub1(path1,file_prefix)

		config = self.config.copy()
		units1=[50,50,32,25,50,25,0,0]
		flanking = 50
		context_size = 2*flanking+1
		n_step_local_ori = 5000
		region_unit_size = 1
		feature_dim = 4
		local_conv_list1 = []
		regularizer2, bnorm, activation = 1e-04, 1, 'relu'

		if self.run_id==110001:
			config_vec1 = [[64, 15, 5, 1, 2, 2, 0.2, 0],
							[32, 5, 1, 1, 10, 10, 0.2, 0],
							[32, 3, 1, 1, 5, 5, 0.2, 0]]

		for t1 in config_vec1:
			n_filters, kernel_size1, stride, dilation_rate1, pool_length1, stride1, drop_out_rate, boundary = t1
			conv_1 = [n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate]
			local_conv_list1.append(conv_1)

		config['local_conv_list1'] = local_conv_list1
		print(local_conv_list1)

		feature_dim1, feature_dim2, return_sequences_flag1, sample_local, pooling_local = 32, 25, True, 0, 0
		n_step_local1 = 10
		feature_dim3 = []
		local_vec_1 = [feature_dim1, feature_dim2, feature_dim3, return_sequences_flag1, sample_local, pooling_local]
		attention2_local = 0
		select2 = 1
		concatenate_1, concatenate_2 = 0, 1
		hidden_unit = 32
		regularizer2_2 = 1e-04
		config.update({'attention1':0,'attention2':1,'select2':select2,'context_size':context_size,'n_step_local':n_step_local1,'n_step_local_ori':n_step_local_ori})
		config.update({'local_vec_1':local_vec_1,'attention2_local':attention2_local})

		config['feature_dim_vec'] = units1[2:]
		config['feature_dim_vec_basic'] = units1[2:]
		config.update({'local_conv_list1':local_conv_list1,'local_vec_1':local_vec_1})
		config.update({'attention1':0,'attention2':1,'context_size':context_size,
									'n_step_local_ori':n_step_local_ori})
		config.update({'select2':select2,'attention2_local':attention2_local})
		config.update({'concatenate_1':concatenate_1,'concatenate_2':concatenate_2})
		config.update({'feature_dim':feature_dim,'output_dim':hidden_unit,'regularizer2_2':regularizer2_2})

		model = utility_1.get_model2a1_attention_1_2_2_sample5(config)

		# find feature vectors with the serial
		self.x = dict()
		self.idx = dict()
		self.prep_data_2_sub2(type_id1=0,keys=['train','valid'],stride=1)

		mtx_train = self.x['train']
		idx_sel_list_train, y_train_ori_1, y_train_ori, vec_serial_train, vec_local_train = self.local_serial_dict['train']

		mtx_valid = self.x['valid']
		idx_sel_list_valid, y_valid_ori_1, y_valid_ori, vec_serial_valid, vec_local_valid = self.local_serial_dict['valid']

		train_num1, valid_num1 = len(y_train_ori), len(y_valid_ori)
		print('train',len(idx_sel_list_train),len(y_train_ori),mtx_train.shape)
		print('valid',len(idx_sel_list_valid),len(y_valid_ori),mtx_valid.shape)

		x_valid = mtx_valid[vec_local_valid]
		y_valid = y_valid_ori

		print(x_valid.shape,y_valid.shape)

		type_id2 = 2
		MODEL_PATH = 'test%d.h5'%(self.run_id)
		n_epochs = 1
		BATCH_SIZE = 32
		n_step_local = n_step_local_ori

		earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

		num_sample1 = 1
		interval = 2500
		select_num = np.int(np.ceil(train_num1/interval))
		# select_num1 = select_num*interval
		# print(num_sample1,select_num,interval,select_num1)
		if select_num>1:
			t1 = np.arange(0,train_num1,interval)
			pos = np.vstack((t1,t1+interval)).T
			pos[-1][1] = train_num1
			print(train_num1,select_num,interval)
			print(pos)
		else:
			pos = [[0,train_num1]]

		start2 = time.time()
		train_id_1 = np.arange(train_num1)
		valid_id_1 = np.arange(valid_num1)
		np.random.shuffle(valid_id_1)
		cnt1 = 0
		mse1 = 1e5
		decay_rate = 0.95
		decay_step = 1
		init_lr = self.config['lr']

		for i1 in range(50):

			self.config['lr'] = init_lr*((decay_rate)**(int(i1/decay_step)))
			np.random.shuffle(train_id_1)
			
			start1 = time.time()

			valid_num2 = 2500
			num2 = np.min([valid_num1,valid_num2])
			valid_id2 = valid_id_1[0:num2]
			x_valid1, y_valid1 = x_valid[valid_id2], y_valid[valid_id2]

			for l in range(select_num):
				s1, s2 = pos[l]
				print(l,s1,s2)
				sel_id = train_id_1[s1:s2]

				x_train = mtx_train[vec_local_train[sel_id]]
				y_train = y_train_ori[sel_id]

				x_train, y_train = np.asarray(x_train), np.asarray(y_train)
				print(x_train.shape,y_train.shape)
				n_epochs = 1

				train_num = x_train.shape[0]
				print('x_train, y_train', x_train.shape, y_train.shape)
				print('x_valid, y_valid', x_valid1.shape, y_valid1.shape)
				
				# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
				model.fit(x_train,y_train,epochs = n_epochs, batch_size = BATCH_SIZE, validation_data = [x_valid1,y_valid1],
									callbacks=[earlystop,checkpointer])
				
				# model.load_weights(MODEL_PATH)
				model_path2 = '%s/model_%d_%d_%d_%d.h5'%(self.path,self.run_id,type_id2,context_size,i1)
				model.save(model_path2)
				# model_path2 = MODEL_PATH
				if l%5==0:
					print('loading weights... ', MODEL_PATH)
					model.load_weights(MODEL_PATH) # load model with the minimum training error
					y_predicted_valid1 = model.predict(x_valid)
					y_predicted_valid = np.ravel(y_predicted_valid1[:,flanking])
					temp1 = score_2a(np.ravel(y_valid[:,flanking]), y_predicted_valid)
					print(temp1)

				print('loading weights... ', model_path2)
				model.load_weights(model_path2) # load model with the minimum training error

			print('loading weights... ', model_path2)
			model.load_weights(model_path2) # load model with the minimum training error

			y_predicted_valid1 = model.predict(x_valid)
			y_predicted_valid = np.ravel(y_predicted_valid1[:,flanking])
			temp1 = score_2a(np.ravel(y_valid[:,flanking]), y_predicted_valid)
			print([i1,l]+list(temp1))
			t_mse1 = temp1[0]

			if np.abs(t_mse1-mse1)<self.min_delta:
				cnt1 += 1
			else:
				cnt1 = 0

			if t_mse1 < mse1:
				mse1 = t_mse1

			if cnt1>=self.step:
				break

		stop1 = time.time()
		print(stop1-start1)

		print('loading weights... ', MODEL_PATH)
		model.load_weights(MODEL_PATH) # load model with the minimum training error
		y_predicted_valid1 = model.predict(x_valid)
		y_predicted_valid = np.ravel(y_predicted_valid1[:,flanking])
		temp1 = score_2a(np.ravel(y_valid[:,flanking]), y_predicted_valid)
		print(temp1)

		self.config.update({'model_path1':MODEL_PATH})

		return model

	def config_pre_3(self,sel_conv_id=1,kernel_size2=3):

		if sel_conv_id==0:
			config_vec1 = [[64, 15, 10, 1, 2, 2, 0.2, 0],
							[32, 5, 1, 1, 10, 10, 0.2, 0],
							[32, 3, 1, 1, 5, 5, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0]]

			config_vec2 = [[25, kernel_size2, 1, 1, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 2, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 4, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 8, 1, 1, 0.2, 0]]

		elif sel_conv_id==1:
			# run_id: 100011
			config_vec1 = [[64, 15, 10, 1, 2, 2, 0.2, 0],
							[50, 5, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0]]

			config_vec2 = [[25, kernel_size2, 1, 1, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 2, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 4, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 8, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 16, 1, 1, 0.2, 0]]

		elif sel_conv_id==2:
			# run_id: 101001
			config_vec1 = [[64, 15, 5, 1, 2, 2, 0.2, 0],
							[50, 5, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0]]

			config_vec2 = [[25, kernel_size2, 1, 1, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 2, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 4, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 8, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 16, 1, 1, 0.2, 0]]

		elif sel_conv_id==3:
			# run_id: 110001
			config_vec1 = [[64, 15, 10, 1, 2, 2, 0.2, 0],
							[50, 5, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0]]

			config_vec2 = [[25, kernel_size2, 1, 1, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 2, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 4, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 8, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 16, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 32, 1, 1, 0.2, 0]]

		elif sel_conv_id==4:
			# run_id: 510101
			config_vec1 = [[64, 15, 10, 1, 2, 2, 0.2, 0],
							[50, 5, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0]]

			# run_id: 510501
			config_vec2 = [[25, kernel_size2, 1, 1, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 2, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 4, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 8, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 16, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 32, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 64, 1, 1, 0.2, 0]]

		elif sel_conv_id==5:
			config_vec1 = [[128, 15, 5, 1, 2, 2, 0.2, 0],
							[50, 5, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0]]

			config_vec2 = [[25, kernel_size2, 1, 1, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 2, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 4, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 8, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 16, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 32, 1, 1, 0.2, 0]]

		elif sel_conv_id==6:
			config_vec1 = [[128, 15, 5, 1, 2, 2, 0.2, 0],
							[50, 5, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0]]

			config_vec2 = [[25, kernel_size2, 1, 1, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 2, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 4, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 8, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 16, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 32, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 64, 1, 1, 0.2, 0]]

		elif sel_conv_id==7:
			config_vec1 = [[128, 15, 10, 1, 2, 2, 0.2, 0],
							[50, 5, 1, 1, 10, 10, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0],
							[50, 3, 1, 1, 5, 5, 0.2, 0]]

			config_vec2 = [[25, kernel_size2, 1, 1, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 2, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 4, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 8, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 16, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 32, 1, 1, 0.2, 0]]

		else:
			config_vec1 = [[64, 15, 5, 1, 2, 2, 0.2, 0],
							[32, 5, 1, 1, 10, 10, 0.2, 0],
							[32, 3, 1, 1, 10, 10, 0.2, 0],
							[25, 3, 1, 1, 5, 5, 0.2, 0]]

			config_vec2 = [[25, kernel_size2, 1, 1, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 2, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 4, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 8, 1, 1, 0.2, 0],
							[25, kernel_size2, 1, 16, 1, 1, 0.2, 0]]

		return config_vec1, config_vec2

	# dilated convolution with sequence features (compared methods)
	def control_pre_test1_2(self,path1,file_prefix,model_path1=''):

		self.file_path, self.file_prefix = path1, file_prefix
		self.prep_data_2_sub1(path1,file_prefix,type_id1=1,type_id2=1)

		# x_train1 = np.asarray(np.random.rand(5000,context_size,n_step_local,feature_dim),dtype=np.float32)
		# y_train1 = np.asarray(np.random.rand(5000,context_size,1),dtype=np.float32)

		# find feature vectors with the serial
		self.x = dict()
		self.idx = dict()
		self.prep_data_2_sub2(type_id1=1,keys=['train','valid'],stride=1)

		config = self.config.copy()
		# units1=[50,50,50,50,1,25,25,1]
		# units1=[50,50,50,25,50,25,0,0]
		# n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
		units1=[50,50,50,25,50,25,0,0]
		# config['feature_dim_vec'] = units1[2:]
		# # n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
		# config['feature_dim_vec_basic'] = units1[2:]
		regularizer2, bnorm, activation = 1e-04, 1, 'relu'

		config['attention1']=0
		config['attention2']=1
		config['select2']=1
		print('get_model2a1_attention_1_2_2_sample6_1')
		context_size = 2*self.flanking+1
		config['context_size'] = context_size
		config['feature_dim'] = self.x['train'].shape[-1]

		sel_conv_id = self.config['sel_conv_id']
		if 'dilated_conv_kernel_size2' in self.config:
			kernel_size2 = self.config['dilated_conv_kernel_size2']
		else:
			kernel_size2 = 3
		config_vec1, config_vec2 = self.config_pre_3(sel_conv_id,kernel_size2=kernel_size2)
		print(config_vec1)
		print(config_vec2)
		
		# model = utility_1.get_model2a1_word()

		local_conv_list1, local_conv_list2 = [], []
		for t1 in config_vec2:
			n_filters, kernel_size1, stride, dilation_rate1, pool_length1, stride1, drop_out_rate, boundary = t1
			conv_2 = [n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate]
			local_conv_list2.append(conv_2)

		config['local_conv_list1'] = local_conv_list1
		config['local_conv_list2'] = local_conv_list2
		print(local_conv_list2)

		model = utility_1.get_model2a1_attention_1_2_2_sample6_1(context_size,config)
		# return -1

		mtx_train = self.x['train']
		idx_sel_list_train, y_train_ori_1, y_train_ori, vec_serial_train, vec_local_train = self.local_serial_dict['train']

		mtx_valid = self.x['valid']
		idx_sel_list_valid, y_valid_ori_1, y_valid_ori, vec_serial_valid, vec_local_valid = self.local_serial_dict['valid']

		train_num1, valid_num1 = mtx_train.shape[0], mtx_valid.shape[0]
		print('train',len(idx_sel_list_train),len(y_train_ori),mtx_train.shape)
		print('valid',len(idx_sel_list_valid),len(y_valid_ori),mtx_valid.shape)

		x_train = np.asarray(mtx_train[vec_local_train])
		y_train = np.asarray(y_train_ori)

		x_valid = np.asarray(mtx_valid[vec_local_valid])
		y_valid = np.asarray(y_valid_ori)

		print(x_train.shape,y_train.shape)
		print(x_valid.shape,y_valid.shape)

		type_id2 = 2
		MODEL_PATH = 'test%d.h5'%(self.run_id)
		n_epochs = 100
		# BATCH_SIZE = 64
		BATCH_SIZE = 512

		start1 = time.time()
		if self.train==1:
			print('x_train, y_train', x_train.shape, y_train.shape)
			earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=0, mode='auto')
			checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
			# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
			model.fit(x_train,y_train,epochs = n_epochs, batch_size = BATCH_SIZE, validation_data = [x_valid,y_valid],
							callbacks=[earlystop,checkpointer])
			# model.load_weights(MODEL_PATH)
			model_path2 = '%s/model_%d_%d_%d.h5'%(self.path,self.run_id,type_id2,context_size)
			model.save(model_path2)
			model_path2 = MODEL_PATH
			print('loading weights... ', model_path2)
			model.load_weights(model_path2) # load model with the minimum training error
		else:
			# model_path1 = './mnt/yy3/test_29200.h5'
			if model_path1!="":
				MODEL_PATH = model_path1
			print('loading weights... ', MODEL_PATH)
			model.load_weights(MODEL_PATH)

		self.model = model

		stop1 = time.time()
		print(stop1-start1)

		y_predicted_valid = model.predict(x_valid)
		y_predicted_valid = np.ravel(y_predicted_valid[:,self.flanking])
		valid_score_1 = score_2a(np.ravel(y_valid[:,self.flanking]), y_predicted_valid)
		print(valid_score_1)

		type_id1=1
		type_id=1
		interval=5000
		y_predicted_test, y_test, idx_sel_list_test, predicted_attention_test, score_vec1, score_dict1 = self.test_pre_1(type_id1=type_id1,type_id=type_id,
																												est_attention=0,select_config={},interval=interval)

		output_filename = 'test_%d_%d'%(self.run_id,self.run_id)
		self.output_predict_1(idx_sel_list_test, y_predicted_test, y_test, predicted_attention_test, 
							score_vec1, score_dict1, output_filename, valid_score=valid_score_1)

		return model

	# prediction (compared methods)
	def control_pre_test1_2_predict(self,path1,file_prefix,run_id_load=-1):
		
		if run_id_load<0:
			run_id_load = self.run_id

		config_filename = 'config_%d.npy'%(run_id_load)
		if os.path.exists(config_filename)==True:
			config = np.load(config_filename,allow_pickle=True)
			config = config[()]
		else:
			print('file does not exist',config_filename)
			return -1

		self.x = dict()
		self.idx = dict()
		self.file_path, self.file_prefix = path1, file_prefix
		self.prep_data_2_sub1(path1,file_prefix)

		model = utility_1.get_model2a1_attention_1_2_2_sample6(config)
		# model = utility_1.get_model2a1_word()
		# return -1
		
		model_path1 = self.config['model_path1']
		# epoch_id, local_id = self.config['train_pre_epoch']
		print('loading weights...', model_path1)
		model.load_weights(model_path1)
		self.model = model

		y_predicted_test, y_test, idx_sel_list_test, predicted_attention_test, score_vec1, score_dict1 = self.test_pre_1(type_id1=0,est_attention=0)

		output_filename = 'test_%d_%d'%(self.run_id,run_id_load)
		self.output_predict_1(idx_sel_list_test, y_predicted_test, y_test, predicted_attention_test, 
							score_vec1, score_dict1, output_filename)

		return True

	# convolution with sequences
	def control_pre_test1_3(self,path1,file_prefix,run_id_load=-1):

		self.prep_data_2_sub1(path1,file_prefix)
		self.x = dict()
		self.idx = dict()

		flanking = 50
		context_size = 2*flanking+1
		n_step_local_ori = 5000
		region_unit_size = 1
		feature_dim = 4
		self.config.update({'feature_dim':feature_dim,'n_step_local_ori':n_step_local_ori,'context_size':context_size})

		config_filename = 'config_%d.npy'%(self.run_id)
		flag1 = (os.path.exists(config_filename)==True)
		sel_conv_id = self.config['sel_conv_id']
		if self.train==0:
			if flag1==1:
				config = np.load(config_filename,allow_pickle=True)
				config = config[()]
			else:
				print('config file does not exist', config_filename)
				return -1
		else:
			# sel_conv_id = self.config['sel_conv_id']
			if flag1==1:
				print('previous config file exists', config_filename)
				config = np.load(config_filename,allow_pickle=True)
				config = config[()]
			else:
				config = self.config_pre_1_1(sel_conv_id)
				np.save(config_filename,config,allow_pickle=True)

		print('get_model2a1_convolution')
		model = utility_1.get_model2a1_convolution(config)

		# find feature vectors with the serial
		if self.config['predict_test']==1:
			if self.train==1:
				self.prep_data_2_sub2(type_id1=0,keys=['train','valid','test'],stride=1)
			else:
				self.prep_data_2_sub2(type_id1=0,keys=['test'],stride=1)
		else:
			self.prep_data_2_sub2(type_id1=0,keys=['train','valid'],stride=1)	# type_id1=0, load sequences

		file_path_1 = 'mouse_pred1'
		if self.train==1:
			mtx_train = self.x['train']
			idx_sel_list_train, y_train_ori_1, y_train_ori, vec_serial_train, vec_local_train = self.local_serial_dict['train']

			mtx_valid = self.x['valid']
			idx_sel_list_valid, y_valid_ori_1, y_valid_ori, vec_serial_valid, vec_local_valid = self.local_serial_dict['valid']

			train_num1, valid_num1 = len(y_train_ori), len(y_valid_ori)
			print('train',len(idx_sel_list_train),len(y_train_ori),mtx_train.shape)
			print('valid',len(idx_sel_list_valid),len(y_valid_ori),mtx_valid.shape)

			# x_valid = mtx_valid[vec_local_valid]
			x_valid = mtx_valid
			y_valid = y_valid_ori_1

			# x_valid, y_valid = np.asarray(x_valid), np.asarray(y_valid)
			# print(x_valid.shape,y_valid.shape)

			type_id2 = 2
			MODEL_PATH = '%s/test%d.h5'%(file_path_1,self.run_id)
			n_epochs = 100
			BATCH_SIZE = 256
			n_step_local = n_step_local_ori

			earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=1, mode='auto')
			checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)

			num_sample1 = 1
			interval = self.config['interval']
			select_num = np.int(np.ceil(train_num1/interval))
			# select_num1 = select_num*interval
			# print(num_sample1,select_num,interval,select_num1)
			select_num = 1
			if select_num>1:
				t1 = np.arange(0,train_num1,interval)
				pos = np.vstack((t1,t1+interval)).T
				pos[-1][1] = train_num1
				print(train_num1,select_num,interval)
				print(pos)
			else:
				pos = [[0,train_num1]]

			start2 = time.time()
			train_id_1 = np.arange(train_num1)
			valid_id_1 = np.arange(valid_num1)
			np.random.shuffle(valid_id_1)
			cnt1 = 0
			mse1 = 1e5
			decay_rate = 0.95
			decay_step = 1
			init_lr = self.config['lr']
			if 'n_epochs' in self.config:
				n_epochs = self.config['n_epochs']
			else:
				n_epochs = 100

			for i1 in range(1):

				self.config['lr'] = init_lr*((decay_rate)**(int(i1/decay_step)))
				np.random.shuffle(train_id_1)
				
				start1 = time.time()
				valid_id2 = valid_id_1
				x_valid, y_valid = x_valid[valid_id2], y_valid[valid_id2]

				for l in range(select_num):
					s1, s2 = pos[l]
					# print(l,s1,s2)
					sel_id = train_id_1[s1:s2]

					x_train = mtx_train[sel_id]
					y_train = y_train_ori_1[sel_id]

					x_train, y_train = np.asarray(x_train), np.asarray(y_train)
					# print(x_train.shape,y_train.shape)

					train_num = x_train.shape[0]
					print('x_train, y_train', x_train.shape, y_train.shape)
					print('x_valid, y_valid', x_valid.shape, y_valid.shape)
					
					model.fit(x_train,y_train,epochs = n_epochs, batch_size = BATCH_SIZE, validation_data = [x_valid,y_valid],
										callbacks=[earlystop,checkpointer])

			stop1 = time.time()
			print(stop1-start1)

			self.config.update({'model_path1':MODEL_PATH})

		else:
			if 'model_path1' in self.config:
				MODEL_PATH = self.config['model_path1']
			print('loading weights... ', MODEL_PATH)
			model.load_weights(MODEL_PATH) # load model with the minimum training error

		flag = 0
		valid_score_1 = []
		if self.config['predict_valid']==1:
			if self.train==1:
				print('loading weights... ', MODEL_PATH)
				model.load_weights(MODEL_PATH) # load model with the minimum training error
				flag = 1
			else:
				self.prep_data_2_sub2(type_id1=0,keys=['train','valid'],stride=1)	# type_id1=0, load sequences
				x_valid = self.x['valid']
				idx_sel_list_valid, y_valid_ori_1, y_valid_ori, vec_serial_valid, vec_local_valid = self.local_serial_dict['valid']

			y_predicted_valid = model.predict(x_valid)
			print(y_valid_ori_1.shape,y_predicted_valid.shape)
			valid_score_1 = score_2a(np.ravel(y_valid_ori_1), np.ravel(y_predicted_valid))
			print(valid_score_1)

		if self.config['predict_test']==1:
			# load serial and feature vectors
			key1 = 'test'
			x_test = self.x[key1]
			idx_sel_list_test, y_ori, y, vec_serial, vec_local = self.local_serial_dict[key1]

			if flag==0:
				print('loading weights... ', MODEL_PATH)
				model.load_weights(MODEL_PATH) # load model with the minimum training error

			y_predicted_test = model.predict(x_test,batch_size=16)
			y_test = y_ori
			score_vec1, score_dict1 = self.score_1(y_test,y_predicted_test,idx_sel_list_test,type_id=1)

			# file_path_1 = 'mouse_pred1'
			if run_id_load<0:
				run_id_load = self.run_id
			output_filename = '%s/test_%d_%d'%(file_path_1,self.run_id,run_id_load)
			# output_filename = 'test_%d_%d'%(self.run_id,run_id_load)
			predicted_attention_test = []
			self.output_predict_1(idx_sel_list_test, y_predicted_test, y_test, predicted_attention_test, 
								score_vec1, score_dict1, output_filename, valid_score=valid_score_1)

		return model

	# matching serial
	def matching_serial(self,serial1,serial2,list1,list2=[]):

		serial1, serial2 = np.asarray(serial1), np.asarray(serial2)
		id1 = utility_1.mapping_Idx(serial1,serial2)
		b1 = np.where(id1>=0)[0]
		id1 = id1[b1]

		list_1, list_2 = [], []
		for t_array in list1:
			t_array = t_array[id1]
			list_1.append(t_array)

		if len(list2)>0:
			for t_array in list2:
				t_array = t_array[id2]
				list_2.append(t_array)

		return id1, b1, list_1, list_2

	# concatenate features
	def feature_concatenate(self,serial1,serial2,feature1,feature2,list1,list2=[]):

		id1, id2, list_1, list_2 = self.matching_serial(serial1,serial2,list1,list2)
		idx_sel_list, y_ori = list_1[0:2]
		mtx_feature = np.hstack((feature1[id1],feature2[id2]))

		gap_tol, seq_len_thresh = self.config['gap_thresh'], self.config['seq_len_thresh']
		seq_list = generate_sequences(idx_sel_list, gap_tol=gap_tol, region_list=[])
		ref_serial = idx_sel_list[:,1]
		idx_sel_list, seq_list = self.select_region_local_1(idx_sel_list,seq_list, 
									gap_tol=gap_tol, seq_len_thresh=seq_len_thresh, region_list=[])

		x1 = idx_sel_list
		y1 = y_ori
		x, y, vec_serial, vec_local = sample_select2a1(x1,y1,idx_sel_list,seq_list,self.tol,self.flanking)

		id1 = utility_1.mapping_Idx(ref_serial,idx_sel_list[:,1])
		b1 = np.where(id1>=0)[0]
		assert len(b1)==len(idx_sel_list)
		id1 = id1[b1]
		mtx_feature, y_ori = mtx_feature[id1], y_ori[id1]

		return mtx_feature, [idx_sel_list, y_ori, y, vec_serial, vec_local]

	# load configuration parameters
	def load_config_1(self,run_id_load):

		config_filename = 'config_%d.npy'%(run_id_load)
		if os.path.exists(config_filename)==True:
			config = np.load(config_filename,allow_pickle=True)
			config = config[()]

		else:
			config = self.config_pre_1(run_id_load)
			np.save(config_filename,config,allow_pickle=True)

		return config

	# merge data from training, validation and test samples
	def merge_pre_1(self,filename_list,layer_name_list,output_filename):

		list1 = []
		dict1 = dict()
		for layer1 in layer_name_list:
			layer_name, type_id = layer1[0], layer1[1]
			dict1[layer_name] = []

		for filename1 in filename_list:
			with h5py.File(filename1,'r') as fid:

				encoded_serial = fid["serial"][:]
				print(len(encoded_serial),encoded_serial[0:10])

				for layer1 in layer_name_list:
					layer_name, type_id = layer1[0], layer1[1]
					encoded_vec = np.asarray(fid[layer_name][:],dtype=np.float32)
					print(encoded_vec.shape)

					assert len(encoded_serial)==encoded_vec.shape[0]
					dict1[layer_name].extend(encoded_vec)

				list1.extend(encoded_serial)

		serial_vec = np.asarray(list1)
		print(serial_vec.shape)
		with h5py.File(output_filename,'w') as fid:
			fid.create_dataset("serial", data=serial_vec, compression="gzip")
			for layer1 in layer_name_list:
				layer_name, type_id = layer1
				f_mtx = np.asarray(dict1[layer_name],dtype=np.float32)
				assert len(serial_vec)==f_mtx.shape[0]
				fid.create_dataset(layer_name, data=f_mtx, compression="gzip")

		return True

	# save predicted signals, predicted importances scores, and prediction performance evaluation scores
	def output_predict_1(self, idx_sel_list_test, y_predicted_test, y_test, predicted_attention_test, 
							score_vec1, score_dict1, output_filename, valid_score=[]):

		output_filename1 = '%s_predict.txt'%(output_filename)
		output_filename2 = '%s_score.txt'%(output_filename)

		# fields = ['chrom','start','stop','serial','signal','predicted_signal','predicted_attetion']
		# data1 = pd.DataFrame(columns=fields)
		if len(predicted_attention_test)>0:
			columns = ['serial','signal','predicted_signal','predicted_attention']
			# value = [idx_sel_list_test[:,1],y_test,y_predicted_test,predicted_attention_test]
			y_test, y_predicted_test = np.ravel(y_test), np.ravel(y_predicted_test)
			predicted_attention_test = np.ravel(predicted_attention_test)
			value = [idx_sel_list_test[:,1],y_test,y_predicted_test,predicted_attention_test]
			id1 = mapping_Idx(self.serial,idx_sel_list_test[:,1])
		else:
			columns = ['serial','signal','predicted_signal']
			# value = [idx_sel_list_test[:,1],y_test,y_predicted_test,predicted_attention_test]
			y_test, y_predicted_test = np.ravel(y_test), np.ravel(y_predicted_test)
			value = [idx_sel_list_test[:,1],y_test,y_predicted_test]
			id1 = mapping_Idx(self.serial,idx_sel_list_test[:,1])

		assert np.sum(id1<0)==0

		data1 = self.test_result_3_sub1(id1,columns,value,output_filename1,sort_flag=True,sort_column='serial')

		columns_1 = ['chrom','mse','Pearson_correlation','p_value1','explained_variance',
						'mean_abs_err','median_abs_err','r2_score','Spearman_correlation','p_value2']

		data2 = pd.DataFrame(columns=columns_1)
		if len(score_dict1)>0:
			list1 = score_dict1[0]
			chrom_vec = np.int32(list1[:,0])
			data2['chrom'] = chrom_vec
			num1 = len(columns_1)
			data2.loc[:,columns_1[1:]] = list1[:,1:]

		if len(valid_score)>0:
			valid_score_vec = np.asarray([0]+list(valid_score))
			valid_score_vec = valid_score_vec[np.newaxis,:]
			print(valid_score_vec)
			print(valid_score_vec.shape)
			data_2 = pd.DataFrame(columns=columns_1,data=valid_score_vec)

			data2 = pd.concat([data_2,data2], axis=0, join='outer', ignore_index=True, 
					keys=None, levels=None, names=None, verify_integrity=False, copy=True)

		# data2.to_csv(output_filename2,index=False,sep='\t',float_format='%.6f')

		return True

	# save prediction performance evaluation scores
	def output_predict_2(self, run_id, output_filename, valid_score=[]):

		# output_filename1 = 'test_score_valid.txt'
		columns_1 = ['run_id','chrom','mse','Pearson_correlation','p_value1','explained_variance',
						'mean_abs_err','median_abs_err','r2_score','Spearman_correlation','p_value2']

		filename1 = output_filename
		flag = 0
		if os.path.exists(filename1):
			data_1 = pd.read_csv(filename1,sep='\t')
			flag = 1
			t_columns = list(data_1)
			assert t_columns==columns_1

		if len(valid_score)>0:
			valid_score_vec = np.asarray([self.run_id]+[0]+list(valid_score))
			valid_score_vec = valid_score_vec[np.newaxis,:]
			print(valid_score_vec)
			print(valid_score_vec.shape)
			data_2 = pd.DataFrame(columns=columns_1,data=valid_score_vec)

		if flag==1:
			data_1 = data_1.append(data_2, ignore_index=True)
		else:
			data_1 = data_2

		data_1['run_id'] = np.int64(data_1['run_id'])
		data_1['chrom'] = np.int64(data_1['chrom'])
		data_1.to_csv(output_filename,index=False,sep='\t',float_format='%.6f')

		return data_1

	# test
	# type_id1: load sequences
	# type_id: by chromosome
	def test_pre_1(self,type_id1,type_id=1,est_attention=1,select_config={},interval=2500):

		# load serial and feature vectors
		self.prep_data_2_sub2(type_id1=type_id1,keys=['test'],stride=1,type_id=0,select_config=select_config)

		key1 = 'test'
		mtx_test = self.x[key1]
		idx_sel_list, y_ori, y, vec_serial, vec_local = self.local_serial_dict[key1]

		y_predicted_test, idx_sel_list_test, predicted_attention_test = self.predict_test_1(idx_sel_list,mtx_test,vec_serial,vec_local,
																			type_id=type_id,type_id1=1,est_attention=est_attention,interval=interval)
		# predicted_attention_test, idx_sel_list_test = self.estimate_attention_test_1(idx_sel_list,mtx_test,vec_serial,vec_local,type_id=0)

		y_test = y_ori
		score_vec1, score_dict1 = self.score_1(y_test,y_predicted_test,idx_sel_list_test,type_id=type_id)

		return (y_predicted_test, y_test, idx_sel_list_test, predicted_attention_test, score_vec1, score_dict1)

	# evaluate prediction
	def score_1(self,y,y_predicted,idx_sel_list=[],type_id=0):

		if y.ndim<2:
			y = y[:,np.newaxis]

		dim1 = y.shape[-1]
		vec1 = []
		dict1 = {}
		for i in range(dim1):
			# start1 = time.time()
			y_predicted_1 = y_predicted[:,i]
			y1 = y[:,i]
			t_score = utility_1.score_2a(y1, y_predicted_1)
			vec1.append([i]+t_score)
			# stop1 = time.time()
			# print(i, t_score, stop1-start1)
			print(i, t_score)
			if type_id==1:
				chrom1 = idx_sel_list[:,0]
				chrom_vec = np.unique(chrom1)
				list1 = [[-1]+list(t_score)]
				for chrom_id in chrom_vec:
					b1 = np.where(chrom1==chrom_id)[0]
					t_score1 = utility_1.score_2a(y1[b1], y_predicted_1[b1])
					print(i,chrom_id,t_score1)
					list1.append([chrom_id]+list(t_score1))
				dict1[i] = np.asarray(list1)

		vec1 = np.asarray(vec1)
		# print(vec1)

		return vec1, dict1

	# predict
	def predict_test_1(self,idx_sel_list,x_test_ori,vec_serial,vec_local,
						type_id=0,type_id1=1,est_attention=1,interval=2500):

		# load serial and feature vectors
		ref_serial = idx_sel_list[:,1]
		serial_1 = np.unique(vec_serial)
		assert list(np.sort(ref_serial))==list(serial_1)

		# id1 = mapping_Idx(ref_serial,serial_1)
		# idx_sel_list_1 = idx_sel_list[id1]
		test_num = len(idx_sel_list)
		print(test_num,idx_sel_list[0:5])

		start = time.time()
		if est_attention==1:
			attention_vec1 = np.zeros(test_num,dtype=np.float32)
			layer_name = self.config['layer_name_est']
			model = self.model
			intermediate_layer = Model(inputs=model.input,
										 outputs=model.get_layer(layer_name).output)
		else:
			attention_vec1 = []

		if type_id==0:
			x_test_ori1 = x_test_ori[vec_local]
			y_predicted_test1_ori = self.model.predict(x_test_ori1,batch_size=32)
			if type_id1==0:
				y_predicted = y_predicted_test1_ori[:,self.flanking,]
			else:
				y_predicted = utility_1.read_predict_1(y_predicted_test1_ori, vec_local, 
											idx=[], flanking1=3, type_id=0, base1=0.25)

			if est_attention==1:
				predicted_attention_ori = intermediate_layer.predict(x_test_ori)
				predicted_attention = self.process_attention_test_1(predicted_attention_ori)
				print('predicted attention',predicted_attention_ori.shape,predicted_attetion.shape)
		else:
			chrom = idx_sel_list[:,0]
			chrom_vec = np.unique(idx_sel_list[:,0])

			test_num = vec_local.shape[0]
			print('test',chrom_vec,test_num)
			num1 = len(chrom_vec)

			for i in range(num1):
				chrom_id = chrom_vec[i]
				id1 = np.where(chrom==chrom_id)[0]

				if interval>0:
					test_num1 = len(id1)
					t1 = np.arange(0,test_num1,interval)
					pos = np.vstack((t1,t1+interval)).T
					pos[-1][1] = test_num1
				else:
					pos = [[0,test_num1]]
				select_num = len(pos)
				print(test_num1,interval,select_num)
				# print(pos)

				for l in range(select_num):
					start1 = time.time()
					t_id1 = id1[pos[l][0]:pos[l][1]]
					t_vec_local = vec_local[t_id1]

					x_test1 = x_test_ori[t_vec_local]
					print(chrom_id,pos[l],t_id1,x_test1.shape)
					print(idx_sel_list[t_id1[0:5]])
					y_predicted_test1_ori = self.model.predict(x_test1,batch_size=32) # shape: (sample_num,context_size,output_dim)

					# shape: (sample_num,output_dim)
					# t1 = np.unique(t_vec_local)
					id2 = t_vec_local[:,self.flanking]
					# print(len(id2),len(t_vec_local),len(t1))
					if type_id1==0:
						y_predicted_test1 = y_predicted_test1_ori[:,self.flanking,]
					else:
						y_predicted_test1 = utility_1.read_predict_1(y_predicted_test1_ori, t_vec_local, 
													idx=id2, flanking1=3, type_id=0, base1=0.25)
					# print(y_predicted_test1_ori.shape,y_predicted_test1.shape)
					if est_attention==1:
						start2 = time.time()
						predicted_attention_ori = intermediate_layer.predict(x_test1)
						print(predicted_attention_ori.shape)
						predicted_attention = self.process_attention_test_1(predicted_attention_ori)
						attention_vec1[t_id1] = np.ravel(predicted_attention)
						stop2 = time.time()
						print('predicted attention',chrom_id,l,predicted_attention_ori.shape,predicted_attention.shape,stop2-start2)

					if (i==0) and (l==0):
						dim1 = y_predicted_test1.shape[1]
						y_predicted = np.zeros((test_num,dim1),dtype=np.float32)

					print(chrom_id,l,pos[l],x_test1.shape,y_predicted_test1.shape)
					y_predicted[t_id1] = y_predicted_test1

					stop1 = time.time() 
					print('predict',chrom_id,pos[l],stop1-start1)

		stop = time.time()
		print('predict',stop-start)

		return y_predicted, idx_sel_list, attention_vec1

	# process predicted attention
	def process_attention_test_1(self,predicted_attention,vec_serial=[],vec_local=[],type_id=0):

		predicted_attention_1 = predicted_attention[:,self.flanking]

		return predicted_attention_1

	# predict attention
	def estimate_attention_test_1(self,idx_sel_list,x_test_ori,vec_serial=[],vec_local=[],type_id=0):

		# load serial and feature vectors
		ref_serial = idx_sel_list[:,1]
		serial_1 = np.unique(vec_serial)
		assert list(np.sort(ref_serial))==list(serial_1)

		# id1 = mapping_Idx(ref_serial,serial_1)
		# idx_sel_list_1 = idx_sel_list[id1]

		layer_name = 'logits_T_3'
		model = self.model
		intermediate_layer = Model(inputs=model.input,
									 outputs=model.get_layer(layer_name).output)

		if type_id==0:
			predicted_attention_ori = intermediate_layer.predict(x_test_ori)
			print('predicted attention',predicted_attention_ori.shape)
			predicted_attention = self.process_attention_test_1(predicted_attention_ori)
		else:
			chrom = idx_sel_list[:,0]
			chrom_vec = np.unique(idx_sel_list[:,0])

			test_num = vec_local.shape[0]
			print('test',chrom_vec,test_num)
			num1 = len(chrom_vec)

			for i in range(num1):
				chrom_id = chrom_vec[i]
				id1 = np.where(chrom==chrom_id)[0]
				t_vec_local = vec_local[id1]

				x_test1 = x_test_ori[t_vec_local]
				# shape: (sample_num,context_size,1)
				predicted_attention_ori = intermediate_layer.predict(x_test1)
				print('predicted attention',predicted_attention_ori.shape)
				predicted_attention = self.process_attention_test_1(predicted_attention_ori)
				
				if i==0:
					# dim1 = y_predicted_test1.shape[1]
					attention_vec1 = np.zeros(test_num,dtype=np.float32)

				print(chrom_id,x_test1.shape,predicted_attention.shape)
				attention_vec1[id1] = predicted_attention

		return attention_vec1, idx_sel_list

	# predict with context
	def control_predict_1(self,x_pre,y_pre,idx_sel_list,vec_serial,vec_local,model,intermediate_layer,
								batch_size,predict_type=1):

		sample_num1 = len(vec_serial)
		n_batch = int(np.ceil(sample_num1/batch_size))
		s1 = np.arange(0,sample_num1,batch_size)
		s2 = s1+batch_size
		s2[-1] = sample_num1
		pos = np.vstack((s1,s2)).T

		list1 = []
		dict1 = dict()
		layer_name_list = list(intermediate_layer.keys())
		num1 = len(layer_name_list)
		for t_layer_name in layer_name_list:
			dict1[t_layer_name] = []

		print(n_batch,batch_size,pos)
		start1 = time.time()
		ref_serial = idx_sel_list[:,1]
		sample_num = len(np.unique(vec_serial))
		assert len(ref_serial)==sample_num, 'error! %d %d'%(len(ref_serial),sample_num)

		id_vec1 = np.zeros(sample_num,dtype=np.int8)

		t1 = ref_serial[vec_local]
		t2 = np.abs(vec_serial-t1)
		b1 = np.where(np.ravel(t2)>0)[0]
		print(len(b1),b1)

		assert len(b1)==0

		for i in range(n_batch):
			s1, s2 = pos[i]
			t_serial = vec_serial[s1:s2]
			x1 = x_pre[vec_local[s1:s2]]
			y1 = y_pre[s1:s2]
			print(i,s1,s2,x1.shape,y1.shape)

			for t_layer_name in layer_name_list:
				start = time.time()
				intermediate_layer1, type_id = intermediate_layer[t_layer_name]
				feature1 = intermediate_layer1.predict(x1)
				print(feature1[0].shape)
				print(len(feature1))
				dim1, dim2 = feature1.shape[-2], feature1.shape[-1]

				if type_id<2:

					if i==0:
						f_mtx = np.zeros((sample_num,dim1,dim2),dtype=np.float32)
						dict1[t_layer_name] = f_mtx

					if type_id==0:
						t_serial_local = t_serial[:,self.flanking]
						feature_1 = feature1[:,self.flanking]
						
						assert np.max(np.abs(ref_serial[s1:s2]-t_serial_local))==0

					else:
						t_serial = np.ravel(t_serial)
						t_feature = np.reshape(feature1,(-1,dim1,dim2))
						t1 = np.unique(t_serial,return_index=True,return_inverse=True)
						t_serial_local, t_id1 = t1[0], t1[1]
						# print('t_serial_local',len(t_serial_local),len(t_id1))

						assert np.max(np.abs(t_serial[t_id1]-t_serial_local))==0

						feature_1 = t_feature[t_id1]
						print(t_feature.shape,feature_1.shape)

					id1 = mapping_Idx(ref_serial,t_serial_local)
					assert np.sum(id1<0)==0
					id_vec1[id1]=1
					print(np.sum(id_vec1),sample_num)
					dict1[t_layer_name][id1] = feature_1

				else:
					t_serial_local = t_serial[:,self.flanking]
					id1 = mapping_Idx(ref_serial,t_serial_local)
					assert np.sum(id1<0)==0
					id_vec1[id1]=1

					feature_1 = feature1
					dict1[t_layer_name].extend(feature_1)

				stop = time.time()
				# dict1[t_layer_name].extend(feature1)
				print(feature_1.shape,stop-start)

			if predict_type==1:
				start = time.time()
				y_predicted = model.predict(x1)
				stop = time.time()
				print(y_predicted.shape,stop-start)
				list1.extend(y_predicted)

		idx_sel_list1 = idx_sel_list[id_vec1>0]
				
		stop1 = time.time()
		list1 = np.asarray(list1)
		print(stop1-start1)

		return idx_sel_list1, dict1, list1

	def control_predict_local(self,key1,model,intermediate_layer,batch_size,predict_type=1,predict_type1=0):

		# key1 = 'train'
		x1 = self.x[key1]
		idx_sel_list1, y_ori, y1, vec_serial_1, vec_local_1 = self.local_serial_dict[key1]

		print(key1,len(idx_sel_list1),y1.shape,vec_serial_1.shape,vec_local_1.shape)
		idx_sel_list_1, feature1, y_predicted1 = self.control_predict_1(x1,y1,idx_sel_list1,vec_serial_1,vec_local_1,model,intermediate_layer,
																batch_size=batch_size,predict_type=predict_type)
		print(key1,len(idx_sel_list_1),len(feature1),len(y_predicted1))

		# serial1 = vec_serial_1[:,self.flanking]
		serial1 = idx_sel_list_1[:,1]
		# id1 = mapping_Idx(idx_sel_list1[:,1],serial1)
		# b1 = np.where(id1<0)[0]
		# if len(b1)>0:
		# 	print('error!',len(b1))
		# 	return -1
		# idx_sel_list_1 = idx_sel_list1[id1]

		score_vec = []
		y_predicted_1 = []
		if len(y_predicted1)>0:
			y_predicted1 = np.asarray(y_predicted1,dtype=np.float32)
			# if y_predicted1.ndim>2:
			# 	y_predicted_1 = np.ravel(y_predicted1[:,self.flanking])
			# 	y = np.ravel(y1[:,self.flanking])
			# else:
			# 	y_predicted_1, y = np.ravel(y_predicted1), np.ravel(y)

			if predict_type1==0:
				y_predicted_1 = np.ravel(y_predicted1[:,self.flanking])
				y = np.ravel(y1[:,self.flanking])
				t_serial_local = vec_serial_1[:,self.flanking]
			else:
				y_predicted_1, y = np.ravel(y_predicted1), np.ravel(y1)

				t1 = np.unique(vec_serial_1,return_index=True,return_inverse=True)
				t_serial_local, t_id1 = t1[0], t1[1]
				y_predicted_1, y = y_predicted_1[t_id1], y[t_id1]

			assert len(t_serial_local)==len(serial1), '%d %d'%(len(t_serial_local),len(serial1))
			serial_1 = np.sort(serial1)
			assert np.max(np.abs(t_serial_local-serial_1))==0

			score1 = utility_1.score_2a(y,y_predicted_1)
			print(key1,y.shape,y_predicted_1.shape)
			print(score1)
			score_vec.append([-1]+list(score1))

			# chrom_vec = np.unique(idx_sel_list1[:,0])
			# serial = idx_sel_list1[:,1]
			# chrom1 = idx_sel_list_1[:,0]

			id1 = mapping_Idx(serial1,t_serial_local)
			chrom1 = idx_sel_list_1[id1,0]
			chrom_vec = np.unique(chrom1)

			for chrom_id in chrom_vec:
				b1 = np.where(chrom1==chrom_id)[0]
				t_score1 = utility_1.score_2a(y[b1],y_predicted_1[b1])
				t1 = [chrom_id]+list(t_score1)
				print(t1)
				score_vec.append(t1)
			score_vec = np.asarray(score_vec)

		return idx_sel_list_1,feature1,y_predicted1,y_predicted_1,vec_serial_1,score_vec

	def config_pre_1(self,run_id,config={}):

		flanking = self.flanking
		context_size = 2*flanking+1
		n_step_local_ori = 5000
		region_unit_size = 1
		feature_dim = 4

		n_step_local_ori = config['n_step_local_ori']
		local_conv_list1 = config['local_conv_list1']

		size1 = n_step_local_ori
		for conv_1 in local_conv_list1:
			n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate = conv_1
			if boundary == 0:
				t1 = int(size1/stride)
			else:
				t1 = int((size1-kernel_size1)/stride)+1

			size1 = int((t1-pool_length1)/stride1)+1
			print(n_filters, kernel_size1, stride, pool_length1, stride1, size1)

		n_step_local1 = size1
		flag = local_vec_1[-1]
		feature_dim_1 = local_vec_1[1]
		if flag==0:
			feature_dim1 = feature_dim_1*n_step_local1
		else:
			feature_dim1 = feature_dim_1

		print('n_step_local', n_step_local1, feature_dim1)
		config.update({'n_step_local':n_step_local1,'feature_dim1':feature_dim1})

		return config

	def config_pre_1_1(self,sel_id=0,config={}):

		flanking = self.flanking
		context_size = 2*flanking+1
		n_step_local_ori = 5000
		region_unit_size = 1
		feature_dim = 4

		regularizer2 = 1e-04
		regularizer2_2 = 1e-05
		bnorm = 1
		dropout_rate = 0.2
		activation = 'relu'
		
		list1 = [[32,50,50,1,1],[32,0,50,1,1],[32,50,0,1,1],
					[32,50,50,0,1],[32,0,50,0,1],[32,50,0,0,1],
					[32,25,0,0,0],[32,25,25,0,0],
					[32,25,25,1,0],[32,0,25,1,0],[32,25,0,1,0]]
		hidden_unit,fc1_output_dim,fc2_output_dim,attention2_local,pooling_local = list1[sel_id]

		if len(config)==0:
			config = self.config.copy()
			units1=[50,50,50,25,50,25,0,0]
			units2=[50,50,50,25,50,0,0,0]

			boundary = 1
			# n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate
			local_conv_list_1 = [[128, 15, 5, regularizer2_2, 1, boundary, bnorm, activation, 1, 1, 0.2],
								[32, 7, 1, regularizer2_2, 1, boundary, bnorm, activation, 5, 5, 0.2],
								[32, 5, 1, regularizer2_2, 1, boundary, bnorm, activation, 5, 5, 0.2],
								[25, 3, 1, regularizer2_2, 1, boundary, bnorm, activation, 2, 2, 0.2]]

			# output_dim, activation2, regularizer2, recurrent_dropout_rate, drop_out_rate, layer_norm = conv_1[0:5]
			conv_list2 = [[hidden_unit, 'tanh', regularizer2_2, 0.1, dropout_rate, 1]]

			# conv_1 = [fc1_output_dim, bnorm, activation, dropout_rate]
			if fc1_output_dim>0:
				conv_1 = [fc1_output_dim, bnorm, activation , dropout_rate]
			else:
				conv_1 = []
			connection = [conv_1]

			if fc2_output_dim>0:
				conv_1 = [fc2_output_dim, bnorm, activation, dropout_rate]
			else:
				conv_1 = []
			conv_list3 = [conv_1]

			# return_sequences_flag, sample_local, pooling_local = select_config['local_vec_1']
			local_vec_1 = [True, 0, pooling_local]
			# attention2_local = 1
			concatenate_1, concatenate_2 = 0, 0
			# hidden_unit = 25
			conv_2 = [1,1,'sigmoid']

			conv_list_ori = [local_conv_list_1,conv_list2,connection,conv_list3]
			config['local_conv_list_ori'] = conv_list_ori
			config['conv_2'] = conv_2

		config['feature_dim_vec'] = units1[2:]
		config['feature_dim_vec_basic'] = units2[2:]
		config.update({'local_vec_1':local_vec_1})
		config.update({'feature_dim':feature_dim})
		config.update({'context_size':context_size,
						'n_step_local_ori':n_step_local_ori})
		config.update({'attention2_local':attention2_local})
		config.update({'concatenate_1':concatenate_1,'concatenate_2':concatenate_2})
		config.update({'feature_dim':feature_dim,'regularizer2_2':regularizer2_2})

		return config

	def config_pre_2(self,n_step_local,feature_dim,attention2_local,select2=0,config={}):

		if len(config)==0:
			config = self.config.copy()

		# units1=[50,50,50,50,1,25,25,1]
		# units1=[50,50,50,25,50,25,0,0]
		# n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
		# units1=[50,50,50,25,50,25,0,0]
		units_1, units_2 = 50, 25
		units1=[50,50,units_1,25,50,units_2,0,0]
		# units1=[50,50,50,25,50,25,0,0]
		config['feature_dim_vec'] = units1[2:]
		# n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
		# units1=[50,50,50,25,50,0,0,0]
		# units1=[50,50,50,25,50,0,0,0]
		units_1, units_2 = 50, 0
		units1=[50,50,units_1,25,50,units_2,0,0]
		config['feature_dim_vec_basic'] = units1[2:]

		n_filter1, kernel_size1, stride1, dilation_rate1, pool_length, stride2 = 32,3,1,1,1,1
		if n_step_local>25:
			pool_length, stride2 = 2,2
		local_vec_1 = [[[n_filter1,kernel_size1,stride1,dilation_rate1,pool_length,stride2]], 32, 50, [], True, 0, 1]
		hidden_unit = 32
		config['local_vec_1'] = local_vec_1

		flanking = self.flanking
		context_size = 2*flanking+1

		concatenate_1, concatenate_2 = 0,0
		regularizer2, regularizer2_2 = 1e-04, 1e-04
		config.update({'attention1':0,'attention2':1,'context_size':context_size,
								'n_step_local':n_step_local})
		config.update({'select2':select2,'attention2_local':attention2_local})
		config.update({'concatenate_1':concatenate_1,'concatenate_2':concatenate_2})
		config.update({'feature_dim':feature_dim,'output_dim':hidden_unit,'regularizer2':regularizer2,'regularizer2_2':regularizer2_2})

		return config

	def set_train_mode(self,train_mode):
		self.config['train_mode'] = train_mode
		self.train = train_mode

	def get_model_sub1(self,model_type_id,context_size,config):

		print(model_type_id)
		if model_type_id==1:
			model = utility_1.get_model2a1_attention_1_2_2_sample(context_size,config)
		elif model_type_id==2:
			config['layer_norm'] = 1
			model = utility_1_5.get_model2a1_attention_1_2_2_sample_1(context_size,config)
		elif model_type_id==3:
			config['layer_norm'] = 0
			model = utility_1_5.get_model2a1_attention_1_2_2_sample_1(context_size,config)
		elif model_type_id==5:
			config['layer_norm'] = 2
			model = utility_1_5.get_model2a1_attention_1_2_2_sample_1(context_size,config)
		elif model_type_id==6:
			config['layer_norm'] = 2
			# construct gumbel selector2
			# get_model2a1_basic1_2
			print('layer_norm',model_type_id,config['layer_norm'])
			model = utility_1_5.get_model2a1_attention_1_2_2_sample_2(context_size,config,layernorm_typeid=1)
			#return -1
		elif model_type_id==8:
			config['layer_norm'] = 0
			# construct gumbel selector2
			# get_model2a1_basic1_2
			print('layer_norm',model_type_id,config['layer_norm'])
			model = utility_1_5.get_model2a1_attention_1_2_2_sample_2(context_size,config,layernorm_typeid=0)
			#return -1
		elif model_type_id==9:
			config['layer_norm'] = 1
			# construct gumbel selector2
			# get_model2a1_basic1_2
			print('layer_norm',model_type_id,config['layer_norm'])
			model = utility_1_5.get_model2a1_attention_1_2_2_sample_2(context_size,config,layernorm_typeid=0)
			#return -1
		elif model_type_id==10:
			config['layer_norm'] = 0
			# construct gumbel selector2
			# get_model2a1_basic1_2
			print('layer_norm',model_type_id,config['layer_norm'])
			model = utility_1_5.get_model2a1_attention_1_2_2_sample_2(context_size,config,layernorm_typeid=1)
			#return -1
		elif model_type_id==11:
			config['layer_norm'] = 1
			# construct gumbel selector2
			# get_model2a1_basic1_2
			print('layer_norm',model_type_id,config['layer_norm'])
			model = utility_1_5.get_model2a1_attention_1_2_2_sample_2(context_size,config,layernorm_typeid=1)
			#return -1
		else:
			model = utility_1_5.get_model2a1_attention_1_2_2_sample(context_size,config)

		return model

	# training and prediction with sequence features
	def control_pre_test3(self,path1,file_prefix,model_path1='',type_id=-1):

		self.file_path, self.file_prefix = path1, file_prefix
		self.prep_data_2_sub1(path1,file_prefix,type_id1=1,type_id2=1)

		# find feature vectors with the serial
		self.x = dict()
		self.idx = dict()
		self.prep_data_2_sub2(type_id1=1,keys=['train','valid'],stride=1)

		config = self.config.copy()
		units1=[50,50,50,25,50,25,0,0]
		config['feature_dim_vec'] = units1[2:]
		config['feature_dim_vec_basic'] = units1[2:]

		config['attention1']=0
		config['attention2']=1
		config['select2']=1
		print('get_model2a1_attention_1_2_2_sample')
		context_size = 2*self.flanking+1
		config['context_size'] = context_size
		config['feature_dim'] = self.x['train'].shape[-1]

		if 'model_type_id' in config:
			model_type_id = config['model_type_id']
		else:
			model_type_id = 0

		model = self.get_model_sub1(model_type_id,context_size,config)

		mtx_train = self.x['train']
		idx_sel_list_train, y_train_ori_1, y_train_ori, vec_serial_train, vec_local_train = self.local_serial_dict['train']

		mtx_valid = self.x['valid']
		idx_sel_list_valid, y_valid_ori_1, y_valid_ori, vec_serial_valid, vec_local_valid = self.local_serial_dict['valid']

		train_num1, valid_num1 = mtx_train.shape[0], mtx_valid.shape[0]
		print('train',len(idx_sel_list_train),len(y_train_ori),mtx_train.shape)
		print('valid',len(idx_sel_list_valid),len(y_valid_ori),mtx_valid.shape)

		x_train = np.asarray(mtx_train[vec_local_train])
		y_train = np.asarray(y_train_ori)

		x_valid = np.asarray(mtx_valid[vec_local_valid])
		y_valid = np.asarray(y_valid_ori)

		print(x_train.shape,y_train.shape)
		print(x_valid.shape,y_valid.shape)

		type_id2 = 2
		MODEL_PATH = 'test%d.h5'%(self.run_id)
		n_epochs = 100
		BATCH_SIZE = 512

		start1 = time.time()
		if self.train==1:
			print('x_train, y_train', x_train.shape, y_train.shape)
			earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=0, mode='auto')
			checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
			# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
			model.fit(x_train,y_train,epochs = n_epochs, batch_size = BATCH_SIZE, validation_data = [x_valid,y_valid],
							callbacks=[earlystop,checkpointer])
			# model.load_weights(MODEL_PATH)
			model_path2 = '%s/model_%d_%d_%d.h5'%(self.path,self.run_id,type_id2,context_size)
			model.save(model_path2)
			model_path2 = MODEL_PATH
			print('loading weights... ', model_path2)
			model.load_weights(model_path2) # load model with the minimum training error
		else:
			# model_path1 = './mnt/yy3/test_29200.h5'
			print(self.run_id,MODEL_PATH)
			if model_path1!="":
				MODEL_PATH = model_path1
			if os.path.exists(MODEL_PATH)==False:
				print('file does not exist',MODEL_PATH)
				return -1

			print('loading weights... ', MODEL_PATH, model_path1)
			model.load_weights(MODEL_PATH)

		self.model = model

		stop1 = time.time()
		print(stop1-start1)

		predict_valid = 1
		if ('predict_valid' in self.config) and (self.config['predict_valid']==0):
			predict_valid = 0

		if predict_valid==1:
			y_predicted_valid = model.predict(x_valid)
			y_predicted_valid = np.ravel(y_predicted_valid[:,self.flanking])
			valid_score_1 = score_2a(np.ravel(y_valid[:,self.flanking]), y_predicted_valid)
			print(valid_score_1)
		else:
			valid_score_1 = []

		type_id1=1
		type_id=1
		interval=5000
		est_attention = 0
		if ('predict_test' in self.config) and (self.config['predict_test']==0):
			
			output_filename = self.config['valid_output_filename']
			self.output_predict_2(self.run_id,output_filename=output_filename,valid_score=valid_score_1)

		else:
			if 'est_attention' in self.config:
				est_attention = self.config['est_attention']
			y_predicted_test, y_test, idx_sel_list_test, predicted_attention_test, score_vec1, score_dict1 = self.test_pre_1(type_id1=type_id1,type_id=type_id,interval=interval,
																																est_attention=est_attention,select_config={})

			if 'predict_test_filename' in self.config:
				output_filename = 'test_%d_%d.%s'%(self.run_id,self.run_id,self.config['predict_test_filename'])
			else:
				output_filename = 'test_%d_%d'%(self.run_id,self.run_id)
			self.output_predict_1(idx_sel_list_test, y_predicted_test, y_test, predicted_attention_test, 
								score_vec1, score_dict1, output_filename, valid_score=valid_score_1)


		return model

	# training and prediction with sequence features
	def control_pre_test3_predict(self,path1,file_prefix,model_path1='',type_id=-1,est_attention=1,output_filename=''):

		self.file_path, self.file_prefix = path1, file_prefix
		self.prep_data_2_sub1(path1,file_prefix,type_id1=1,type_id2=1)

		# x_train1 = np.asarray(np.random.rand(5000,context_size,n_step_local,feature_dim),dtype=np.float32)
		# y_train1 = np.asarray(np.random.rand(5000,context_size,1),dtype=np.float32)

		# find feature vectors with the serial
		self.x = dict()
		self.idx = dict()
		self.prep_data_2_sub2(type_id1=1,keys=['test'],stride=1)

		config = self.config.copy()
		# units1=[50,50,50,50,1,25,25,1]
		# units1=[50,50,50,25,50,25,0,0]
		# n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
		units1=[50,50,50,25,50,25,0,0]
		config['feature_dim_vec'] = units1[2:]
		# n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
		# units1=[50,50,50,25,50,25,0,0]
		config['feature_dim_vec_basic'] = units1[2:]

		config['attention1']=0
		config['attention2']=1
		config['select2']=1
		print('get_model2a1_attention_1_2_2_sample')
		context_size = 2*self.flanking+1
		config['context_size'] = context_size
		config['feature_dim'] = self.x['test'].shape[-1]

		if 'model_type_id' in config:
			model_type_id = config['model_type_id']
		else:
			model_type_id = 0

		model = self.get_model_sub1(model_type_id,context_size,config)

		# model = utility_1.get_model2a1_word()
		# return -1

		if model_path1!="":
			MODEL_PATH = model_path1
		else:
			MODEL_PATH = 'test%d.h5'%(self.run_id_load)

		print('loading weights... ', MODEL_PATH, model_path1)
		model.load_weights(MODEL_PATH)
		self.model = model

		# mtx_train = self.x['train']
		# idx_sel_list_train, y_train_ori_1, y_train_ori, vec_serial_train, vec_local_train = self.local_serial_dict['train']

		# mtx_valid = self.x['valid']
		# idx_sel_list_valid, y_valid_ori_1, y_valid_ori, vec_serial_valid, vec_local_valid = self.local_serial_dict['valid']

		# train_num1, valid_num1 = mtx_train.shape[0], mtx_valid.shape[0]
		# print('train',len(idx_sel_list_train),len(y_train_ori),mtx_train.shape)
		# print('valid',len(idx_sel_list_valid),len(y_valid_ori),mtx_valid.shape)

		# x_train = np.asarray(mtx_train[vec_local_train])
		# y_train = np.asarray(y_train_ori)

		# x_valid = np.asarray(mtx_valid[vec_local_valid])
		# y_valid = np.asarray(y_valid_ori)

		# print(x_train.shape,y_train.shape)
		# print(x_valid.shape,y_valid.shape)

		start1 = time.time()
		valid_score_1 = []
		# y_predicted_valid = model.predict(x_valid)
		# y_predicted_valid = np.ravel(y_predicted_valid[:,self.flanking])
		# valid_score_1 = score_2a(np.ravel(y_valid[:,self.flanking]), y_predicted_valid)
		# print(valid_score_1)

		print('predict')

		type_id1=1
		type_id=1
		interval=5000
		y_predicted_test, y_test, idx_sel_list_test, predicted_attention_test, score_vec1, score_dict1 = self.test_pre_1(type_id1=type_id1,type_id=type_id,interval=interval,
																															est_attention=est_attention,select_config={})
		if output_filename=='':
			# output_filename = 'test_%d_%d'%(self.run_id,self.run_id)
			# for the simulation mode
			# file_path_1 = 'vbak2_pred6_1_local'
			# file_path_1 = 'vbak2_pred6_3_local'
			# file_path_1 = 'vbak2_pred7'
			file_path_1 = 'vbak2_pred7_1'
			output_filename = '%s/test_%d_%d_simu%s'%(file_path_1,self.run_id,self.run_id,self.config['filename_prefix_predict'])	

		self.output_predict_1(idx_sel_list_test, y_predicted_test, y_test, predicted_attention_test, 
							score_vec1, score_dict1, output_filename, valid_score=valid_score_1)

		stop1 = time.time()
		print(stop1-start1)

		return model

	# training and prediction with sequence features
	# quantile of attention
	def control_pre_test3_predict_sub1(self,path1,file_prefix,model_path1='',type_id=-1,est_attention=1,output_filename=''):

		self.file_path, self.file_prefix = path1, file_prefix
		self.prep_data_2_sub1(path1,file_prefix,type_id1=1,type_id2=1)

		data1, chromvec = utility_1.select_region1_sub(filename,type_id=1,data1=[],filename_centromere='')


	# training and prediction with sequence features
	# important loci estimation
	def control_pre_test3_predict_sub2(self,path1,file_prefix,model_path1='',type_id=-1,est_attention=1,output_filename=''):

		self.file_path, self.file_prefix = path1, file_prefix
		self.prep_data_2_sub1(path1,file_prefix,type_id1=1,type_id2=1)


	# training and prediction with sequence features
	# important loci estimated compared with specific elements
	def control_pre_test3_predict_sel1(self,path1,file_prefix,model_path1='',type_id=-1,est_attention=1,output_filename=''):

		self.file_path, self.file_prefix = path1, file_prefix
		self.prep_data_2_sub1(path1,file_prefix,type_id1=1,type_id2=1)

	# training and prediction with sequence features
	# important loci estimated compared with sepcific elements: ERCEs
	def control_pre_test3_predict_sel2(self,path1,file_prefix,model_path1='',type_id=-1,est_attention=1,output_filename=''):

		self.file_path, self.file_prefix = path1, file_prefix
		self.prep_data_2_sub1(path1,file_prefix,type_id1=1,type_id2=1)

	def control_pre_test3_1(self,path1,file_prefix,model_path1=''):

		self.file_path, self.file_prefix = path1, file_prefix
		self.prep_data_2_sub1(path1,file_prefix,type_id1=1,type_id2=1)

		# x_train1 = np.asarray(np.random.rand(5000,context_size,n_step_local,feature_dim),dtype=np.float32)
		# y_train1 = np.asarray(np.random.rand(5000,context_size,1),dtype=np.float32)

		# find feature vectors with the serial
		self.x = dict()
		self.idx = dict()
		self.prep_data_2_sub2(type_id1=1,keys=['train','valid'],stride=1)

		config = self.config.copy()
		# units1=[50,50,50,50,1,25,25,1]
		# units1=[50,50,50,25,50,25,0,0]
		# n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
		units1=[50,50,50,25,50,25,0,0]
		config['feature_dim_vec'] = units1[2:]
		# n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
		# units1=[50,50,50,25,50,25,0,0]
		config['feature_dim_vec_basic'] = units1[2:]

		config['attention1']=0
		config['attention2']=1
		config['select2']=1
		print('get_model2a1_attention_1_2_2_sample')
		context_size = 2*self.flanking+1
		config['context_size'] = context_size
		config['feature_dim'] = self.x['train'].shape[-1]
		model = utility_1.get_model2a1_attention_1_2_2_sample(context_size,config)
		
		# model = utility_1.get_model2a1_word()
		# return -1

		config1 = config.copy()
		# model = utility_1.get_model2a1_attention_1_2_2_sample2(context_size,config1)
		model = get_model2a1_convolution(config1)

		mtx_train = self.x['train']
		idx_sel_list_train, y_train_ori_1, y_train_ori, vec_serial_train, vec_local_train = self.local_serial_dict['train']

		mtx_valid = self.x['valid']
		idx_sel_list_valid, y_valid_ori_1, y_valid_ori, vec_serial_valid, vec_local_valid = self.local_serial_dict['valid']

		train_num1, valid_num1 = mtx_train.shape[0], mtx_valid.shape[0]
		print('train',len(idx_sel_list_train),len(y_train_ori),mtx_train.shape)
		print('valid',len(idx_sel_list_valid),len(y_valid_ori),mtx_valid.shape)

		x_train = np.asarray(mtx_train[vec_local_train])
		y_train = np.asarray(y_train_ori)

		x_valid = np.asarray(mtx_valid[vec_local_valid])
		y_valid = np.asarray(y_valid_ori)

		print(x_train.shape,y_train.shape)
		print(x_valid.shape,y_valid.shape)

		type_id2 = 2
		MODEL_PATH = 'test%d.h5'%(self.run_id)
		n_epochs = 100
		BATCH_SIZE = 512

		start1 = time.time()
		if self.train==1:
			print('x_train, y_train', x_train.shape, y_train.shape)
			earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=0, mode='auto')
			checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
			# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
			model.fit(x_train,y_train,epochs = n_epochs, batch_size = BATCH_SIZE, validation_data = [x_valid,y_valid],
							callbacks=[earlystop,checkpointer])
			# model.load_weights(MODEL_PATH)
			model_path2 = '%s/model_%d_%d_%d.h5'%(self.path,self.run_id,type_id2,context_size)
			model.save(model_path2)
			model_path2 = MODEL_PATH
			print('loading weights... ', model_path2)
			model.load_weights(model_path2) # load model with the minimum training error
		else:
			# model_path1 = './mnt/yy3/test_29200.h5'
			if model_path1!="":
				MODEL_PATH = model_path1
			print('loading weights... ', MODEL_PATH)
			model.load_weights(MODEL_PATH)

		self.model = model

		stop1 = time.time()
		print(stop1-start1)

		y_predicted_valid = model.predict(x_valid)
		y_predicted_valid = np.ravel(y_predicted_valid[:,self.flanking])
		valid_score_1 = score_2a(np.ravel(y_valid[:,self.flanking]), y_predicted_valid)
		print(valid_score_1)

		type_id1=1
		type_id=1
		interval=5000
		y_predicted_test, y_test, idx_sel_list_test, predicted_attention_test, score_vec1, score_dict1 = self.test_pre_1(type_id1=type_id1,type_id=type_id,interval=interval,est_attention=0,select_config={})

		output_filename = 'test_%d_%d.%d'%(self.run_id,self.run_id,self.config['predict_annot'])
		self.output_predict_1(idx_sel_list_test, y_predicted_test, y_test, predicted_attention_test, 
							score_vec1, score_dict1, output_filename, valid_score=valid_score_1)

		return model

	# select region by the length of the local sequence
	def select_region_local_1(self,idx_sel_list,seq_list, gap_tol=5, seq_len_thresh=5, region_list=[]):

		sample_num = len(idx_sel_list)
		ref_serial = idx_sel_list[:,1]
		id_vec = np.zeros(sample_num,dtype=np.int8)

		seq_len = seq_list[:,1]-seq_list[:,0]+1
		thresh1 = seq_len_thresh
		b1 = np.where(seq_len>thresh1)[0]
		print(len(seq_list),len(b1))
		seq_list = seq_list[b1]
		seq_len1 = seq_list[:,1]-seq_list[:,0]+1
		print(sample_num,np.sum(seq_len1),len(seq_list),np.max(seq_len),np.min(seq_len),np.median(seq_len),np.max(seq_len1),np.min(seq_len1),np.median(seq_len1))

		for t_seq_list in seq_list:
			s1, s2 = t_seq_list[0], t_seq_list[1]+1
			id_vec[s1:s2] = 1

		id1 = np.where(id_vec>0)[0]
		idx_sel_list1 = idx_sel_list[id1]

		seq_list = generate_sequences(idx_sel_list1, gap_tol=gap_tol, region_list=region_list)
		seq_len = seq_list[:,1]-seq_list[:,0]+1
		thresh1 = seq_len_thresh
		b1 = np.where(seq_len<=thresh1)[0]
		if len(b1)>0:
			print('error!',len(b1),len(seq_list))
			# return -1

		self.output_generate_sequences(idx_sel_list1,seq_list)

		return idx_sel_list1, seq_list

	# prepare data from predefined features
	# one hot encoded feature vectors for each chromosome
	def prep_data_sequence_ori(self):

		# self.feature_dim_transform = feature_dim_transform
		# map_idx = mapping_Idx(serial_ori,serial)

		file_path1 = '/work/magroup/yy3/data1/replication_timing3'
		if self.species_id=='mm10':
			if self.cell_type1==1:
				filename1 = '%s/mouse/mm10_5k_seq_genome1_1.txt'%(file_path1)
			else:
				filename1 = '%s/mouse/mm10_5k_seq_genome2_1.txt'%(file_path1)
			annot1 = '%s_%d'%(self.species_id,self.cell_type1)
		else:
			filename1 = '%s/hg38_5k_seq'%(file_path1)
			annot1 = self.species_id

		data_1 = pd.read_csv(filename1,sep='\t')
		colnames = list(data_1)
		local_serial = np.asarray(data_1['serial'])
		local_seq = np.asarray(data_1['seq'])
		print('local_serial,local_seq',local_serial.shape,local_seq.shape)

		if ('centromere' in self.config) and (self.config['centromere']==1):
			regionlist_filename = 'hg38.centromere.bed'
			serial1 = local_serial
			serial_list1, centromere_serial = self.select_region(serial1, regionlist_filename)
			id1 = mapping_Idx(serial1,serial_list1)
			id1 = id1[id1>=0]
			local_seq = local_seq[id1]
			local_serial = local_serial[id1]
			print('centromere',local_seq.shape,local_serial.shape,len(serial_list1),len(centromere_serial))

		if self.chrom_num>0:
			chrom_num = self.chrom_num
		else:
			chrom_num = len(np.unique(self.chrom))

		chrom_vec = list(range(1,chrom_num+1))
		train_sel_list1, train_sel_list2 = [],[]
		
		for t_chrom in chrom_vec:
		# for t_chrom in [22]:
			t_chrom1 = 'chr%d'%(t_chrom)
			b1 = np.where(self.chrom_ori==t_chrom1)[0]
			t_serial = self.serial_ori[b1]
			id1 = mapping_Idx(local_serial,t_serial)
			id1 = id1[id1>=0]

			filename = '%s_%s_encoded1.h5'%(annot1,t_chrom1)
			if os.path.exists(filename)==False:
				encoded_vec, encoded_serial = utility_1.one_hot_encoding(local_seq[id1],local_serial[id1])
				print(encoded_vec.shape,encoded_serial.shape)
				# dict1 = {'serial':encoded_serial,'vec':encoded_vec}
				# np.save(filename,dict1,allow_pickle=True)
				# with open(filename, "wb") as fid:
				# 	pickle.dump(dict1,fid,protocol=4)
				# data1 = np.load(filename,allow_pickle=True)
				# data1 = data1[()]
				# print(list(data1.keys()))
				with h5py.File(filename,'w') as fid:
					fid.create_dataset("serial", data=encoded_serial, compression="gzip")
					fid.create_dataset("vec", data=encoded_vec, compression="gzip")
			else:
				with h5py.File(filename,'r') as fid:
					encoded_serial = fid["serial"][:]
					encoded_vec = fid["vec"][:]
					print(t_chrom,encoded_serial.shape,encoded_vec.shape)
			
			train_sel_list1.extend([t_chrom]*len(encoded_serial))
			train_sel_list2.extend(encoded_serial)

		# train_sel_list = np.vstack((train_sel_list1,train_sel_list2)).T
		train_sel_list1, train_sel_list2 = np.asarray(train_sel_list1), np.asarray(train_sel_list2)
		train_sel_list = np.hstack((train_sel_list1[:,np.newaxis],train_sel_list2))

		# np.savetxt('%s_serial_encode1.txt'%(annot1),train_sel_list,fmt='%d',delimiter='\t')

		return True

	def sample_region(self,sample_weight,sel_num,thresh=0.7,sel_ratio=1,type_id=1):
		
		select_serial = utility_1.sample_region(sample_weight,sel_num,
													thresh,sel_ratio,type_id)

		return select_serial

	# prepare sequence data
	# type_id: Q1 or Q2, Q1: across chromosomes, Q2: by chromosome
	# type_id1: chromosome set
	def select_region1(self,select_config):

		run_idlist,filename_list = select_config['run_idlist'], select_config['filename_list']
		output_filename_list, output_filename = select_config['output_filename_list'], select_config['output_filename']
		type_id, type_id1 = select_config['quantile_typeid'], select_config['chrom_typeid']
		thresh1, thresh2 = select_config['thresh1'], select_config['thresh2']
		load = select_config['load']

		num1 = len(filename_list)
		b1 = np.where((self.chrom!='chrX')&(self.chrom!='chrY')&(self.chrom!='chrM'))[0]
		ref_chrom, ref_start, ref_serial = self.chrom[b1], self.start[b1], self.serial[b1]
		# num_sameple = len(ref_chrom)
		# mtx1 = np.zeros((num_sameple,num1))
		# mask = np.zeros_like(mtx1,dtype=np.int8)
		vec1 = ['Q1','Q2']
		cnt1 = 0
		type_id2 = 1 # 0: training, 1: test, 2: valid
		if (load<=1) or (os.path.exists(output_filename)==False):
			for i in range(num1):
				# if load==0:
				# 	filename_list1, output_filename1 = filename_list[i], output_filename_list[i]
				# 	print(filename_list1)
				# 	data1 = select_region1_merge(filename_list1,output_filename1,type_id1=type_id1,type_id2=type_id2)
				# elif load==1:
				# 	data1 = pd.read_csv(output_filename1,sep='\t')
				# else:
				# 	break

				filename_list1, output_filename1 = filename_list[i], output_filename_list[i]
				if (load==0) or (os.path.exists(output_filename1)==False):
					print(filename_list1)
					data1, chrom_numList = select_region1_merge(filename_list1,output_filename1,type_id1=type_id1,type_id2=type_id2)
				else:
					data1 = pd.read_csv(output_filename1,sep='\t')

				# data2 = data1.loc[:,['chrom','start','serial','Q1','Q2']]
				# data2 = data1.loc[:,['chrom','start','serial','Q2']]
				# data2 = data1.loc[:,['chrom','start','serial','signal','predicted_signal','predicted_attention','Q1','Q2']]
				data2 = data1
				print(output_filename1,list(data2),data2.shape)
				t_serial = data2['serial']
				# result = pd.merge(left, right, left_index=True, right_index=True, how='outer')
				t_score = data2[vec1[type_id]]
				# b1 = utility_1.search_Idx(ref_serial,t_serial)
				# mtx1[b1,i] = t_score
				# b1 = mapping_Idx(ref_serial,t_serial)[0]
				# print('mapping',len(ref_serial),len(t_serial))
				# b2 = np.where(b1>=0)[0]
				# if len(b2)!=len(b1):
				# 	print('error! select_region1',len(b1),len(b2))
				# 	return

				if i==0:
					ref_chrom1, ref_start1, ref_serial1 = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['serial'])
					signal1_1 = np.asarray(data2['signal'])
					num_sameple = len(ref_chrom1)
					mtx1 = np.zeros((num_sameple,num1),dtype=np.float32)
					mtx1[:,i] = t_score
					print('ref_serial,ref_serial1',len(ref_serial),len(ref_serial1))
				else:
					print(filename_list1, output_filename1)
					print('ref_serial1,t_serial',len(ref_serial1),len(t_serial))
					b1 = utility_1.search_Idx(ref_serial1,t_serial)
					mtx1[b1,i] = t_score

		# if load<=1:
			fields = ['chrom','start','serial','signal']
			data_1 = pd.DataFrame(columns=fields)
			# data_1['chrom'], data_1['start'], data_1['serial'] = ref_chrom, ref_start, ref_serial
			data_1['chrom'], data_1['start'], data_1['serial'] = ref_chrom1, ref_start1, ref_serial1

			signal1 = self.signal.copy()
			b_1 = mapping_Idx(self.serial,ref_serial1)
			b_2 = np.where(b_1<0)[0]
			if len(b_2)>0:
				print('error!', len(self.serial), len(ref_serial1), len(b_2))
			signal1 = signal1[b_1]
			data_1['signal'] = signal1

			# b1 = utility_1.search_Idx(ref_serial,ref_serial1)
			for l in range(0,num1):
				# data_1['train%d'%(run_idlist[l])] = mtx1[b1,l]
				data_1['train%d'%(run_idlist[l][0])] = mtx1[:,l]
			if output_filename=='':
				print('error! select_region1')
				output_filename = 'temp1_%d_%d.txt'%(type_id,type_id1)	# multiple runs
				print(output_filename)
			data_1.to_csv(output_filename,index=False,sep='\t')
		else:
			data_1 = pd.read_csv(output_filename,sep='\t')
			ref_chrom1, ref_start1, ref_serial1 = np.asarray(data_1['chrom']), np.asarray(data_1['start']), np.asarray(data_1['serial'])
			signal1 = np.asarray(data_1['signal'])
			colnames = list(data_1)
			mtx1 = np.asarray(data_1.loc[:,colnames[4:]],dtype=np.float32)

		print('signal',np.max(signal1),np.min(signal1))
		mask = np.int8(mtx1>thresh1)
		s1 = np.sum(mask,axis=1)
		if thresh2>=0:
			if thresh2==0:
				thresh_2 = 1
			else:
				thresh_2 = num1*thresh2

			b1 = np.where(s1>thresh_2)[0]
			print(thresh2,len(b1),len(s1))
			t_chrom1, t_serial1, t_signal1 = ref_chrom1[b1], ref_serial1[b1], signal1[b1]
			mtx1 = mtx1[b1]
			s1 = s1[b1]
		else:
			t_chrom1, t_serial1 = ref_chrom1, ref_serial1
			t_signal1 = signal1

		# train_sel_list = np.zeros((num2,4)) # chrom, serial, max, median
		# chrom1 = [str1[str1.find('chr')+3:] for str1 in t_chrom1]
		# chrom1 = np.int32(chrom1)
		chrom1 = t_chrom1
		t_max = np.max(mtx1,axis=1)
		t_median = np.median(mtx1,axis=1)
		t_mean = np.mean(mtx1,axis=1)
		t_min = np.min(mtx1,axis=1)
		quantile_thresh = 0.75
		t_sel1 = np.quantile(mtx1,quantile_thresh,axis=1)
		print('ref_chrom, sel_chrom', len(ref_chrom1), len(chrom1))
		t_column = 'sel_num_%.2f'%(thresh1)
		t_column1 = 'quantile_%.2f'%(quantile_thresh)
		fields = ['chrom','serial','signal','max','min','median',t_column]
		data_2 = pd.DataFrame(columns=fields)
		data_2['chrom'], data_2['serial'] = t_chrom1, t_serial1
		data_2['max'], data_2['median'], = t_max, t_median
		data_2['min'], data_2['mean'] = t_min, t_mean
		data_2['signal'] = t_signal1
		data_2[t_column] = s1
		data_2[t_column1] = t_sel1
		# output_filename1 = 'temp2_%d_%d.%d_%d.txt'%(run_idlist[0][0],run_idlist[0][1],type_id,type_id1)
		output_filename1 = select_config['output_filename1']
		data_2.to_csv(output_filename1,index=False,sep='\t') # multiple runs
		print('selected regions %d from selected runs %d'%(len(b1),num1))

		return data_2

	# select region
	def select_region2(self,train_sel_list_ori,select_config):

		# run_idlist,filename_list = select_config['run_idlist'], select_config['filename_list']
		# output_filename_list, output_filename = select_config['output_filename_list'], select_config['output_filename']
		# quantile_typeid, chrom_typeid = select_config['quantile_typeid'], select_config['chrom_typeid']
		# thresh1, thresh2 = select_config['thresh1'], select_config['thresh2']
		data_2 = self.select_region1(select_config)
		colnames = list(data_2)
		sel_list1 = np.asarray(data_2.loc[:,colnames])
		# prob = np.asarray(data_2.loc[:,['max','median']])
		sample_num = len(sel_list1)
		sel_column = select_config['sel_column']
		sample_weight = np.asarray(data_2[sel_column])
		print('sel_list1',sample_num)
		sel_num = sample_num*0.25
		sample_ratio = 1
		type_id = 1
		thresh = 0.7
		select_serial = self.sample_region(sample_weight,sel_num,thresh,sample_ratio,type_id)
		print('select_serial',len(select_serial))

		id1 = mapping_Idx(train_sel_list_ori[:,1],sel_list1[select_serial,1])
		b1 = np.where(id1>=0)[0]
		id1 = id1[b1]
		train_sel_list_ori = train_sel_list_ori[id1]
		self.sample_weight1 = sel_list1[select_serial[b1]] # chrom, serial, weight1, weight2
		print('select_serial',len(select_serial))

		data_3 = data_2.loc[select_serial,colnames]
		data_3.to_csv('temp3_select_serial_%d_%d.txt'%(type_id,self.run_id),index=False,sep='\t')

		return train_sel_list_ori

	def sample_path(self,select_config,gap_tol=20):
		
		data_2 = self.select_region1(select_config)
		colnames = list(data_2)
		sel_list1 = np.asarray(data_2.loc[:,colnames])
		# prob = np.asarray(data_2.loc[:,['max','median']])
		sample_num = len(sel_list1)
		sel_column = select_config['sel_column']
		sample_weight = np.asarray(data_2[sel_column])

		chrom, serial = np.asarray(data_2['chrom']), np.asarray(data_2['serial'])
		chrom1 = np.zeros(sample_num,dtype=np.int64)

		chrom_vec = np.unique(chrom)
		for chrom_id in chrom_vec:
			try:
				id1 = chrom_id.find('chr')
				if id1>=0:
					chrom_id1 = int(chrom_id[3:])
					b1 = np.where(chrom==chrom_id)
					chrom1[b1] = chrom_id1
			except:
				continue

		b1 = np.where(chrom1>0)[0]
		chrom1, serial = chrom1[b1], serial[b1]
		idx_sel_list = np.vstack((chrom1,serial)).T

		print('sel_list1',sample_num)
		sel_num = sample_num*0.25
		sample_ratio = 1
		type_id = 1
		thresh = 0.75
		select_serial = self.sample_region(sample_weight,sel_num,thresh,sample_ratio,type_id,thresh_1=0.75)
		print('select_serial',len(select_serial))

		id1 = mapping_Idx(serial,select_serial)

		idx_sel_list1 = idx_sel_list[id1]

		seq_list = generate_sequences(idx_sel_list, gap_tol=gap_tol, region_list=[])

		return seq_list

	# prepare sequencde data
	# return x_train1_trans, y_signal_train1
	def prep_data_sequence_1(self,select_config={}):

		# rng_state = np.random.get_state()
		# if os.path.exists(filename1)==True:
		# 	print("loading data...")
		# 	data1 = np.load(filename1,allow_pickle=True)
		# 	data_1 = data1[()]
		# 	x_train1_trans, train_sel_list_ori = np.asarray(data_1['x1']), np.asarray(data_1['idx'])
		# 	print('train_sel_list',train_sel_list_ori.shape)
		# 	print('x_train1_trans',x_train1_trans.shape)
		# else:
		# 	print("data not found!")
		# 	return

		if self.species_id=='mm10':
			annot1 = '%s_%d'%(self.species_id,self.cell_type1)
		else:
			annot1 = self.species_id

		print('signal',np.max(self.signal),np.min(self.signal))
		train_sel_list_ori = np.loadtxt('%s_serial_encode1.txt'%(annot1),delimiter='\t')
		train_sel_list_ori = train_sel_list_ori[:,0:2]
		if len(select_config)>0:
			train_sel_list_ori = self.select_region2(train_sel_list_ori,select_config)

		train_sel_list, val_sel_list, test_sel_list = self.prep_training_test(train_sel_list_ori)
		print('signal',np.max(self.signal),np.min(self.signal))
		print(len(train_sel_list),len(val_sel_list),len(test_sel_list))

		self.train_sel_list = train_sel_list_ori
		
		list1 = [train_sel_list,val_sel_list,test_sel_list]
		# keys = ['train','valid','test']
		keys = ['train','valid']
		num1 = len(keys)
		# self.bin_size = 5000
		print('bin_size',self.bin_size)
		seq_len = self.bin_size

		if self.train==0:
			for i1 in range(0,num1):
				i = keys[i1]
				self.x[i] = []
				self.y[i] = self.y_signal[i]
			return

		for i1 in range(0,num1):
			i = keys[i1]
			idx_sel_list = list1[i1]

			chrom1 = np.int64(idx_sel_list[:,0])
			num_sample1 = len(idx_sel_list)
			chrom_vec = np.unique(chrom1)
			# data_mtx = np.zeros((num_sample1,seq_len,4),dtype=np.float32)
			data_mtx = np.zeros((num_sample1,seq_len,4),dtype=np.int8)
			# data_mtx = []
			serial2 = []

			for t_chrom in chrom_vec:
				b1 = np.where(chrom1==t_chrom)[0]
				filename1 = '%s_chr%d_encoded1.h5'%(annot1,t_chrom)
				# data1 = np.load(filename1,allow_pickle=True)
				# data1 = data1[()]
				# seq1 = np.asarray(data1['vec'])
				# serial1 = data1['serial']

				with h5py.File(filename1,'r') as fid:
					serial1_ori = fid["serial"][:]
					seq1 = fid["vec"][:]
				
				serial1 = serial1_ori[:,0]
				print(serial1.shape, seq1.shape)
				id1 = mapping_Idx(serial1,idx_sel_list[b1,1])
				b2 = np.where(id1<0)[0]
				if len(b2)>0:
					print('error!',t_chrom,len(b2))
					return
				data_mtx[b1] = seq1[id1]
				# data_mtx.extend(seq1[id1])
				serial2.extend(b1)
				print(t_chrom,len(serial1),len(b1),b1[0:20])
				# print(b1[0:20],idx_sel_list[b1,1])

			# temp1 = np.asarray(data_mtx).copy()
			# serial2 = np.asarray(serial2)
			# temp1[serial2] = data_mtx
			# self.x[i] = np.asarray(data_mtx)
			self.x[i] = data_mtx
			self.y[i] = self.y_signal[i]
			print(keys[i1], self.x[i].shape, self.y[i].shape)

		return True

	# prepare sequence data
	def prep_data_sequence_chrom(self,idx_sel_list,intermediate_layer,pre_config):

		chrom1 = np.int64(idx_sel_list[:,0])
		num_sample1 = len(idx_sel_list)
		chrom_vec = np.unique(chrom1)
		# data_mtx = np.zeros((num_sample1,seq_len,4),dtype=np.float32)
		print(num_sample1)
		# data_mtx = np.zeros((num_sample1,seq_len,4),dtype=np.int8)
		# data_mtx = []
		# serial2 = []
		cnt1 = -1

		for t_chrom in chrom_vec:
			b1 = np.where(chrom1==t_chrom)[0]
			cnt1 += 1
			# filename1 = 'chr%d_encoded1.h5'%(t_chrom)
			# data1 = np.load(filename1,allow_pickle=True)
			# data1 = data1[()]
			# seq1 = np.asarray(data1['vec'])
			# serial1 = data1['serial']

			# filename2 = 'chr%s_encoded2_%d.h5'%(t_chrom,pre_config['run_id'])
			filename2 = '%s_chr%s_encoded2_%d.h5'%(self.species_id,t_chrom,pre_config['run_id'])
			if os.path.exists(filename2)==False:
				# filename1 = 'chr%s_encoded1.h5'%(t_chrom)
				filename1 = '%s_chr%s_encoded1.h5'%(self.species_id,t_chrom)
				with h5py.File(filename1,'r') as fid:
					serial1 = fid["serial"][:]
					seq1 = fid["vec"][:]

					print(serial1.shape, seq1.shape)
					# id1 = mapping_Idx(serial1,idx_sel_list[b1,1])
					id1 = utility_1.search_Idx(serial1,idx_sel_list[b1,1])
					t_data_mtx = seq1[id1]
					start = time.time()
					feature1 = intermediate_layer.predict(t_data_mtx)
					stop = time.time()
					print('feature1',t_chrom,feature1.shape,stop-start)
					if cnt1==0:
						feature_dim = feature1.shape[1]
						data_mtx = np.zeros((num_sample1,feature_dim))
					data_mtx[b1] = feature1
					# data_mtx.extend(seq1[id1])
					# serial2.extend(b1)
					print(t_chrom,len(serial1),len(b1))
					print(b1[0:20],idx_sel_list[b1,1])

				with h5py.File(filename2,'w') as fid:
					t_serial = idx_sel_list[b1,0]
					fid.create_dataset("serial", data=t_serial, compression="gzip")
					fid.create_dataset("feature1", data=feature1, compression="gzip")

			else:
				with h5py.File(filename2,'r') as fid:
					serial1 = fid["serial"][:]
					feature1 = fid["feature1"][:]
					id1 = utility_1.search_Idx(serial1,idx_sel_list[b1,1])
					if cnt1==0:
						feature_dim = feature1.shape[1]
						data_mtx = np.zeros((num_sample1,feature_dim))
					data_mtx[b1] = feature1[id1]

		return data_mtx

	# prepare sequence data
	# return x_train1_trans, y_signal_train1
	def prep_data_sequence_2(self,pre_config,select_config={}):

		if self.species_id=='mm10':
			annot1 = '%s_%d'%(self.species_id,self.cell_type1)
		else:
			annot1 = self.species_id

		train_sel_list_ori = np.loadtxt('%s_serial_encode1.txt'%(annot1),delimiter='\t')
		if len(select_config)>0:
			train_sel_list_ori = self.select_region2(train_sel_list_ori,select_config)

		train_sel_list, val_sel_list, test_sel_list = self.prep_training_test(train_sel_list_ori)

		self.train_sel_list = train_sel_list_ori
		
		list1 = [train_sel_list,val_sel_list,test_sel_list]
		# keys = ['train','valid','test']
		keys = ['train','valid']
		num1 = len(keys)
		# self.bin_size = self.resolution
		seq_len = self.bin_size
		print('bin_size',self.bin_size)

		context_size = pre_config['context_size']
		method1 = self.method
		predict_context1 = self.predict_context
		self.method = 62
		self.predict_context = 1
		model = self.get_model(pre_config,context_size)
		print('loading model...')
		model_path = pre_config['model_path']
		model.load_weights(model_path)
		self.method = method1
		self.predict_context = predict_context1
		self.model_load1 = model
		self.seq_list = dict()

		# layer_name = 'activation3_1' # 'dense3_1', 'bnorm3_1'
		layer_name = pre_config['layer_name']
		intermediate_layer = Model(inputs=model.input,
								 outputs=model.get_layer(layer_name).output)

		flag = 1
		# if self.method<5 or self.method in [56]:
		# 	flag = 0
		if (self.method<10) or (self.method in self.method_vec[2]):
			flag = 0

		if self.train==0:
			for i1 in range(0,num1):
				i = keys[i1]
				self.x[i] = []
				self.y[i] = self.y_signal[i]
			return

		for i1 in range(0,num1):
			i = keys[i1]
			idx_sel_list = list1[i1]

			chrom1 = np.int64(idx_sel_list[:,0])
			num_sample1 = len(idx_sel_list)
			chrom_vec = np.unique(chrom1)
			# data_mtx = np.zeros((num_sample1,seq_len,4),dtype=np.float32)
			print(num_sample1,seq_len)
			# data_mtx = np.zeros((num_sample1,seq_len,4),dtype=np.int8)
			# data_mtx = []
			# serial2 = []

			cnt1 = -1

			filename2 = 'encoded2_%d.h5'%(pre_config['run_id'])
			if os.path.exists(filename2)==True:
				with h5py.File(filename2,'r') as fid:
					serial1 = fid["serial"][:]
					feature1 = fid["feature1"][:]
					id1 = utility_1.search_Idx(serial1,idx_sel_list[:,1])
					data_mtx = feature1[id1]
			else:
				data_mtx = self.prep_data_sequence_chrom(idx_sel_list,intermediate_layer,pre_config)

			feature1 = data_mtx
			print('feature1',feature1.shape)

			if flag==0:
				self.x[i] = feature1
				self.y[i] = self.y_signal[i]
				print(keys[i1], self.x[i].shape, self.y[i].shape)
			else:
				# idx_sel_list = self.train_sel_list[i]
				# x_train1_trans_vec[i] = feature1
				start = time.time()
				self.seq_list[i] = generate_sequences(idx_sel_list,region_list=self.region_boundary)
				print(len(self.seq_list[i]))
				stop = time.time()
				print('generate_sequences', stop-start)

				# generate initial state index
				# self.init_id = dict()
				# self.init_index(keys)
				start = time.time()
				x, y, self.vec[i], self.vec_local[i] = sample_select2a1(feature1,self.y_signal[i],
															idx_sel_list, self.seq_list[i], self.tol, self.flanking)
				stop = time.time()
				print('sample_select2a1',stop-start)

				self.x[i] = x
				self.y[i] = y

		return True

	# return x_train1_trans, y_signal_train1
	def prep_data_sequence_3(self,pre_config,test_chromvec,model=None,select_config={}):

		# train_sel_list_ori = np.loadtxt('serial_encode1.txt',delimiter='\t')
		
		if self.species_id=='mm10':
			annot1 = '%s_%d'%(self.species_id,self.cell_type1)
		else:
			annot1 = self.species_id

		if model==None:
			context_size = pre_config['context_size']
			method1 = self.method
			predict_context1 = self.predict_context
			self.method = 62
			model = self.get_model(pre_config,context_size)
			print('loading model...')
			model_path = pre_config['model_path']
			model.load_weights(model_path)
			self.method = method1
			self.predict_context = predict_context1
			self.model_load1 = model

			seq_len = self.bin_size
			print('bin_size',self.bin_size)

		layer_name = pre_config['layer_name']
		intermediate_layer = Model(inputs=model.input,
								 outputs=model.get_layer(layer_name).output)

		run_id = pre_config['run_id']
		list1, list2 = [], []
		filename2 = 'encoded2_%d.h5'%(run_id)
		if os.path.exists(filename2)==False:
			for t_chrom in test_chromvec:
				# b1 = np.where(chrom1==int(t_chrom))[0]
				if species_id=='mm10':
					annot2 = '%s_chr%s'%(species_id,t_chrom)
				else:
					annot2 = 'chr%s'%(t_chrom)

				filename1 = '%s_encoded1.h5'%(annot2,t_chrom)
				with h5py.File(filename1,'r') as fid:
					serial1 = fid['serial'][:]
					seq1 = fid['vec'][:]
					feature1 = intermediate_layer.predict(seq1)
					print('feature1',t_chrom,feature1.shape)

				# with h5py.File(filename2,'a') as fid:
				# 	# t_serial = idx_sel_list[b1,0]
				# 	fid.create_dataset("serial", data=serial1, compression="gzip")
				# 	fid.create_dataset("feature1", data=feature1, compression="gzip")

				# list1.append([serial1,feature1])
				list1.extend(feature1)
				temp1 = np.asarray([t_chrom]*len(serial1))
				serial_1 = np.hstack((temp1[:,np.newaxis],serial1)).T
				list2.extend(serial_1)

			feature1 = np.asarray(list1)
			t_serial1 = serial_1[:,0:2]
			with h5py.File(filename2,'w') as fid:
				# t_serial = idx_sel_list[b1,0]
				fid.create_dataset("serial", data=serial_1, compression="gzip")
				fid.create_dataset("feature1", data=np.asarray(list1), compression="gzip")
		else:
			with h5py.File(filename2,'r') as fid:
				serial1 = fid["serial"][:]
				feature1 = fid["feature1"][:]
				t_serial1 = serial1[:,0:2]

		# for t_chrom in test_chromvec:
		# 	# b1 = np.where(chrom1==int(t_chrom))[0]
		# 	if species_id=='mm10':
		# 		annot2 = '%s_chr%s'%(species_id,t_chrom)
		# 	else:
		# 		annot2 = 'chr%s'%(t_chrom)
				
		# 	filename2 = '%s_encoded2_%d.h5'%(annot2,run_id)
		# 	if os.path.exists(filename2)==False:
		# 		filename1 = '%s_encoded1.h5'%(annot2,t_chrom)
		# 		with h5py.File(filename1,'r') as fid:
		# 			serial1 = fid['serial'][:]
		# 			seq1 = fid['vec'][:]
		# 			feature1 = intermediate_layer.predict(seq1)
		# 			print('feature1',t_chrom,feature1.shape)

		# 		# with h5py.File(filename2,'a') as fid:
		# 		# 	# t_serial = idx_sel_list[b1,0]
		# 		# 	fid.create_dataset("serial", data=serial1, compression="gzip")
		# 		# 	fid.create_dataset("feature1", data=feature1, compression="gzip")
		# 	else:
		# 		with h5py.File(filename2,'r') as fid:
		# 			serial1 = fid["serial"][:]
		# 			feature1 = fid["feature1"][:]

		# 	# list1.append([serial1,feature1])
		# 	list1.extend(features1)
		# 	temp1 = np.asarray([t_chrom]*len(serial1))
		# 	serial_1 = np.vstack((temp1,serial1[:,0])).T
		# 	list2.extend(serial_1)

		return feature1, t_serial1

	# return x_train1_trans, y_signal_train1
	def prep_data_sequence_1_1(self,pre_config,chromvec,model=None,select_config={},type_id=0):

		# train_sel_list_ori = np.loadtxt('serial_encode1.txt',delimiter='\t')
		
		if model==None:
			context_size = pre_config['context_size']
			method1 = self.method
			self.method = 62
			model = self.get_model(pre_config,context_size)
			print('loading model...')
			model_path = pre_config['model_path']
			model.load_weights(model_path)
			self.method = method1
			self.model_load1 = model

		layer_name = pre_config['layer_name']
		intermediate_layer = Model(inputs=model.input,
								 outputs=model.get_layer(layer_name).output)

		run_id = pre_config['run_id']
		list1, list2 = [], []
		for t_chrom in chromvec:
			# b1 = np.where(chrom1==int(t_chrom))[0]
			filename2 = 'chr%s_encoded2_%d.h5'%(t_chrom,run_id)
			if os.path.exists(filename2)==False:
				filename1 = 'chr%s_encoded1.h5'%(t_chrom)
				with h5py.File(filename1,'r') as fid:
					serial1 = fid["serial"][:]
					seq1 = fid["vec"][:]
					feature1 = intermediate_layer.predict(seq1)
					print('feature1',t_chrom,feature1.shape)

				if type_id==0:
					with h5py.File(filename2,'a') as fid:
						# t_serial = idx_sel_list[b1,0]
						fid.create_dataset("serial", data=serial1, compression="gzip")
						fid.create_dataset("feature1", data=feature1, compression="gzip")
				else:
					list1.extend(feature1)
					list2.extend(serial1)

		if type_id!=0:
			feature1 = np.asarray(list1)
			serial1 = np.asarray(list2)
			filename2 = 'encoded2_%d.h5'%(run_id)
			with h5py.File(filename2,'a') as fid:
				fid.create_dataset("serial", data=serial1, compression="gzip")
				fid.create_dataset("feature1", data=feature1, compression="gzip")	

		return True

	# feature dimension reduction
	def dimension_reduction_1(self,x1,serial1,sel_idx,feature_dim_vec,type_id,filename_prefix,output_filename='',save_mode=1):

		vec1 = ['PCA','Incremental PCA','Kernel PCA','Sparse PCA','SVD',
				'GRP','SRP','MDS','ISOMAP','Minibatch','ICA','tSNE','LLE','Encoder']

		x_trans_dict = dict()
		x = x1[sel_idx]
		for feature_dim in feature_dim_vec:
			start = time.time()
			x_trans = self.dimension_reduction(x,feature_dim,shuffle=0,sub_sample=-1,type_id=type_id,filename_prefix=filename_prefix)
			stop = time.time()
			print("%d feature transform %s"%(feature_dim, vec1[type_id]),stop - start)

			x1_trans = self.dimension_model.transform(x1)
			x_trans_dict.update({feature_dim:x1_trans})

			if save_mode==1:
				if output_filename=='':
					output_filename = '%s_%d_%d_trans.h5'%(filename_prefix,type_id,feature_dim)

				print(output_filename)
				with h5py.File(output_filename,'w') as fid:
					fid.create_dataset("serial", data=serial1, compression="gzip")
					fid.create_dataset("vec", data=x1_trans, compression="gzip")
				output_filename = ''

		return x_trans_dict

	# load kmer frequency feature
	# return x_train1_trans, y_signal_train1
	def prep_data_sequence_kmer(self,kmer_size,output_filename='',filename_prefix='',chrom_vec=[],save_mode=1,load_type=1):

		if (os.path.exists(output_filename)) and (load_type==1):
			with h5py.File(output_filename,'r') as fid:
				ref_serial = fid["serial"][:]
				f_list = fid["vec"][:]
				print(len(ref_serial),f_list.shape)

				return f_list, ref_serial

		if self.species_id=='mm10':
			file_path = './mouse'
			filename1 = '%s/mm10_5k_seq_genome%d_1.txt'%(file_path,self.cell_type1)
			filename2 = '%s/mm10_5k_serial.bed'%(file_path)
			if filename_prefix=='':
				filename_prefix = 'test_%s_genome%d'%(self.species_id,self.cell_type1)
			if len(chrom_vec)==0:
				chrom_vec = range(1,20)
			if output_filename=='':
				output_filename = 'test_%s_genome%d_kmer%d.h5'%(self.species_id,self.cell_type1,kmer_size)
		else:
			filename1 = './hg38_5k_seq'
			filename2 = 'hg38_5k_serial.bed'
			if filename_prefix=='':
				filename_prefix = 'test_%s'%(self.species_id)
			if len(chrom_vec)==0:
				chrom_vec = range(1,23)
			if output_filename=='':
				output_filename = 'test_%s_kmer%d.h5'%(self.species_id,kmer_size)

		f_list, ref_serial = utility_1.prep_data_sequence_kmer_chrom(filename1,filename2,kmer_size,
																	chrom_vec=chrom_vec,
																	filename_prefix=filename_prefix,save_mode=save_mode)
		print(kmer_size,f_list.shape,len(ref_serial))

		# save_mode = 0
		if save_mode==1:
			with h5py.File(output_filename,'w') as fid:
				fid.create_dataset("serial", data=ref_serial, compression="gzip")
				fid.create_dataset("vec", data=f_list, compression="gzip")

		return f_list, ref_serial

	# load kmer frequency feature
	# return x_train1_trans, y_signal_train1
	def prep_data_sequence_kmer_transform(self,kmer_size,output_filename='',save_mode=1,ratio=1,select_config={}):

		if 'chrom_vec' in select_config:
			chrom_vec = select_config['chrom_vec']
		else:
			chrom_vec = []

		if self.species_id=='mm10':
			annot1 = '%s_%d_kmer%d'%(self.species_id,self.cell_type1,kmer_size)
			serial2 = self.serial
			chrom2 = self.chrom
			chrom_num = 19
		else:
			annot1 = '%s_kmer%d'%(self.species_id,kmer_size)
			serial2 = self.serial_ori
			chrom2 = self.chrom_ori
			chrom_num = 22

		x1_ori, ref_serial = self.prep_data_sequence_kmer(kmer_size,output_filename,save_mode=1)
	
		b2 = utility_1.find_serial(chrom2,chrom_num,chrom_vec=chrom_vec)
		serial2 = serial2[b2]

		id1 = mapping_Idx(ref_serial,serial2)
		b1 = (id1>=0)
		n1, n2 = np.sum(b1), len(serial2)
		if n1!=n2:
			print('error!',n1,n2)
			return

		id1 = id1[b1]

		x1, serial1 = x1_ori[id1], ref_serial[id1]
		print('load kmer frequence', x1.shape, len(id1))
		region_num = len(serial1)

		if 'region_serial' in select_config:
			t_serial = select_config['region_serial'] # serial
			id1 = mapping_Idx(serial1,t_serial)
			id1 = id1[id1>=0]
			# x1, serial1 = x1[id1], serial1[id1]
			# region_num = len(serial1)
			# sel_idx = np.arange(region_num)
			# np.random.shuffle(sel_idx)
			sel_idx = id1
		else:
			sel_idx = np.arange(region_num)

		np.random.shuffle(sel_idx)
		region_num1 = len(sel_idx)
		if ratio<1:
			sample_num = int(region_num1*ratio)
			sel_idx = sel_idx[0:sample_num]

		print('sel_idx',region_num,len(sel_idx))

		type_idvec = select_config['dimension_reduction_type']
		feature_dim_vec = select_config['dimension_reduction_featuredim']

		num1 = len(type_idvec)
		x_trans_dict = dict()
		for i in range(num1):
			type_id1 = type_idvec[i]
			feature_dim_vec1 = feature_dim_vec[i]
			x_trans_dict1 = self.dimension_reduction_1(x1,serial1,sel_idx,feature_dim_vec1,type_id1,filename_prefix=annot1)
			x_trans_dict[type_id1] = x_trans_dict1

		x_trans_dict.update({'serial':serial1})

		if save_mode==0:
			return x_trans_dict

		return True

	# load motif feature
	# return x_train1_trans, y_signal_train1
	def prep_data_sequence_motif(self,filename2,select_config={}):

		# train_sel_list_ori = np.loadtxt('serial_encode1.txt',delimiter='\t')

		motif_data_ori = pd.read_csv(filename2,sep='\t')
		colnames = list(motif_data_ori)
		print(colnames)
		motif_name = np.asarray(colnames[3:])
		chrom2, start2, stop2 = np.asarray(motif_data_ori[colnames[0]]), np.asarray(motif_data_ori[colnames[1]]), np.asarray(motif_data_ori[colnames[2]])

		if self.species_id=='mm10':
			filename_1 = '/work/magroup/yy3/data2/genome/mm10/mm10.chrom.sizes'
			chrom_num = 19
			annot1 = '%s_%d_motif'%(self.species_id,self.cell_type1)
		else:
			filename_1 = '/work/magroup/yy3/data2/genome/hg38.chrom.sizes'
			chrom_num = 22
			annot1 = '%s_motif'%(self.species_id)
			
		ref_serial, start_vec = generate_serial_start(filename_1,chrom2,start2,stop2,chrom_num=chrom_num)
		# motif_data['serial'] = ref_serial

		b1 = np.where(ref_serial>=0)[0]
		b2 = utility_1.find_serial(chrom2,chrom_num)
		b1 = np.intersect1d(b1,b2)

		ref_serial = ref_serial[b1]
		mtx2_ori = motif_data_ori.loc[b1,motif_name]

		region_len = stop2-start2
		region_len1 = region_len/1000
		region_len1 = region_len1[b1]

		b2 = np.where(region_len!=np.median(region_len))[0]
		print(np.max(region_len),np.min(region_len),len(b1))
		if len(b2)>chrom_num:
			print('error!',len(b1),chrom2[b2])

		print('motif',len(motif_name),mtx2_ori.shape)
		region_num, motif_num = mtx2_ori.shape[0], mtx2_ori.shape[1]
		motif_data = np.asarray(mtx2_ori)/np.outer(region_len1,np.ones(motif_num))
		print(np.max(motif_data),np.min(motif_data))

		start = time.time()
		type_id = select_config['dimension_reduction_type']
		feature_dim_vec = select_config['dimension_reduction_featuredim']

		id1 = np.random.permutation(region_num)
		x_trans_dict = self.dimension_reduction_1(motif_data,ref_serial,id1,feature_dim_vec,type_id,annot1)
		x_trans_dict.update({'serial':ref_serial})

		return x_trans_dict

	# load motif feature
	def prep_data_sequence_motif_transform(self,pre_config,test_chromvec,model=None,select_config={}):

		# train_sel_list_ori = np.loadtxt('serial_encode1.txt',delimiter='\t')
		
		if self.species_id=='mm10':
			annot1 = '%s_%d'%(self.species_id,self.cell_type1)
		else:
			annot1 = self.species_id

	# prepare data
	def prep_test_data(self):

		x_train1_trans = self.x_train1_trans
		feature_dim_transform = self.feature_dim_transform

		idx = self.idx_list['test']
		idx_sel_list = self.train_sel_list[idx]
		i = 'test'
		self.seq_list[i] = generate_sequences(idx_sel_list,region_list=self.region_boundary)
		print(len(self.seq_list[i]))
		x, y, self.vec[i], self.vec_local[i] = sample_select2a1(x_train1_trans[idx],self.y_signal[i],
												idx_sel_list, self.seq_list[i], self.tol, self.flanking)
		# concate context for baseline methods
		if self.method<=10:
			# x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.2, random_state=42)
			x = x.reshape(x.shape[0],x.shape[1]*x.shape[-1])
			y = y[:,self.flanking]

		self.x[i], self.y[i] = x, y

		return True

	def get_model(self,config,context_size):

		attention = self.attention
		print(self.predict_context, self.attention)
		if self.predict_context==1:
			if attention==1:
				# if self.method==12:
				# 	print('get_model2a1_attention_1_1')
				# 	config['attention2']=0
				# 	model = utility_1.get_model2a1_attention_1_1(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==15:
				# 	print('get_model2a1_attention_1_2')
				# 	model = utility_1.get_model2a1_attention_1_2(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==16:
				# 	print('get_model2a1_attention_1_3')
				# 	model = utility_1.get_model2a1_attention_1_3(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==17:
				# 	print('get_model2a1_attention_1_1')
				# 	config['attention2']=1
				# 	model = utility_1.get_model2a1_attention_1_1(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==18:
				# 	print('get_model2a1_attention_1_2')
				# 	config['attention2']=1
				# 	model = utility_1.get_model2a1_attention_1_2(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==19:
				# 	print('get_model2a1_attention_1_2_select')
				# 	config['attention1']=1
				# 	config['attention2']=1
				# 	model = utility_1.get_model2a1_attention_1_2_select(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==20:
				# 	print('get_model2a1_attention_1_2_select')
				# 	config['attention1']=0
				# 	config['attention2']=1
				# 	model = utility_1.get_model2a1_attention_1_2_select(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==21:
				# 	print('get_model2a1_attention_1_2_2')
				# 	config['attention1']=1
				# 	config['attention2']=1
				# 	model = get_model2a1_attention_1_2_2(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==22:
				# 	print('get_model2a1_attention_1_2_2')
				# 	config['attention1']=0
				# 	config['attention2']=1
				# 	# config['feature_dim_vec'] = [50,25,50,25,0,0]
				# 	model = utility_1.get_model2a1_attention_1_2_2(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==23:
				# 	config['attention1']=1
				# 	print('get_model2a1_attention')
				# 	model = utility_1.get_model2a1_attention1(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==24:
				# 	config['attention1']=0 
				# 	print('get_model2a1_attention')
				# 	model = utility_1.get_model2a1_attention1(context_size,config)		# context self-attention model and prediction per position
				# elif self.method==31:
				# 	config['attention1']=1
				# 	config['attention2']=1
				# 	print('get_model2a1_attention1_1')
				# 	model = utility_1.get_model2a1_attention1_1(context_size,config)
				# elif self.method==35:
				# 	config['attention1']=0
				# 	config['attention2']=1
				# 	print('get_model2a1_attention1_1')
				# 	model = utility_1.get_model2a1_attention1_1(context_size,config)
				if self.method==32:
					config['attention1']=0
					config['attention2']=1
					config['select2']=0
					# print('get_model2a1_attention_1_2_2_sample')
					model = utility_1.get_model2a1_attention_1_2_2_sample(context_size,config)
				elif self.method==52:
					config['attention1']=0
					config['attention2']=1
					config['select2']=1
					# print('get_model2a1_attention_1_2_2_sample')
					model = utility_1.get_model2a1_attention_1_2_2_sample(context_size,config)
				elif self.method==58:
					config['attention1']=1
					config['attention2']=1
					config['select2']=1
					config['sample_attention']=1
					# print('get_model2a1_attention1_1')
					model = utility_1.get_model2a1_attention_1_2_2_sample(context_size,config)
				elif self.method==60:
					config['attention1']=1
					config['attention2']=0
					config['select2']=1
					config['sample_attention']=1
					# print('get_model2a1_attention1_1')
					model = utility_1.get_model2a1_attention_1_2_2_sample(context_size,config)
				elif self.method==62:
					config['attention1']=1
					config['attention2']=0
					# print('get_model2a1_convolution')
					model = utility_1.get_model2a1_convolution(config)
				# elif self.method==51:
				# 	config['attention1']=0
				# 	config['attention2']=1
				# 	config['select2']=1
				# 	config['sample_attention']=0
				# 	# print('get_model2a1_attention_1_2_2_sample')
				# 	model = utility_1.get_model2a1_attention_1_2_2_sample(context_size,config)
				# elif self.method==53:
				# 	config['attention1']=1
				# 	config['attention2']=1
				# 	config['select2']=1
				# 	config['sample_attention']=0
				# 	# print('get_model2a1_attention1_1')
				# 	model = utility_1.get_model2a1_attention1_1(context_size,config)
				# elif self.method==55:
				# 	config['attention1']=0
				# 	config['attention2']=1
				# 	config['select2']=1
				# 	config['sample_attention']=0
				# 	model = utility_1.get_model2a1_attention1_1(context_size,config)
				# elif self.method==57:
				# 	config['attention1']=1
				# 	config['attention2']=0
				# 	config['select2']=1
				# 	config['sample_attention']=0
				# 	model = utility_1.get_model2a1_attention1_1(context_size,config)
				else:
					print('get_model2a1_attention')
					model = utility_1.get_model2a1_attention(context_size,config)		# context self-attention model and prediction per position
			else:
				print('get_model2a1_sequential')
				model = utility_1.get_model2a1_sequential(context_size,config)	# context without attention and prediction per position
		else:
			if self.method==56:
				# config['attention1']=1
				# config['attention2']=1
				config['select2']=1
				model = utility_1.get_model2a1_attention_1_2_2_single(config)
			elif (attention==1) and (self.method==2):
				model = utility_1.get_model2a1_attention_1(context_size,config)	# context with attention from intermediate layer
			elif (attention==2) and (self.method==3):
				model = utility_1.get_model2a1_attention_2(context_size,config)	# context with attention from input
			else:
				model = utility_1.get_model2a_sequential(context_size,config)	# context without attention

		return model

	###########################################################
	## compare with other methods
	# baseline method
	# return: trained model
	def compare_single_1(self,x_train,y_train,x_valid,y_valid,type_id=0,model_path1=""):

		# x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.2, random_state=42)
		if 'select_config_comp' in self.config:
			select_config_comp = self.config['select_config_comp']
			max_depth, n_estimators = select_config_comp['max_depth'], select_config_comp['n_estimators']
			if (type_id==1) and (max_depth==None):
				max_depth = 20
		else:
			max_depth, n_estimators = 10, 500

		if type_id==0:
			print("linear regression")
			model = LinearRegression().fit(x_train, y_train)
		elif type_id==1:
			print("xgboost regression")
			model = xgboost.XGBRegressor(colsample_bytree=1,
				 gamma=0,    
				 n_jobs=20,             
				 learning_rate=0.1,
				 max_depth=max_depth,
				 min_child_weight=1,
				 n_estimators=n_estimators,                                                                    
				 reg_alpha=0,
				 reg_lambda=1,
				 objective='reg:squarederror',
				 subsample=1,
				 seed=0)
			print("fitting model...")
			model.fit(x_train, y_train)
		elif type_id==2:
			print("random forest regression")
			model = RandomForestRegressor(
				 n_jobs=20,
				 n_estimators=n_estimators,
				 max_depth=max_depth,
				 random_state=0)
			print("fitting model...")
			model.fit(x_train, y_train)
		else:
			config = self.config
			if self.method==62:
				config['feature_dim'] = 1
			else:
				config['feature_dim'] = x_train.shape[-1]
			BATCH_SIZE = config['batch_size']
			n_epochs = config['n_epochs']
			context_size = 1
			config['context_size'] = context_size
			model = self.get_model(self.config,context_size)
			MODEL_PATH = self.model_path
			if self.train==1:
				earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=0, mode='auto')
				checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
				# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
				model.fit(x_train,y_train,epochs = n_epochs, batch_size = BATCH_SIZE, validation_data = [x_valid,y_valid],
								callbacks=[earlystop,checkpointer])
				# model.load_weights(MODEL_PATH)
				type_id2 = 0
				model_path2 = 'model_%d_%d_%d.h5'%(self.run_id,type_id2,context_size)
				model.save(model_path2)
				model_path2 = MODEL_PATH
				print('loading weights... ', model_path2)
				model.load_weights(model_path2) # load model with the minimum training error
			else:
				if model_path1!="":
					MODEL_PATH = model_path1
				print('loading weights... ', MODEL_PATH)
				model.load_weights(MODEL_PATH)
				# model = keras.models.load_model(MODEL_PATH)

		# y_predicted_valid = self.model.predict(x_valid)
		# # y_predicted_test = self.model_single.predict(x_test)

		# vec1 = []
		# score1 = score_2a(y_valid,y_predicted_valid)
		# # score2 = score_2a(y_test,y_predicted_test)
		# vec1.append(score1)
		# # vec1.append(score2)
		# print(score1)

		# # y_train, y_valid = np.ravel(y_train), np.ravel(y_valid)
		# # temp1 = score_2a(y_valid, y_predicted_valid)
		# # vec1.append(temp1)
		# # print(temp1)

		# dict1 = dict()
		# dict1['vec1'] = vec1
		# dict1['y_valid'], dict1['y_predicted_valid'] = y_valid, y_predicted_valid

		return model

	# concatenate features in the context for the compared methods
	# return: trained model, performance on validation data
	def training_single_1(self,x_train1_trans,y_signal_train1_ori,idx_train,idx_valid,type_id=0):
		
		# train_sel_list = self.idx_sel_list['train']
		tol = self.tol
		L = self.flanking
		run_id = self.run_id
			
		x_train, y_train = self.x['train'], self.y['train']
		x_valid, y_valid = self.x['valid'], self.y['valid']
		print(x_train.shape,x_valid.shape,y_train.shape,y_valid.shape)

		x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[-1])
		x_valid = x_valid.reshape(x_valid.shape[0],x_valid.shape[1]*x_valid.shape[-1])
		y_train = y_train[:,L]
		y_valid = y_valid[:,L]
		print(x_train.shape,x_valid.shape,y_train.shape,y_valid.shape)

		model = self.compare_single_1(x_train,y_train,x_valid,y_valid,type_id)
		
		y_predicted_valid = model.predict(x_valid)

		vec1 = []
		score1 = score_2a(y_valid,y_predicted_valid)
		# score2 = score_2a(y_test,y_predicted_test)
		vec1.append(score1)
		# vec1.append(score2)

		dict1 = dict()
		dict1['vec1'] = vec1
		dict1['y_valid'], dict1['y_predicted_valid'] = y_valid, y_predicted_valid

		return model, dict1

	###########################################################
	## configuration
	def set_predict_type_id(self, predict_type_id):
		self.predict_type_id = predict_type_id

	def set_train_mode(self, train_mode):
		self.train = train_mode

	def set_chromvec_training(self,chrom_vec,run_id=-1,ratio=0.9):
		self.chromvec_sel = chrom_vec
		if run_id>=0:
			self.run_id = run_id
		self.ratio = ratio

	def set_model_path(self,path1):
		self.model_path = path1

	def set_species_id(self,species_id,resolution):

		self.species_id = species_id
		self.resolution = resolution
		print('species_id, resolution', species_id, resolution)

	def set_generate(self,generate,filename=None):

		self.generate = generate
		if filename!=None:
			self.filename_load = str(filename)

		return True

	def set_featuredim_motif(self,feature_dim_motif):

		self.feature_dim_motif = feature_dim_motif

		return True

	###########################################################
	def get_serial(self):

		return self.serial

	###########################################################
	## loading data and data processing
	# load local serial and signal
	# input: filename1: genome position and signal file
	def load_local_signal_1_1(self, filename1):

		file2 = pd.read_csv(filename1,sep='\t')
		colnames = list(file2)
		# col1, col2, col3, col_serial = colnames[0], colnames[1], colnames[2], colnames[3]
		col1, col2, col3, col4 = colnames[0], colnames[1], colnames[2], colnames[3]
		chrom, start, stop = np.asarray(file2[col1]), np.asarray(file2[col2]), np.asarray(file2[col3])
		# bin_size = stop[2]-stop[1]
		bin_size = stop[1]-start[1]
		self.bin_size = bin_size
		# start = stop-bin_size
		signal = np.asarray(file2[col4])

		return chrom, start, stop, signal

	# load local serial and signal
	# intput: filename1: signal file
	#		  genome_file: file of genome size
	# 		  region_list: regions to be excluded
	def load_local_signal_1(self, filename1, colnames, genome_file, chrom_num,
								region_list=[], type_id2=0):

		chrom, start, stop, signal = self.load_local_signal_1_1(filename1)
		default = -10
		b = np.where(signal!=default)[0]
		# deleted regions

		if len(region_list)>0:
			b1 = self.region_search(chrom,start,stop,region_list,type_id2)
			print('region',len(b1))
			print(b1)
			b = np.setdiff1d(b,b1)
		self.chrom, self.start, self.stop, self.signal = chrom[b], start[b], stop[b], signal[b]
			
		# chrom_num = 19
		# filename1 = '/volume01/yy3/seq_data/dl/replication_timing3/mouse/mm10.chrom.sizes'
		# filename1 = '../mm10.chrom.sizes'
		filename1 = genome_file
		serial_vec = generate_serial_local(filename1,self.chrom,self.start,self.stop,chrom_num)
		self.serial = serial_vec
		print('load local serial', self.serial.shape, self.signal.shape)

		fields = ['chrom','start','stop','serial','signal']
		data1 = pd.DataFrame(columns=fields)
		data1['chrom'], data1['start'], data1['stop'], data1['serial'], data1['signal'] = self.chrom, self.start, self.stop, self.serial, self.signal

		# filename2 = 'local_rt_%s.txt'%(self.species_id)
		filename2 = '%s.local_rt.1'%(filename1)
		data1.to_csv(filename2,index=False,sep='\t')

		# self.signal = self.signal_normalize(self.signal,[0,1])
		if self.chrom_num>0:
			chrom_num = self.chrom_num
		else:
			chrom_num = len(np.unique(self.chrom))
		chrom_vec = [str(i) for i in range(1,chrom_num+1)]
		print(self.species_id,chrom_num)
		self.signal = self.signal_normalize_chrom(self.signal,chrom_vec[0,1])
		
		return self.serial, self.signal

	def local_serial_1(self,id1,type_id=0):

		self.chrom, self.start, self.stop, self.serial = self.chrom[id1], self.start[id1], self.stop[id1], self.serial[id1]

		if type_id==0:
			self.signal = self.signal[id1]

		return True

	# signal normalization
	def signal_normalize(self,signal, scale):

		s1, s2 = scale[0], scale[1]

		s_min, s_max = np.min(signal), np.max(signal)
		scaled_signal = s1+(signal-s_min)*1.0/(s_max-s_min)*(s2-s1)

		return scaled_signal

	def signal_scale(self,signal,s_min,s_max,s1,s2,tol1=0.5):

		if s1> -1e-12:
			scaled_signal = s1+(signal-s_min)*1.0/(s_max-s_min)*(s2-s1)
		else:
			middle1 = 0	# middel point of signal
			t_middle1 = 0
			scaled_signal = np.zeros_like(signal)
			b1 = signal>=middle1
			scaled_signal[b1] = t_middle1+(signal[b1]-middle1)*1.0/(s_max-middle1)*(s2-t_middle1)
			b2 = signal<middle1
			scaled_signal[b2] = t_middle1+(signal[b2]-middle1)*1.0/(middle1-s_min)*(t_middle1-s1)

		if 'signal_normalize_clip' in self.config and self.config['signal_normalize_clip']==1:
			scaled_signal[scaled_signal>s2] = s2
			scaled_signal[scaled_signal<s1] = s1
		else:
			s2_1, s1_1 = s2+tol1, s1-tol1
			scaled_signal[scaled_signal>s2_1] = s2_1
			scaled_signal[scaled_signal<s1_1] = s1_1

		return scaled_signal

	def signal_normalize_sub1(self,serial,scale):

		s_min, s_max = self.signal_limit[0], self.signal_limit[1]
		s1, s2 = scale[0], scale[1]

		id1 = mapping_Idx(self.serial, serial)
		b1 = np.where(id1>=0)[0]
		if len(id1)!=len(serial):
			print('signal_normalize_sub1: error!', len(b1))
			# return b1
		id1 = id1[b1]

		chrom, serial = self.chrom[id1], self.serial[id1]
		chrom_vec = np.unique(t_chrom)
		chrom_num = len(chrom_vec)
		t_vec1 = []
		# chrom_num1 = 22
		# for i in range(0,chrom_num):
		# id2 = []
		for i in chrom_vec:
			# chrom_id = chrom_vec[i]
			chrom_id = 'chr%s'%(i)
			b = np.where(chrom==chrom_id)[0]
			# id2.extend(b)
			t_signal = signal[b]
			s_min, s_max = np.min(t_signal), np.max(t_signal)
			t_vec1.append([i, s_min, s_max])
			print(chrom_id,s_min,s_max)

		# s_min, s_max = np.min(signal), np.max(signal)
		s_min = np.quantile(t_vec1[:,1], 0.1)
		s_max = np.quantile(t_vec1[:,2], 0.9)

		vec1 = self.signal_scale(self.signal_pre[id1],s_min,s_max,s1,s2)
		self.signal_pre1 = self.signal.copy()
		self.signal[id1] = vec1 

		return vec1

	def signal_normalize_chrom(self, chrom, signal, chrom_vec, scale):

		s1, s2 = scale[0], scale[1]
		# chrom_vec = np.unique(chrom)
		chrom_num = len(chrom_vec)
		t_vec1 = np.zeros((chrom_num,3))
		# chrom_num1 = 22
		# for i in range(0,chrom_num):
		id1 = []
		i1 = 0
		for i in chrom_vec:
			# chrom_id = chrom_vec[i]
			chrom_id = 'chr%s'%(i)
			b = np.where(chrom==chrom_id)[0]
			id1.extend(b)
			t_signal = signal[b]
			print(chrom_id,len(b))
			s_min, s_max = np.min(t_signal), np.max(t_signal)
			t_vec1[i1] = [i, s_min, s_max]
			print(chrom_id,s_min,s_max)
			i1 += 1
			# s_min, s_max = np.min(signal), np.max(signal)

		# s_min = np.quantile(t_vec1[:,1], 0.1)
		# s_max = np.quantile(t_vec1[:,2], 0.9)
		s_min = np.quantile(t_vec1[:,1], 0.1)
		s_max = np.quantile(t_vec1[:,2], 0.9)
		print(s_min,s_max)
		self.signal_limit = [s_min,s_max,np.min(t_vec1[:,1]),np.max(t_vec1[:,2]),t_vec1]

		scaled_signal = self.signal_scale(signal[id1],s_min,s_max,s1,s2)

		if self.config['signal_plot']==1:
			for i in chrom_vec:
				### Plot the results
				plt.figure(figsize=(18, 6))
				plt.subplot(131)
				plt.title('Signal')
				chrom_id = 'chr%s'%(i)
				# chrom_id = chrom_vec[i]
				b = np.where(chrom==chrom_id)[0]
				t_signal = signal[b]
				s_min, s_max = np.min(t_signal), np.max(t_signal)

				interval1 = np.linspace(s_min-0.1,s_max+0.1,num=100)
				plt.hist(t_signal, bins = interval1)

				plt.subplot(132)
				plt.title('Signal 2')
				t_signal = scaled_signal[b]
				interval2 = np.linspace(-0.05,1.05,num=100)
				plt.hist(t_signal, bins = interval2)
				
				plt.subplot(133)
				interval3 = np.linspace(-0.05,1.05,num=100)
				plt.title('Log signal')
				plt.hist(np.log2(t_signal+1), bins = interval3) 

				plt.savefig('signal_%s.png' % (chrom_id), dpi=300)

		return scaled_signal, id1, t_vec1

	def signal_normalize_bychrom(self, chrom, signal, chrom_vec, scale):

		s1, s2 = scale[0], scale[1]
		# chrom_vec = np.sort(np.unique(chrom))
		print('chrom_vec',chrom_vec)
		chrom_num = len(chrom_vec)
		t_vec1 = np.zeros((chrom_num,4))
		# chrom_num1 = 22
		signal1 = signal.copy()
		# for i in range(0,chrom_num):
		id1 = []
		t_vec1 = []
		for i in chrom_vec:
			chrom_id = str(i)
			if chrom_id.find('chr')<0:
				chrom_id = 'chr%s'%(chrom_id)
			print(chrom_id)
			print(chrom[0:10])
			b = np.where(chrom==chrom_id)[0]
			id1.extend(b)
			print('signal_normalize_bychrom',b)
			t_signal = signal[b]
			s_min1, s_max1 = np.min(t_signal), np.max(t_signal)
			t_signal1 = np.sort(t_signal)
			t1, t2, t3 = np.quantile(t_signal,[0.25,0.75,0.50])
			interval = t2-t1
			s_1 = t3-1.5*(t2-t1)
			s_2 = t3+1.5*(t2-t1)
			b1 = np.where((t_signal1<s_2)&(t_signal1>s_1))[0]
			s_min, s_max = t_signal1[b1[0]], t_signal1[b1[-1]]
			t_vec1[i,0], t_vec1[i,1], t_vec1[i,2], t_vec1[i,3] = s_min, s_max, s_min1, s_max1
			# s_min, s_max = np.min(signal), np.max(signal)
			# scaled_signal = s1+(t_signal-s_min)*1.0/(s_max-s_min)*(s2-s1)
			scaled_signal = self.signal_scale(t_signal1,s_max,s_min,s1,s2)
			t_vec1.append([i, s_min, s_max])
			
			signal1[b] = scaled_signal
			print(chrom_id,s_min1,s_max1,s_min,s_max)

		t_vec1 = np.asarray(t_vec1)
		s_min = np.quantile(t_vec1[:,1], 0.1)
		s_max = np.quantile(t_vec1[:,2], 0.9)
		print(s_min,s_max)
		self.signal_limit = [s_min,s_max,np.min(t_vec1[:,1]),np.max(t_vec1[:,2]),t_vec1]

		return signal1, id1, t_vec1

	# query features of a specific set of samples
	def load_feature_1(self,signal_ori,serial_ori,query_serial,feature_idx=-1):

		trans_id1a = mapping_Idx(serial_ori,query_serial)	# mapped index
		signal1 = signal_ori[trans_id1a]
		if feature_idx!=-1:
			signal1 = signal_ori[:,feature_idx]

		return signal1

	# query features of a specific set of samples
	def load_feature_2(self,signal_ori,sel_idx,feature_idx=-1):

		signal1 = signal_ori[sel_idx]
		if feature_idx!=-1:
			signal1 = signal1[:,feature_idx]

		return signal1

	# chrom id of regions
	def find_chrom_id(self,region_data,pos_sample):

		chrom, start, stop, serial = np.asarray(region_data['chrom']), np.asarray(region_data['start']), np.asarray(region_data['stop']), np.asarray(region_data['serial'])
		pos1, pos2 = pos_sample[0], pos_sample[1]-1
		chrom_id1 = chrom[pos1]
		chrom_id2 = chrom[pos2]

		b1 = np.where(chrom_id1!=chrom_id2)[0]

		if len(b1)>0:
			print('error!',len(b1))
			return -1

		num1 = len(chrom_id1)
		t_chrom_id1 = np.zeros(num1,dtype=np.int32)
		chrom_vec = np.unique(chrom_id1)
		for t_chrom_id in chrom_vec:
			b1 = np.where(chrom_id1==t_chrom_id)[0]
			t_chrom_id1[b1] = int(t_chrom_id[3:])

		return t_chrom_id1

	###########################################################
	## generate training and test data
	# prepare data: test mode
	def prep_data_test(self,path1,file_prefix,type_id2,feature_dim_transform):

		self.feature_dim_transform = feature_dim_transform
		# map_idx = mapping_Idx(serial_ori,serial)

		sub_sample_ratio = 1
		shuffle = 0
		normalize, flanking, attention, run_id = self.normalize, self.flanking, self.attention, self.run_id
		config = self.config
		# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
		tol = self.tol
		L = flanking
		# path1 = '/mnt/yy3'

		# np.save(filename1)
		print("feature transform")
		filename1 = '%s/%s_%d_%d_%d.npy'%(path1,file_prefix,type_id2,feature_dim_transform[0],feature_dim_transform[1])

		if self.species_id=='mm10':
			filename1 = '%s/%s_%d_100_50.npy'%(path1,file_prefix,type_id2)

		if os.path.exists(filename1)==True:
			print("loading data...")
			data1 = np.load(filename1,allow_pickle=True)
			data_1 = data1[()]
			x_train1_trans, train_sel_list = np.asarray(data_1['x1']), np.asarray(data_1['idx'])
			print('train_sel_list',train_sel_list.shape)
		else:
			print("data not found!")
			return

		feature_dim_motif = self.feature_dim_motif
		self.feature_dim_motif = 1
		x_train1_trans = self.feature_dim_select(x_train1_trans,feature_dim_transform)
		self.feature_dim_motif = feature_dim_motif
		train_id1, test_id1, y_signal_train1, y_signal_test, train1_sel_list, test_sel_list = self.generate_train_test_1(train_sel_list)

		# print('y_signal_train',np.max(y_signal_train),np.min(y_signal_train))
		print('y_signal_test',np.max(y_signal_test),np.min(y_signal_test))
		# y_signal_test_ori = self.signal_normalize(y_signal_test,[0,1])

		# shuffle array
		# x_test_trans, shuffle_id2 = shuffle_array(x_test_trans)
		# test_sel_list = test_sel_list[shuffle_id2]

		print(train1_sel_list[0:5])

		# # split training and validation data
		# type_id_1 = 0
		# idx_train, idx_valid, idx_test = self.generate_index_1(train1_sel_list, test_sel_list, ratio, type_id_1)
		# train_sel_list, val_sel_list = train1_sel_list[idx_train], train1_sel_list[idx_valid]
		
		# self.x, self.y = dict(), dict()	# feature matrix and signals
		# self.vec, self.vec_local = dict(), dict()	# serial
		self.idx_list_test = {'test1':test_id1,'test2':test_sel_list}

		# seq_list_train, seq_list_valid: both locally calculated
		keys = ['test1']
		for i in keys:
			self.seq_list[i] = generate_sequences(self.idx_sel_list[i],region_list=self.region_boundary)
			print(len(self.seq_list[i]))

		# generate initial state index
		self.init_index(keys)
		self.y['test1'] = y_signal_test

		# training and validation data
		for i in keys:
			idx = self.idx_list_test[i]
			if self.method<5:
				self.x[i] = x_train1_trans[idx]
			else:
				x, y, vec, vec_local = sample_select2a1(x_train1_trans[idx],self.y[i],
														test_sel_list, self.seq_list[i], self.tol, self.flanking)
				# concate context for baseline methods
				self.vec[i], self.vec_local[i] = vec, vec_local
				if self.method<=10:
					# x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.2, random_state=42)
					x = x.reshape(x.shape[0],x.shape[1]*x.shape[-1])
					y = y[:,self.flanking]

				self.x[i], self.y[i] = x, y
				print(self.x[i].shape, self.y[i].shape)

		return x,y,test_sel_list,vec,vec_local

	# generate training idx and validation idx
	def generate_index_1(self, train_sel_list, test_sel_list, ratio=0.9, type_id=0):

		idx_test = []
		if type_id==0:
			train_num = len(train_sel_list)
			id1 = range(0,train_num)
			num1 = int(train_num*ratio)
			# num2 = int(train_num*0.9)
			idx_train = id1[0:num1]
			# idx_valid = id1[num1:num2]
			idx_valid = id1[num1:]
			# idx_test = id1[num2:]
		else:
			train_sel_list = np.asarray(train_sel_list)
			chrom = train_sel_list[:,0]
			chrom_vec = np.unique(chrom)
			print(chrom_vec)
			idx_train, idx_valid, idx_test = [], [], []
			for chrom_id in chrom_vec:
				id1 = np.where(chrom==chrom_id)[0]
				train_num = len(id1)
				num1 = int(train_num*ratio)
				# num2 = int(train_num*0.9)
				idx_train.extend(id1[0:num1])
				# idx_valid.extend(id1[num1:num2])
				idx_valid.extend(id1[num1:])
				print('generate_index_1',chrom_id,train_num,num1,train_num-num1)
				print(id1[0:5],id1[num1:num1+5])
				# idx_test.extend(id1[num2:])

		return idx_train, idx_valid, idx_test

	# generate training idx and validation idx
	def generate_index_2(self, train_sel_list, n_fold=5, ratio=0.9, type_id=1):

		idx_test = []
		if type_id==1:
			train_sel_list = np.asarray(train_sel_list)
			serial_ori = train_sel_list[:,1]
			chrom = train_sel_list[:,0]
			chrom_vec = np.unique(chrom)
			print(chrom_vec)
			
			idx_train_pre = dict()
			ratio1 = 1.0/n_fold
			for chrom_id in chrom_vec:
				id1 = np.where(chrom==chrom_id)[0]
				train_num = len(id1)
				interval1 = int(train_num*ratio1)
				list1 = interval1*np.arange(n_fold)
				list1 = np.vstack((list1,list1+interval1)).T
				list1[-1][1] = train_num
				list2 = []
				idx_train_pre[chrom_id] = []
				for i in range(n_fold):
					list2.append(id1[list1[i,0]:list1[i,1]])
					print(chrom_id,i,len(list1))

				idx_train_pre[chrom_id] = list2

			train_serial1, valid_serial1, test_serial1 = dict(), dict(), dict()
			dict_1 = dict()
			for chrom_id in chrom_vec:
				dict_1[chrom_id] = np.random.permutation(n_fold)

			dict1 = dict()
			for i in range(n_fold):
				dict1[i] = {'train':[],'valid':[],'test':[]}
				for chrom_id in chrom_vec:
					list2 = idx_train_pre[chrom_id]
					t_idvec = dict_1[chrom_id]
					test_sel_id = t_idvec[i]
					test_id1 = list2[test_sel_id]
					
					train_sel_id = np.setdiff1d(range(n_fold),test_sel_id)
					print(i,chrom_id,dict_1[chrom_id],len(test_id1),test_id1[0],test_id1[-1])
					train_id1_pre = []
					for l1 in train_sel_id:
						train_id1_pre.extend(list2[l1])

					train_id1_pre = np.asarray(train_id1_pre)
					num1 = len(train_id1_pre)
					valid_num = int(num1*(1-ratio))

					if self.config['valid_random']==1:
						start_idx = np.random.randint(num1-valid_num-1)
						valid_id1 = train_id1_pre[start_idx:start_idx+valid_num]
						train_id1 = np.setdiff1d(train_id1_pre,valid_id1)
						print('valid_id',valid_id1[0],valid_id1[-1])
					else:
						valid_id1 = train_id1_pre[(num1-valid_num):]
						train_id1 = train_id1_pre[0:-valid_num]

					dict1[i]['train'].extend(train_id1)
					dict1[i]['valid'].extend(valid_id1)
					dict1[i]['test'].extend(test_id1)

				print(i,len(dict1[i]['train']),len(dict1[i]['valid']),len(dict1[i]['test']))

				print('train,valid,test',len(dict1[i]['train']),len(dict1[i]['valid']),len(dict1[i]['test']))

		return dict1

	# generate training and test data
	# input: x_train1_trans: data
	#		 train_sel_list: list of chromosome and serial
	def generate_train_test_1(self,train_sel_list):

		# training data and test data
		print('signal',np.max(self.signal),np.min(self.signal))
		train_sel_list, test_sel_list, train_id1, test_id1, id_1, id_2 = self.training_serial(train_sel_list)

		# y_signal_train = self.signal[id_1]
		y_signal_test = self.signal[id_2]

		if ('train_signal_update' in self.config) and self.config['train_signal_update']==1:
			train_serial = train_sel_list[:,1]
			y_signal_train = self.signal_normalize_sub1(train_serial,self.scale)
		else:
			y_signal_train = self.signal[id_1]

		return train_id1, test_id1, y_signal_train, y_signal_test, train_sel_list, test_sel_list

	def training_serial_pre(self,train_sel_list,chrom_vec):

		train_id1 = []
		train_id2 = []
		# print(self.train_chromvec,self.test_chromvec)
		# print(train_sel_list[:,0])
		print(chrom_vec)
		for t_chrom in chrom_vec:
			b1 = np.where(train_sel_list[:,0]==int(t_chrom))[0]
			# train_id1.extend(train_sel_list[b1,1])
			train_id1.extend(b1)
			train_id2.append(b1)

		return np.asarray(train_id1), train_id2

	# generate serial for training and test data
	def training_serial(self,train_sel_list):
		# # training data and test data
		test_id1, test_id2 = self.training_serial_pre(train_sel_list,self.test_chromvec)
		region_list = self.region_list_train
		if len(region_list)==0:
			# train_id1, test_id1, test_id2 = self.training_serial_pre(train_sel_list)
			train_id1, train_id2 = self.training_serial_pre(train_sel_list,self.train_chromvec)
		else:
			train_id1, test_id1_pre = self.generate_train_test_idx_2(train_sel_list,region_list)
			# test_id1 = list(set(test_id1_pre)|set(test_id1))	# include the reserved test regions
			# test_id1 = list(set(test_id1)-set(train_id1))	# not including training regions
			# test_id1 = np.asarray(test_id1)
			test_id1 = np.union1d(test_id1_pre,test_id1)	# include the reserved test regions
			test_id1 = np.setdiff1d(test_id1,train_id1)	# not including training regions

		print('train_id1 test_id1',len(train_id1),len(test_id1))

		# train_serial = np.intersect1d(train_id1,self.serial)
		# test_serial = np.intersect1d(test_id1,self.serial)
		# print('train_id1',train_id1)
		# print(len(self.serial))

		id1 = mapping_Idx(train_sel_list[train_id1,1],self.serial)
		id_1 = np.where(id1>=0)[0]
		print(len(train_id1),len(id_1))
		train_id1 = train_id1[id1[id_1]]

		# print('test_id1',test_id1)
		id2 = mapping_Idx(train_sel_list[test_id1,1],self.serial)
		id_2 = np.where(id2>=0)[0]
		print(len(test_id1),len(id_2))
		test_id1 = test_id1[id2[id_2]]

		# print(x_train1_trans.shape,train_sel_list.shape)
		# print(train_sel_list.shape)
		train_sel_list_ori = train_sel_list.copy()
		test_sel_list = train_sel_list_ori[test_id1]
		train_sel_list = train_sel_list[train_id1]

		# id_1, id_2: relative index mapping
		return train_sel_list, test_sel_list, train_id1, test_id1, id_1, id_2

	# generate initial index
	def init_index(self,keys):

		# self.init_id = dict()
		for i in keys:
			self.init_id[i] = self.seq_list[i][:,0]

		return True

	# search the regions
	def region_search(self, chrom, start, stop, region_list, type_id2, tol=1):

		num1 = len(region_list)
		id1 = []
		c1 = tol
		bin_size = 5000
		for i in range(0,num1):
			t_region = region_list[i]
			t_chrom, t_start, t_stop = t_region[0], t_region[1], t_region[2]
			t_chrom1 = 'chr%s'%(t_chrom)

			if (type_id2==1):
				t_stop = t_stop+bin_size
				t_start = t_start-bin_size

			b = np.where((chrom==t_chrom1)&(start<t_stop)&(stop>t_start))[0]
			if len(b)>0:
				# if type_id2==1 or len(b)<=1:
				# 	id_1 = np.max([0,b[0]-c1])
				# 	id_2 = np.min([len(chrom)-1,b[-1]+c1])
				# 	b = list(range(id_1,b[0]))+list(b)+list(range(b[-1],id_2+1))
				# 	# print(b)
				if i%5000==0:
					print(t_region,len(b),start[b],stop[b])
				id1.extend(b)

		return np.unique(id1)

	# search the boundary of regions
	def region_search_boundary(self, chrom, start, stop, serial, region_list, type_id2=0, tol=0):

		num1 = len(region_list)
		list1 = []
		for i in range(0,num1):
			t_region = region_list[i]
			t_chrom, t_start, t_stop = t_region[0], t_region[1], t_region[2]
			t_chrom1 = 'chr%s'%(t_chrom)
			b = np.where(chrom==t_chrom1)[0]
			b1 = np.where(stop[b]<=t_start)[0]
			b2 = np.where(start[b]>=t_stop)[0]
			if len(b1)>0 and len(b2)>0:
				id1, id2 = b[b1[-1]], b[b2[0]]
				boundary1, boundary2 = serial[id1], serial[id2]
				# boundary1, boundary2 = id1, id2
			
				# print(t_region,id1,id2)
				list1.append([t_chrom,boundary1,boundary2])

		return np.asarray(list1)

	# search the regions
	def region_search_1(self, chrom, start, stop, serial, region_list, type_id2=0, tol=0):
		
		if len(region_list)>0:
			t_region_list = []
			for t_region in region_list:
				t_chrom, t_start, t_stop = t_region[0], t_region[1], t_region[2]
				t_chrom1 = 'chr%s'%(t_chrom)
				t_region_len = (t_stop-t_start)/self.bin_size
				b = np.where(chrom==t_chrom1)[0]
				if (type_id2==1):
					t_start1 = np.max((0,t_start-tol*self.bin_size))
					t_stop1 = np.min((stop[b][-1],t_stop+tol*self.bin_size))
					t_region_list.append([t_chrom,t_start1,t_stop1])

			region_list = t_region_list
			# print(region_list)

			b1 = self.region_search(chrom,start,stop,region_list,type_id2=0,tol=0)
			print('region search',len(b1))
			# print(b1)
			id1 = np.setdiff1d(range(len(start)),b1)

		return id1, region_list

	# select regions and exclude the regions from the original list of serials
	# return: serial_list1: selected regions excluded
	#		  serial_list2: selected regions
	def select_region(self, serial1, regionlist_filename):

		region_list = pd.read_csv(regionlist_filename,header=None,sep='\t')
		colnames = list(region_list)
		col1, col2, col3 = colnames[0], colnames[1], colnames[2]
		chrom1, start1, stop1 = region_list[col1], region_list[col2], region_list[col3]
		num1 = len(chrom1)
		# serial_list1 = self.serial.copy()
		serial_list2 = []
		for i in range(0,num1):
			b1 = np.where((self.chrom==chrom1[i])&(self.start>=start1[i])&(self.stop<=stop1[i]))[0]
			serial_list2.extend(self.serial[b1])
		
		print(len(serial1),len(serial_list2))
		serial_list1 = np.setdiff1d(serial1,serial_list2)

		return serial_list1, serial_list2

	# find serial of a genomic locus
	# input: chromosome name, poistion, chromosome vector to find the serial of the position
	# output: serial of the genomic locus
	def query_serial(self,chrom,position,thresh1=10000):

		# thresh1 = 2*self.bin_size
		# thresh1 = 10000
		chrom1 = 'chr%s'%(chrom)
		b1 = np.where(self.chrom==chrom1)[0]
		b2 = np.where((self.start[b1]<=position)&(self.stop[b1]>position))[0]
		if len(b2)==0:
			distance = self.stop[b1]-position
			if np.min(distance)<thresh1:
				b2 = np.argmin(distance)
		t_serial = self.serial[b1[b2]]

		return t_serial

	# find serial of a genomic locus
	# input: chromosome name, poistion, chromosome vector to find the serial of the position
	# output: serial of the genomic locus
	def query_serial_1(self,chrom,position,thresh1=10000):

		# thresh1 = 10000
		chrom1 = 'chr%s'%(chrom)
		b1 = np.where(self.chrom_ori==chrom1)[0]
		b2 = np.where((self.start_ori[b1]<=position)&(self.stop_ori[b1]>position))[0]
		if len(b2)==0:
			distance = self.stop_ori[b1]-position
			if np.min(distance)<thresh1:
				b2 = np.argmin(distance)
		t_serial = self.serial_ori[b1[b2]]

		return t_serial

	# function1
	# search by chromosome
	def search_chrom(self,train_sel_list,train_chromvec):
		# training data and test data
		train_id1 = []
		for t_chrom in train_chromvec:
			b1 = np.where(train_sel_list[:,0]==int(t_chrom))[0]
			train_id1.extend(b1)

		return train_id1

	# generate training and test indices
	def generate_train_test_idx_1(self,train_sel_list):

		train_id1 = []
		test_id1 = []
		print(self.train_chromvec,self.test_chromvec)
		for t_chrom in self.train_chromvec:
			b1 = np.where(train_sel_list[:,0]==int(t_chrom))[0]
			num1 = len(b1)
			print(t_chrom,num1)
			vec1 = list(range(0,int(num1*0.4)))+list(range(int(num1*0.5),num1))
			vec2 = list(range(int(num1*0.4),int(num1*0.5)))
			train_id1.extend(train_sel_list[b1[vec1],1])
			test_id1.extend(train_sel_list[b1[vec2],1])

		return np.asarray(train_id1), np.asarray(test_id1)

	# generate training and test indices
	def generate_train_test_idx_2(self,train_sel_list,region_list):

		train_id1 = []
		test_id1 = []
		print(self.train_chromvec,self.test_chromvec)
		region_list = np.asarray(region_list)
		print(region_list[0:5])
		# return
		for t_chrom in self.train_chromvec:
			b1 = np.where(train_sel_list[:,0]==int(t_chrom))[0]
			b2 = np.where(region_list[:,0]==int(t_chrom))[0]
			if len(b2)==0:
				train_id1.extend(b1)	# index
				continue
			print(region_list)
			start1, stop1 = np.min(region_list[b2,1]), np.max(region_list[b2,2])
			# serial of the region start and stop positions
			serial1_1, serial2_1 = self.query_serial(t_chrom,start1), self.query_serial(t_chrom,stop1)
			serial1, serial2 = self.query_serial_1(t_chrom,start1), self.query_serial_1(t_chrom,stop1)
			print(serial1_1,serial2_1,serial1,serial2)
			num1 = len(b1)
			print(t_chrom,num1,start1,stop1,serial1,serial2)
			vec2 = np.where((train_sel_list[b1,1]>=serial1)&(train_sel_list[b1,1]<=serial2))[0]
			vec2 = b1[vec2]
			vec1 = np.setdiff1d(b1,vec2)
			# vec1 = list(range(0,int(num1*0.4)))+list(range(int(num1*0.5),num1))
			# vec2 = list(range(int(num1*0.4),int(num1*0.5)))
			# train_id1.extend(train_sel_list[vec1,1])	# serial
			# test_id1.extend(train_sel_list[vec2,1])	# serial
			train_id1.extend(vec1)	# index
			test_id1.extend(vec2)	# index

		train_id1, test_id1 = np.int64(np.asarray(train_id1)), np.int64(np.asarray(test_id1))

		return train_id1, test_id1

	# input: regions reserved for training and validation
	def generate_train_test_2(self,train_sel_list,idx_train,idx_valid):

		# training regions on the chromosome
		region_list = self.region_list_train
		# region_list = region_list[np.newaxis,:]
		print(len(train_sel_list))
		if len(region_list)>0:
			# train_id2: included training region
			train_id1, train_id2 = self.generate_train_test_idx_2(train_sel_list,region_list)
			# temp1 = np.setdiff1d(idx_valid,train_id1)
			idx_train = np.union1d(idx_train,train_id2)
			temp1 = list(range(0,len(train_sel_list)))
			# idx_valid = np.intersect1d(idx_valid,train_id1)
			idx_valid = np.setdiff1d(temp1,idx_train)
		print('idx_train,idx_valid',len(idx_train),len(idx_valid))

		# validation regions on the chromosome
		region_list = self.region_list_valid
		# region_list = region_list[np.newaxis,:]
		if len(region_list)>0:
			# train_id2: included training region
			train_id1, valid_id2 = self.generate_train_test_idx_2(train_sel_list,region_list)
			# temp1 = np.setdiff1d(idx_valid,train_id1)
			idx_valid = np.union1d(idx_valid,valid_id2)
			# idx_valid = np.intersect1d(idx_valid,train_id1)
			idx_train = np.setdiff1d(temp1,idx_valid)
		print('idx_train,idx_valid',len(idx_train),len(idx_valid))

		return idx_train, idx_valid

	def find_region(self,region_list):
		
		region_list = np.asarray([16,43500000,51000000])
		region_list = region_list[np.newaxis,:]
		print(len(train_sel_list),len(region_list))
		print(self.train_chromvec)
		train_id1, test_id1 = self.generate_train_test_idx_2(train_sel_list,region_list)

		print('train_id1 test_id1 ratio',len(train_id1),len(test_id1), len(train_id1)*1.0/len(test_id1))

		# train_serial = np.intersect1d(train_id1,self.serial)
		# test_serial = np.intersect1d(test_id1,self.serial)
		id1 = mapping_Idx(train_sel_list[train_id1,1],self.serial)
		id_1 = np.where(id1>=0)[0]
		print(len(train_id1),len(id_1))
		train_id1 = train_id1[id1[id_1]]

		id2 = mapping_Idx(train_sel_list[test_id1,1],self.serial)
		id_2 = np.where(id2>=0)[0]
		print(len(test_id1),len(id_2))
		test_id1 = test_id1[id2[id_2]]

		print(x_train1_trans.shape,train_sel_list.shape)
		x_train1_trans_ori, train_sel_list_ori = x_train1_trans.copy(), train_sel_list.copy()
		x_test1_trans, test_sel_list = x_train1_trans_ori[test_id1], train_sel_list_ori[test_id1]
		x_train1_trans, train_sel_list = x_train1_trans[train_id1], train_sel_list[train_id1]

		return True

	###########################################################
	## data transformation and feature selection
	# feature dimension reduction
	def feature_transform(self, x_train, x_test, feature_dim_transform, shuffle, sub_sample_ratio, type_id, normalize=0):
		
		x_ori1 = np.vstack((x_train,x_test))
		dim1 = x_ori1.shape[1]
		feature_dim_kmer = np.sum(self.feature_dim_kmer)
		dim2 = dim1-feature_dim_kmer-self.feature_dim_motif
		feature_dim = feature_dim_transform[0]
		feature_dim1 = feature_dim_transform[1]
		
		print("feature_dim_kmer",feature_dim_kmer,dim2)
		if self.kmer_size[1]==-1:
			feature_kmer_idx = np.asarray(range(dim2,dim2+self.feature_dim_kmer[0]))
		else:
			feature_kmer_idx = np.asarray(range(dim2,dim2+feature_dim_kmer))
		x_ori = x_ori1[:,feature_kmer_idx]
		if self.feature_dim_motif>0:
			feature_motif_idx = np.asarray(range(dim2+feature_dim_kmer,dim1))
			x_ori_motif = x_ori1[:,feature_motif_idx]
		if normalize>=1:
			sc = StandardScaler()
			x_ori = sc.fit_transform(x_ori)	# normalize data
			x_ori_motif = sc.fit_transform(x_ori_motif)

		num_train, num_test = x_train.shape[0], x_test.shape[0]
		vec1 = ['PCA','Incremental PCA','Kernel PCA','Sparse PCA','SVD','GRP','SRP','MDS','ISOMAP','Minibatch','ICA','tSNE','LLE','Encoder']
		
		if sub_sample_ratio<1:
			sub_sample = int(x_ori.shape[0]*sub_sample_ratio)
		else:
			sub_sample = -1

		shuffle = 0
		start = time.time()
		x = self.dimension_reduction(x_ori,feature_dim,shuffle,sub_sample,type_id)
		stop = time.time()
		print("feature transform %s"%(vec1[type_id]),stop - start)

		# save transfrom model
		filename1 = '%s_%d_dimensions1.h5'%(self.filename_load,feature_dim)
		# pickle.dump(self.dimension_model, open(filename1, 'wb'))

		if self.feature_dim_motif>0:
			start = time.time()
			motif_type_id = 4 # Truncated SVD
			x_motif = self.dimension_reduction(x_ori_motif,feature_dim1,shuffle,sub_sample,motif_type_id)
			stop = time.time()
			print("feature transform motif %s"%(vec1[motif_type_id]),stop - start)
			x1 = np.hstack((x_ori1[:,0:dim2],x,x_motif))

			# save transfrom model
			filename1 = '%s_%d_dimensions2.2.h5'%(self.filename_load,feature_dim1)
			pickle.dump(self.dimension_model, open(filename1, 'wb'))
		else:
			x1 = np.hstack((x_ori1[:,0:dim2],x))

		if normalize>=2:
			sc = StandardScaler()
			x1 = sc.fit_transform(x1)
		x_train1, x_test1 = x1[0:num_train], x1[num_train:num_train+num_test]
		print(x_train.shape,x_train1.shape,x_test.shape,x_test1.shape)

		return x_train1, x_test1

	# dimension reduction methods
	def dimension_reduction(self,x_ori,feature_dim,shuffle,sub_sample,type_id,filename_prefix=''):

		# if shuffle==1 and sub_sample>0:
		# 	idx = np.random.permutation(x_ori.shape[0])
		# else:
		# 	idx = np.asarray(range(0,x_ori.shape[0]))
		idx = np.asarray(range(0,x_ori.shape[0]))
		if (sub_sample>0) and (type_id!=7) and (type_id!=11):
			id1 = idx[0:sub_sample]
		else:
			id1 = idx

		if type_id==0:
			# PCA
			pca = PCA(n_components=feature_dim, whiten = False, random_state = 0)
			if sub_sample>0:
				pca.fit(x_ori[id1,:])
				x = pca.transform(x_ori)
			else:
				x = pca.fit_transform(x_ori)
			self.dimension_model = pca
		# X_pca_reconst = pca.inverse_transform(x)
		elif type_id==1:
		# Incremental PCA
			n_batches = 10
			inc_pca = IncrementalPCA(n_components=feature_dim)
			for X_batch in np.array_split(x_ori, n_batches):
				inc_pca.partial_fit(X_batch)
			x = inc_pca.transform(x_ori)
			self.dimension_model = inc_pca
		# X_ipca_reconst = inc_pca.inverse_transform(x)
		elif type_id==2:
			# Kernel PCA
			kpca = KernelPCA(kernel="rbf",n_components=feature_dim, gamma=None, fit_inverse_transform=True, random_state = 0, n_jobs=50)
			kpca.fit(x_ori[id1,:])
			x = kpca.transform(x_ori)
			self.dimension_model = kpca
			# X_kpca_reconst = kpca.inverse_transform(x)
		elif type_id==3:
			# Sparse PCA
			sparsepca = SparsePCA(n_components=feature_dim, alpha=0.0001, random_state=0, n_jobs=50)
			sparsepca.fit(x_ori[id1,:])
			x = sparsepca.transform(x_ori)
			self.dimension_model = sparsepca
		elif type_id==4:
			# SVD
			SVD_ = TruncatedSVD(n_components=feature_dim,algorithm='randomized', random_state=0, n_iter=5)
			SVD_.fit(x_ori[id1,:])
			x = SVD_.transform(x_ori)
			self.dimension_model = SVD_
			# X_svd_reconst = SVD_.inverse_transform(x)
		elif type_id==5:
			# Gaussian Random Projection
			GRP = GaussianRandomProjection(n_components=feature_dim,eps = 0.5, random_state=2019)
			GRP.fit(x_ori[id1,:])
			x = GRP.transform(x_ori)
			self.dimension_model = GRP
		elif type_id==6:
			# Sparse random projection
			SRP = SparseRandomProjection(n_components=feature_dim,density = 'auto', eps = 0.5, random_state=2019, dense_output = False)
			SRP.fit(x_ori[id1,:])
			x = SRP.transform(x_ori)
			self.dimension_model = SRP
		elif type_id==7:
			# MDS
			mds = MDS(n_components=feature_dim, n_init=12, max_iter=1200, metric=True, n_jobs=4, random_state=2019)
			x = mds.fit_transform(x_ori[id1])
			self.dimension_model = mds
		elif type_id==8:
			# ISOMAP
			isomap = Isomap(n_components=feature_dim, n_jobs = 4, n_neighbors = 5)
			isomap.fit(x_ori[id1,:])
			x = isomap.transform(x_ori)
			self.dimension_model = isomap
		elif type_id==9:
			# MiniBatch dictionary learning
			miniBatchDictLearning = MiniBatchDictionaryLearning(n_components=feature_dim,batch_size = 1000,alpha = 1,n_iter = 25,  random_state=2019)
			if sub_sample>0:
				miniBatchDictLearning.fit(x_ori[id1,:])
				x = miniBatchDictLearning.transform(x_ori)
			else:
				x = miniBatchDictLearning.fit_transform(x_ori)
			self.dimension_model = miniBatchDictLearning
		elif type_id==10:
			# ICA
			fast_ICA = FastICA(n_components=feature_dim, algorithm = 'parallel',whiten = True,max_iter = 100,  random_state=2019)
			if sub_sample>0:
				fast_ICA.fit(x_ori[id1])
				x = fast_ICA.transform(x_ori)
			else:
				x = fast_ICA.fit_transform(x_ori)
			self.dimension_model = fast_ICA
			# X_fica_reconst = FastICA.inverse_transform(x)
		# elif type_id==11:
		# 	# t-SNE
		# 	tsne = TSNE(n_components=feature_dim,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
		# 	x = tsne.fit_transform(x_ori)
		elif type_id==12:
			# Locally linear embedding
			lle = LocallyLinearEmbedding(n_components=feature_dim, n_neighbors = np.max((int(feature_dim*1.5),500)),method = 'modified', n_jobs = 20,  random_state=2019)
			lle.fit(x_ori[id1,:])
			x = lle.transform(x_ori)
			self.dimension_model = lle
		elif type_id==13:
			# Autoencoder
			feature_dim_ori = x_ori.shape[1]
			m = Sequential()
			m.add(Dense(512,  activation='elu', input_shape=(feature_dim_ori,)))
			# m.add(Dense(256,  activation='elu'))
			m.add(Dense(feature_dim,   activation='linear', name="bottleneck"))
			# m.add(Dense(256,  activation='elu'))
			m.add(Dense(512,  activation='elu'))
			m.add(Dense(feature_dim_ori,  activation='sigmoid'))
			m.compile(loss='mean_squared_error', optimizer = Adam())
			history = m.fit(x_ori[id1], x_ori[id1], batch_size=256, epochs=20, verbose=1)

			encoder = Model(m.input, m.get_layer('bottleneck').output)
			x = encoder.predict(x_ori)
			Renc = m.predict(x_ori)
			self.dimension_model = encoder

		# save transfrom model
		if filename_prefix=='':
			filename1 = '%s_%d_dimensions.h5'%(self.filename_load,feature_dim)
		else:
			filename1 = '%s_%d_%d_dimensions.h5'%(filename_prefix,type_id,feature_dim)
		
		# np.save(filename1, self.dimension_model)
		pickle.dump(self.dimension_model, open(filename1, 'wb'))
		# self.dimension_model = pickle.load(open(filename1, 'rb'))

		return x

	# select the feature dimensions to use
	# input: x_train1_trans: original data
	# feature_dim_transform: dimensionality of transformed feature
	def feature_dim_select(self,x_train1_trans,feature_dim_transform):
		
		if self.species_id=='hg38':
			# if self.feature_dim_motif==0:
			# 	x_train1_trans = x_train1_trans[:,0:-feature_dim_transform[1]]
			
			# x_test1_trans = x_test1_trans[:,0:-feature_dim_transform[1]]
			dim1 = x_train1_trans.shape[1]
			t1 = list(range(17,21))
			t2 = list(range(21,dim1))
			vec1 = [list(range(0,2)),list(range(0,21)),list(range(2,21)),
					t1,t2,[0,1]+t2,[0,1]+t1,[0,1]+t1+t2]

			if self.feature_dim_select1>=0:
				sel_idx = vec1[self.feature_dim_select1]
				x_train1_trans = x_train1_trans[:,sel_idx]
		else:
			pass

		return x_train1_trans

	def positional_encoding(self,num_positions,depth1,min_rate=1.0/10000):

		# num_positions = 50
		# depth = 512
		# min_rate = 1/10000

		# assert depth%2 == 0, "Depth must be even."
		depth = depth1
		if depth%2!=0:
			depth = depth1+1
		
		angle_rate_exponents = np.linspace(0,1,depth//2)
		angle_rates = min_rate**(angle_rate_exponents)

		positions = np.arange(num_positions) 
		angle_rads = (positions[:, np.newaxis])*angle_rates[np.newaxis, :]
		# print(angle_rads.shape)

		sines = np.sin(angle_rads)
		cosines = np.cos(angle_rads)
		# pos_encoding = np.concatenate([sines, cosines], axis=-1)
		pos_encoding = np.zeros((num_positions,depth))
		pos_encoding[:,range(0,depth,2)] = sines
		pos_encoding[:,range(1,depth,2)] = cosines

		return pos_encoding

	def positional_encoding1(self,feature_vec,sel_idx_list,feature_dim):
		
		num1 = len(sel_idx_list)
		chrom1 = sel_idx_list[:,0]
		chrom_vec1 = np.unique(chrom1)
		chrom_vec1 = np.sort(chrom_vec1)
		min_rate = 1.0/10000
		print(sel_idx_list.shape)

		for chrom_id in chrom_vec1:
			b1 = np.where(chrom1==chrom_id)[0]
			serial1 = sel_idx_list[b1,1]
			print(serial1.shape)
			# s1, s2 = np.min(serial1), np.max(serial1)
			s1, s2 = serial1[0], serial1[-1]
			t_serial = serial1-s1
			num_positions = s2-s1+1
			# print('num_positions',num_positions)
			pos_encoding = self.positional_encoding(num_positions,feature_dim,min_rate)
			id1 = mapping_Idx(range(0,num_positions),t_serial)
			# self.x_train1_trans[b1] = self.x_train1_trans[id1] + pos_encoding[id1]
			if feature_dim%2==0:
				feature_vec[b1] = feature_vec[b1] + pos_encoding[id1]
			else:
				id2 = list(range(0,21))+list(range(22,feature_dim+1))
				feature_vec[b1] = feature_vec[b1] + pos_encoding[id1][:,id2]
			print(chrom_id,len(b1),pos_encoding.shape,np.max(pos_encoding),np.mean(pos_encoding))

		print(self.feature_dim)
		
		return feature_vec

	###########################################################
	## writing function
	def output_vec(self,vec2,tlist):

		temp1 = []
		for type_id2 in tlist:
			dict1 = vec2[type_id2]['valid']
			vec1 = dict1['score']
			# print(vec1)
			temp1.append([-1]+list(vec1))
			dict2 = vec2[type_id2]['test1']
			vec2 = dict2['score']
			keys_1 = list(vec2.keys())
			# print(keys_1)
			test_chromvec = list(set(keys_1)-set(['aver1']))
			test_chromvec = [int(i) for i in test_chromvec]
			test_chromvec = np.sort(test_chromvec)
			# print(test_chromvec)
			for test_chrom in test_chromvec:
				if test_chrom in keys_1:
					t_score = vec2[test_chrom]
				else:
					t_score = vec2[str(test_chrom)]
				# t_score = vec2[test_chrom]
				print(t_score)
				temp1.append([test_chrom]+list(t_score))

			temp1.append([0]+list(vec2['aver1']))

		temp1 = np.asarray(temp1)
		print(temp1)
		
		if 'pred_filename1' in self.config:
			filename1 = self.config['pred_filename1']
		else:
			filename1 = 'test_vec2_%d_%d_[%d]_%s.txt'%(self.run_id,self.method,self.feature_dim_select1,self.cell)
		np.savetxt(filename1,temp1,fmt='%.7f',delimiter='\t')

		return True

	# write predicted signals
	def write_predicted_signal(self,serial1,y_true,y_predicted,output_filename,x_min=0,x_max=1):

		id1 = mapping_Idx(self.serial,serial1)

		fields = ['chrom','start','stop','serial','signal','predicted']

		data1 = pd.DataFrame(columns=fields)
		data1['chrom'],data1['start'], data1['stop'], data1['signal'], data1['predicted'] = self.chrom[id1], self.start[id1], self.stop[id1], y_true, y_predicted
		data1['serial'] = serial1
		# test2 = pd.DataFrame(columns=fields)
		# test2['chrom'],test2['start'], test2['stop'], test2['signal'] = self.chrom[id2], self.start[id2], self.stop[id2], y_predicted_test

		data_1 = data1.sort_values(by=['serial'])
		data_1.to_csv('valid_%s.txt'%(output_filename),header=True,index=False,sep='\t')

		return True

	# write predicted attention
	def test_result_3_sub1(self,id1,columns,value,output_filename,sort_flag=True,sort_column='serial'):

		chrom_vec, start_vec, stop_vec = self.chrom[id1], self.start[id1], self.stop[id1]

		# fields = ['chrom','start','stop','serial','signal',
		# 			'predicted_signal','predicted_attention']
		fields = ['chrom','start','stop']+columns
		data2 = pd.DataFrame(columns=fields)
		data2['chrom'] = chrom_vec
		data2['start'] = start_vec
		data2['stop'] = stop_vec
		num1 = len(columns)
		for i in range(num1):
			# print('test_result_3_sub1',len(chrom_vec),value[i].shape)
			data2[columns[i]] = value[i]
		
		if sort_flag==True:
			data2 = data2.sort_values(by=[sort_column])

		data2.to_csv(output_filename,index=False,sep='\t',float_format='%.6f')

		return data2

	# write predicted attention
	def test_result_3(self,filename1,output_filename,data1=None,type_id2=0):

		if data1==None:
			print('loading data...',filename1)
			data1 = np.load(filename1,allow_pickle=True)
			data1 = data1[()]
		data_1 = data1[type_id2]

		test_sel_list = data1['test']
		y_predicted_test = data_1['test1']['pred']
		pred_vec1 = data_1['test1']['attention']
		y_test = self.y_signal['test']
		value_vec = []
		if pred_vec1==[]:
			serial_vec = test_sel_list[:,1]
		else:
			vec1 = pred_vec1.keys()
			chrom_vec, start_vec, stop_vec, serial_vec = [], [], [], []
			for chrom_id in vec1:
				temp1 = pred_vec1[chrom_id]
				value = temp1['value']
				t_serial = temp1['serial']
				num1 = len(t_serial)
				id1 = mapping_Idx(self.serial,t_serial)
				b1 = np.where(id1>=0)[0]
				print(chrom_id,num1)
				if len(b1)!=num1:
					print('error!', chrom_id, len(b1))
					return 

				serial_vec.extend(self.serial[id1])
				value_vec.extend(value)

			id1 = mapping_Idx(test_sel_list[:,1],serial_vec)
			b = np.where(id1<0)[0]
			if len(b)>0:
				print('error! test_result_3',len(b))
				return

				y_test = y_test[id1]

		id1 = mapping_Idx(self.serial,serial_vec)
		b1 = np.where(id1<0)[0]
		if len(b1)>0:
			print('error! test_result_3', len(b1))
			return
		
		if len(value_vec)>0:
			value_vec = np.asarray(value_vec)
			if value_vec.ndim>1:
				print('value_vec',value_vec.shape)
				value_vec = value_vec[:,0]
			print('value_vec',value_vec.shape)
		else:
			value_vec = -np.ones(len(y_predicted_test))

		columns = ['serial','signal','predicted_signal','predicted_attention']
		value = [serial_vec,y_test,y_predicted_test,value_vec]
		data2 = self.test_result_3_sub1(id1,columns,value,output_filename,sort_flag=True,sort_column='serial')

		return data2

	# write predicted attention
	def test_result_3_1(self,filename1,output_filename,data1=None,type_id2=0):
		
		if data1==None:
			print('loading data...',filename1)
			data1 = np.load(filename1,allow_pickle=True)
			data1 = data1[()]
		data_1 = data1[type_id2]

		test_sel_list = data1['test']
		y_predicted_test = data_1['test1']['pred']
		pred_vec1 = data_1['test1']['attention']
		y_test = self.y_signal['test']
		value_vec = []
		value_vec_1 = []
		if pred_vec1==[]:
			serial_vec = test_sel_list[:,1]
		else:
			vec1 = pred_vec1.keys()
			chrom_vec, start_vec, stop_vec, serial_vec = [], [], [], []
			for chrom_id in vec1:
				temp1 = pred_vec1[chrom_id]
				value = temp1['value'] # [predicted_attention1, predicted_attention2]
				predicted_attention1, predicted_attention2 = value
				attention_value2, attention_dict2 = predicted_attention2

				t_serial = temp1['serial']
				num1 = len(t_serial)
				id1 = mapping_Idx(self.serial,t_serial)
				b1 = np.where(id1>=0)[0]
				print(chrom_id,num1)
				if len(b1)!=num1:
					print('error!', chrom_id, len(b1))
					return 

				serial_vec.extend(self.serial[id1])
				value_vec.extend(predicted_attention1)
				value_vec_1.extend(attention_value2)

			id1 = mapping_Idx(test_sel_list[:,1],serial_vec)
			b = np.where(id1<0)[0]
			if len(b)>0:
				print('error! test_result_3_1',len(b))
				return

				y_test = y_test[id1]

		id1 = mapping_Idx(self.serial,serial_vec)
		b1 = np.where(id1<0)[0]
		if len(b1)>0:
			print('error! test_result_3_1', len(b1))
			return

		if len(value_vec)>0:
			value_vec = np.asarray(value_vec)
			print('value_vec1',value_vec.shape)
			if value_vec.ndim>1:
				value_vec = value_vec[:,0]
		else:
			value_vec = -np.ones(len(y_predicted_test))

		if len(value_vec_1)>0:
			value_vec_1 = np.asarray(value_vec_1)
			print('value_vec_1',value_vec_1.shape)
			if value_vec_1.ndim>1:
				value_vec_1 = value_vec_1[:,self.est_attention_sel1]
		else:
			value_vec_1 = -np.ones(len(y_predicted_test))

		columns = ['serial','signal','predicted_signal','predicted_attention','predicted_attention1']
		value = [serial_vec,y_test,y_predicted_test,value_vec,value_vec_1]
		data2 = self.test_result_3_sub1(id1,columns,value,output_filename,sort_flag=True,sort_column='serial')

		return data2

	def process_attention(self,predicted_attention):

		if self.method in self.attention_vec:
			predicted_attention = np.ravel(predicted_attention[:,self.flanking])

		return predicted_attention

	def process_attention_2(self,predicted_attention,vec1):

		# predicted attention: (num_sample, context_size, context_size)
		s1, s2 = predicted_attention.shape[1], predicted_attention.shape[-1]
		serial = vec1[:,self.flanking]
		quantile1 = [0.5,0.9,0.1]
		col_num = 5
		if 'est_border' in self.config: 
			id1 = self.config['est_border']
		else:
			id1 = 1

		sel_idx = range(id1,s1-id1)
		if s2<s1:
			vec1 = vec1[:,sel_idx]	# there may be bias on the border
			# predicted_attention = np.ravel(predicted_attention[:,self.flanking])
			predicted_attetion1 = predicted_attetion[:,sel_idx]	# there may be bias on the border
			mtx1 = np.zeros((len(serial),col_num)) # attention
			mtx2 = []	# received attention
			i = 0
			for t_serial in serial:
				b1 = (vec1==t_serial)
				t_attention = predicted_attention1[b1]
				mtx1[i] = list(np.quantile(t_attention,quantile1))+[np.max(t_attention),np.min(t_attention)]
				i = i+1
		else:
			predicted_attention1 = predicted_attention[:,:,sel_idx]	# there may be bias on the border
			attention1 = np.swapaxes(predicted_attention1,1,2)
			vec1_1 = vec1[:,np.newaxis,:]
			vec1_1 = np.repeat(vec1_1,vec1.shape[1],axis=1)
			vec1_1 = vec1_1[:,sel_idx]
			vec1 = vec1[:,sel_idx]
			dict1 = dict()
			num_sample = len(serial)
			col_num = 2
			mtx1 = np.zeros((num_sample,col_num),np.float32)
			i = 0
			if self.est_attention_type1==0:
				for t_serial in serial:
					b1 = (vec1==t_serial)
					t_attention = attention1[b1].ravel()
					mtx1[i] = [np.mean(t_attention),np.max(t_attention)]
			else:
				sel1 = 2
				for t_serial in serial:
					b1 = (vec1==t_serial)
					# t_attention = np.ravel(attention1[b1])
					t_attention = attention1[b1].ravel()
					id2 = vec1_1[b1].ravel()

					id_vec, count1 = np.unique(id2,return_counts=True)
					num2 = len(id_vec)
					vec2 = np.zeros((len(id_vec),sel1),np.float32)
					for i1 in range(num2):
						value1 = t_attention[id2==id_vec[i1]]
						vec2[i1] = [np.mean(value1),np.max(value1)]
					dict1[t_serial] = {'id':id_vec,'count':count1,'value':vec2}
					mtx1[i] = np.mean(vec2,axis=0)

				i = i+1

		return (mtx1, dict1)

	# attention test
	# calculate skewness
	def attention_test_1(self,data1):

		attention = np.asarray(data1['predicted_attention'])
		vec1 = [0,0.0025,0.01,0.1,0.25,0.5,0.75,0.9,0.99,0.9975,1]
		t1 = np.quantile(attention,vec1)
		ratio1 = (t1[-2]-t1[5])/(t1[5]-t1[1])
		value1 = skew(attention)
		value2 = kurtosis(attention)
		print('test attention: ', t1)
		print('ratio1 (0.0025,0.5,0.9975), skew, kurtosis', 
				ratio1, t1[1], t1[5], t1[-2], value1, value2)
		print(self.run_id,self.config['tau'],self.config['n_select'],
				self.config['activation3'],self.config['activation_self'],
				self.config['type_id1'],self.config['ratio1'],
				self.config['regularizer1'],self.config['regularizer2'],
				self.train_chromvec,self.test_chromvec)

		return (t1,ratio1,skew,kurtosis)
			
	#############################################################
	## load features and feature transformation

	# load features: phyloP scores
	def load_phyloP_score(self,filename,query_serial,feature_idx):

		# load phyloP scores
		temp1 = pd.read_csv(filename,sep='\t')
		temp1 = np.asarray(temp1)

		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,query_serial)	# mapped index
		# select dimensions of the phyloP score
		if feature_idx==-5:
			t_phyloP = phyloP_score[trans_id1]
		elif feature_idx==-6:
			t_phyloP = phyloP_score[trans_id1,0:-4]
		else:
			t_phyloP = phyloP_score[trans_id1]
			t_phyloP = t_phyloP[:,feature_idx]

		return t_phyloP

	# load features: GC-content features
	def load_gc_feature(self,signal_ori,serial_ori,query_serial,feature_idx):

		trans_id1a = mapping_Idx(serial_ori,query_serial)	# mapped index
		t_gc = signal_ori[trans_id1a]
		t_gc = t_gc[:,feature_idx]

		return t_gc

	# load features: kmer features
	def load_kmer_feature(self,filename,kmer_size,sel_idx):

		file1 = np.load(filename)
		t_signal_ori1 = np.asarray(file1)
		t_signal1 = t_signal_ori1[sel_idx]

		return t_signal1

	def load_samples(self,chrom_vec,chrom,y_label_ori,y_group1,y_signal_ori1,filename2,filename2a,kmer_size,kmer_dict1,generate):

		x_mtx_vec, y_label_vec, y_group_vec, y_signal_ori_vec = [], [], [], []
		for chrom_id in chrom_vec:
			chrom_id1 = 'chr%s'%(chrom_id)
			sel_idx = np.where(chrom==chrom_id1)[0]
			print(('sel_idx:%d')%(len(sel_idx)))

			if generate==0:
				filename2 = 'training_mtx/training2_mtx_%s.npy'%(chrom_id)
				if(os.path.exists(filename2)==True):
					x_mtx = np.load(filename2)
					x_kmer = np.load('training_mtx/training2_kmer_%s.npy'%(chrom_id))
				else:
					generate = 1

			if generate==1:
				x_kmer, x_mtx = load_seq_2(filename2a,kmer_size,kmer_dict1,sel_idx)
				np.save('training2_kmer_%s'%(chrom_id),x_kmer)
				np.save('training2_mtx_%s'%(chrom_id),x_mtx)

		x_mtx = np.transpose(x_mtx,(0,2,1))
		y_label, y_group, y_signal_ori = y_label_ori[sel_idx], y_group1[sel_idx], y_signal_ori1[sel_idx]
		x_mtx_vec.extend(x_mtx)
		y_label_vec.extend(y_label)
		y_group_vec.extend(y_group)
		y_signal_ori_vec.extend(y_signal_ori)

		x_mtx, y_label, y_group, y_signal_ori = np.asarray(x_mtx_vec), np.asarray(y_label_vec), np.asarray(y_group_vec), np.asarray(y_signal_ori_vec)
		print(x_mtx.shape,y_signal_ori.shape)
		y_signal = self.signal_normalize(y_signal_ori,[0,1])
		threshold = utility_1.signal_normalize_query(0,[np.min(y_signal_ori),np.max(y_signal_ori)],[0,1])

		return x_mtx, y_signal, y_label, threshold

	def load_samples_kmer(self,chrom_vec,chrom,seq,kmer_size,kmer_dict1,path_1):

		x_mtx_vec, y_label_vec, y_group_vec, y_signal_ori_vec = [], [], [], []
		for chrom_id in chrom_vec:
			chrom_id1 = 'chr%s'%(chrom_id)
			sel_idx = np.where(chrom==chrom_id1)[0]
			print(('sel_idx:%d')%(len(sel_idx)))

			x_kmer = load_seq_2_kmer(seq,kmer_size,kmer_dict1,chrom_id1,sel_idx)
			np.save('%s/training2_kmer_%s'%(path_1,chrom_id),x_kmer)

		return True

	def load_features_single(self,file_path,file_prefix_kmer,kmer_size1,kmer_size2):

		chrom_ori, serial_ori, chrom, serial = self.chrom_ori, self.serial_ori, self.chrom, self.serial
		# signal = self.signal
		species_id = self.species_id
		resolution = self.resolution
		self.file_path = file_path
		self.kmer_size2 = kmer_size2

		if self.type_id==0 or self.type_id==2:
			ratio = 1
		else:
			ratio = 0.5

		# filename2 = '%s/test_hg38_5k_kmer%d.npy'%(file_path,kmer_size1)
		file_prefix_kmer = 'test_%s_%s'%(species_id,resolution)
		filename2 = '%s/%s_kmer%d.npy'%(file_path,file_prefix_kmer,kmer_size1)
		# t_signal2 = self.load_kmer_feature(filename2,kmer_size,trans_id1)
		print('load kmer feature', filename2)
		self.kmer_signal_ori = np.load(filename2)
		feature_dim_kmer = [self.kmer_signal_ori.shape[1],-1]
		if kmer_size2>0:
			filename2 = '%s/%s_kmer%d.npy'%(file_path,file_prefix_kmer,kmer_size2)
			self.kmer_signal_ori2 = np.load(filename2)
			feature_dim_kmer[1] = self.kmer_signal_ori2.shape[1]
			print(self.kmer_signal_ori.shape,self.kmer_signal_ori2.shape)
			# self.kmer_signal_ori = np.hstack((self.kmer_signal_ori,self.kmer_signal_ori2))

		self.feature_dim_kmer = feature_dim_kmer
		filename2 = '%s/test_gc_%s.txt'%(file_path,resolution)
		print('load gc feature', filename2)
		data1 = pd.read_csv(filename2,sep='\t')
		colnames = list(data1)
		self.ref_serial_gc = np.asarray(data1[colnames[3]])
		self.gc_signal_ori = np.asarray(data1.loc[:,colnames[4:]])
		print(self.ref_serial_gc.shape,self.serial_ori.shape,self.gc_signal_ori.shape)

		if self.feature_dim_motif>0:
			filename2 = '%s/%s.motif.txt'%(file_path,species_id)
			data1 = pd.read_csv(filename2,sep='\t')
			colnames = list(data1)
			self.ref_serial_motif = np.asarray(data1[colnames[3]])
			self.motif_signal_ori = np.asarray(data1.loc[:,colnames[4:]])
			self.feature_dim_motif = self.motif_signal_ori.shape[1]
			print('load motif feature', filename2, self.feature_dim_motif)
			print(self.kmer_signal_ori.shape,self.gc_signal_ori.shape,self.motif_signal_ori.shape)

		else:
			print(self.kmer_signal_ori.shape,self.gc_signal_ori.shape)

		for t_chrom in self.train_chromvec:
			x_train1, train_sel_idx, train_sel_list = self.load_features_sub([t_chrom],serial_ori,chrom_ori,chrom,serial)
			# x_test, test_sel_idx, test_sel_list = self.load_features_sub(self.test_chromvec,serial_ori,chrom_ori,chrom,serial)

			x_train1_ori = np.asarray(x_train1)
			print(x_train1_ori.shape)

			feature_idx = self.feature_idx
			f_id1 = np.asarray(range(0,len(feature_idx)))
			f_id2 = np.asarray(range(len(feature_idx),len(feature_idx)+self.t_phyloP.shape[1]))
			f_id3 = np.asarray(range(f_id2[-1]+1,feature_dim_kmer[0]+f_id2[-1]+1))
			if feature_dim_kmer[1]>0:
				f_id4 = np.asarray(range(f_id3[-1]+1,feature_dim_kmer[1]+f_id3[-1]+1))

			feature_type = self.feature_type
			if feature_type==0: # GC
				temp1 = f_id1
			elif feature_type==1: # phyloP
				temp1 = f_id2
			elif feature_type==2: # K-mer5
				temp1 = f_id4
			elif feature_type==3: # K-mer6
				temp1 = f_id3
			elif feature_type==4: # GC+phyloP
				temp1 = np.hstack((f_id1,f_id2))
			elif feature_type==5: # K-mer6+phyloP
				temp1 = np.hstack((f_id2,f_id3))
			elif feature_type==6: # GC+K-mer6+phyloP
				temp1 = np.hstack((f_id1,f_id2,f_id3))
			else:
				pass
				
			if feature_type>-1:
				x_train1_ori = x_train1_ori[:,temp1]

			self.x_train1_ori = x_train1_ori
			self.feature_dim_kmer = feature_dim_kmer
			self.train_sel_list = train_sel_list
			feature_dim_kmer_1 = np.sum(feature_dim_kmer)

			filename1 = '%s_chr%s-chr%s_chr%s-chr%s'%(self.cell_type, t_chrom, t_chrom, self.test_chromvec[0], self.test_chromvec[-1])
			save_filename_ori = '%s_ori1.npy'%(filename1)
			print(save_filename_ori)
			
			d1 = (x_train1_ori,train_sel_list,train_sel_idx,feature_dim_kmer,self.feature_dim_motif)
			pickle.dump(d1, open(save_filename_ori, 'wb'), protocol=4)

		# return x_train1_ori, train_sel_list, train_sel_idx, feature_dim_kmer_1
		return True

	# vectors: serial_ori, chrom_ori, chrom, serial
	# gc data loaded together
	# motif data loaded together
	# kmer and phyloP loaded by chromosomes
	def load_chrom_feature_ori(self,serial_ori,chrom_ori,chrom,serial,chrom_id,data_list,ratio=0):

		chrom_id1 = 'chr%s'%(chrom_id)
		id1 = np.where(chrom==chrom_id1)[0]
		if ratio>0:
			num1 = int(len(id1)*ratio)
			id1 = id1[0:num1]
		# train_sel_idx.extend(id1)
		id1_ori = np.where(chrom_ori==chrom_id1)[0]

		temp1 = serial[id1]
		n1 = len(temp1)
		idx_vec = np.vstack(([int(chrom_id)]*n1,temp1)).T

		# load kmer features
		file_path = self.file_path
		kmer_size = kmer_size1
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		filename2 = '%s/Kmer%d/training_kmer_%s.npy'%(file_path,kmer_size,chrom_id)
		t_signal1 = self.load_kmer_feature(filename2,kmer_size,trans_id1)
		feature_dim_kmer1 = t_signal1.shape[1]

		if kmer_size2>0:
			kmer_size = kmer_size2
			filename2 = '%s/Kmer%d/training_kmer_%s.npy'%(file_path,kmer_size,chrom_id)
			t_signal2 = self.load_kmer_feature(filename2,kmer_size,trans_id1)
			feature_dim_kmer2 = t_signal2.shape[1]

			t_signal = np.hstack((t_signal1,t_signal2))
			feature_dim_kmer = [feature_dim_kmer1,feature_dim_kmer2]
			print(feature_dim_kmer1,feature_dim_kmer2,t_signal.shape[1])
		else:
			t_signal = t_signal1
			feature_dim_kmer = [feature_dim_kmer1,0]
			print(feature_dim_kmer1,feature_dim_kmer)

		# load phyloP scores
		filename3 = '%s/phyloP_chr%s.txt'%(file_path,chrom_id)
		feature_idx1 = self.feature_idx1
		t_phyloP = self.load_phyloP_score(filename,serial[id1],feature_idx1)

		# load gc scores
		# filename = '%s/training_gc_%s.txt'%(file_path,species_name)
		feature_idx = self.feature_idx
		gc_data_ori = data_list[0]
		t_gc = self.load_gc_feature(gc_data_ori,serial_ori,serial[id1],feature_idx)

		return t_gc, t_phyloP, t_signal, idx_vec, id1, feature_dim_kmer

	def load_chrom_feature(self,serial_ori,chrom_ori,chrom,serial,chrom_id,ratio=0):

		chrom_id1 = 'chr%s'%(chrom_id)
		id1 = np.where(chrom==chrom_id1)[0]
		if ratio>0:
			num1 = int(len(id1)*ratio)
			id1 = id1[0:num1]
		# train_sel_idx.extend(id1)
		id1_ori = np.where(chrom_ori==chrom_id1)[0]

		temp1 = serial[id1]
		n1 = len(temp1)
		idx_vec = np.vstack(([int(chrom_id)]*n1,temp1)).T

		# load phyloP scores
		# filename3 = '%s/phyloP_chr%s.txt'%(file_path,chrom_id)
		file_path = self.file_path
		file_path1 = '%s/phyloP_%s'%(file_path, self.resolution)
		filename = '%s/phyloP_%s.txt'%(file_path1,chrom_id)
		feature_idx1 = self.feature_idx1
		t_phyloP = self.load_phyloP_score(filename,serial[id1],feature_idx1)
		print('phyloP',filename, t_phyloP.shape)

		return t_phyloP, idx_vec, id1_ori, id1

	def load_features_sub(self,chromvec,serial_ori,chrom_ori,chrom,serial):
		
		t_phyloP = []
		t_sel_idx, t_sel_list = [], []
		for chrom_id in chromvec:			
			phyloP1, idx_vec, id1_ori, id1 = self.load_chrom_feature(serial_ori,chrom_ori,chrom,serial,chrom_id)

			print("trans_id1", chrom_id, id1.shape, phyloP1.shape)
			t_sel_idx.extend(id1)
			t_sel_list.extend(idx_vec)
			t_phyloP.extend(phyloP1)

		query_serial = serial[t_sel_idx]
		sel_idx = mapping_Idx(serial_ori,query_serial)
		t_kmer_signal = self.load_feature_2(self.kmer_signal_ori,sel_idx)
		if self.kmer_size2>0:
			t_kmer_signal2 = self.load_feature_2(self.kmer_signal_ori2,sel_idx)
			t_kmer_signal = np.hstack((t_kmer_signal,t_kmer_signal2))

		print('kmer signal', t_kmer_signal.shape)

		sel_idx_gc = mapping_Idx(self.ref_serial_gc,query_serial)
		print('gc',self.feature_idx)
		t_gc = self.load_feature_2(self.gc_signal_ori,sel_idx_gc,self.feature_idx)
		print('gc signal', t_gc.shape)

		self.t_phyloP = np.asarray(t_phyloP)

		if self.feature_dim_motif>0:
			sel_idx_motif = mapping_Idx(self.ref_serial_motif,query_serial)
			t_motif_signal = self.load_feature_2(self.motif_signal_ori,sel_idx_motif)
			print('motif signal', t_motif_signal.shape)
			x_vec1 = np.hstack((t_gc,self.t_phyloP,t_kmer_signal,t_motif_signal))
		else:
			x_vec1 = np.hstack((t_gc,self.t_phyloP,t_kmer_signal))

		return x_vec1, t_sel_idx, t_sel_list

	def load_features(self,file_path,file_prefix_kmer,kmer_size1,kmer_size2):

		chrom_ori, serial_ori, chrom, serial = self.chrom_ori, self.serial_ori, self.chrom, self.serial
		# signal = self.signal
		species_id = self.species_id
		resolution = self.resolution
		self.file_path = file_path
		self.kmer_size2 = kmer_size2

		if self.type_id==0 or self.type_id==2:
			ratio = 1
		else:
			ratio = 0.5

		# filename2 = '%s/test_hg38_5k_kmer%d.npy'%(file_path,kmer_size1)
		file_prefix_kmer = 'test_%s_%s'%(species_id,resolution)
		filename2 = '%s/%s_kmer%d.npy'%(file_path,file_prefix_kmer,kmer_size1)
		# t_signal2 = self.load_kmer_feature(filename2,kmer_size,trans_id1)
		print('load kmer feature', filename2)
		self.kmer_signal_ori = np.load(filename2)
		feature_dim_kmer = [self.kmer_signal_ori.shape[1],-1]
		if kmer_size2>0:
			filename2 = '%s/%s_kmer%d.npy'%(file_path,file_prefix_kmer,kmer_size2)
			self.kmer_signal_ori2 = np.load(filename2)
			feature_dim_kmer[1] = self.kmer_signal_ori2.shape[1]
			print(self.kmer_signal_ori.shape,self.kmer_signal_ori2.shape)
			# self.kmer_signal_ori = np.hstack((self.kmer_signal_ori,self.kmer_signal_ori2))

		self.feature_dim_kmer = feature_dim_kmer
		filename2 = '%s/test_gc_%s.txt'%(file_path,resolution)
		print('load gc feature', filename2)
		data1 = pd.read_csv(filename2,sep='\t')
		colnames = list(data1)
		self.ref_serial_gc = np.asarray(data1[colnames[3]])
		self.gc_signal_ori = np.asarray(data1.loc[:,colnames[4:]])
		print(self.ref_serial_gc.shape,self.serial_ori.shape,self.gc_signal_ori.shape)

		if self.feature_dim_motif>0:
			filename2 = '%s/%s.motif.txt'%(file_path,species_id)
			data1 = pd.read_csv(filename2,sep='\t')
			colnames = list(data1)
			self.ref_serial_motif = np.asarray(data1[colnames[3]])
			self.motif_signal_ori = np.asarray(data1.loc[:,colnames[4:]])
			self.feature_dim_motif = self.motif_signal_ori.shape[1]
			print('load motif feature', filename2, self.feature_dim_motif)
			print(self.kmer_signal_ori.shape,self.gc_signal_ori.shape,self.motif_signal_ori.shape)

		else:
			print(self.kmer_signal_ori.shape,self.gc_signal_ori.shape)

		x_train1, train_sel_idx, train_sel_list = self.load_features_sub(self.train_chromvec,serial_ori,chrom_ori,chrom,serial)
		x_test, test_sel_idx, test_sel_list = self.load_features_sub(self.test_chromvec,serial_ori,chrom_ori,chrom,serial)

		x_train1_ori, x_test_ori = np.asarray(x_train1), np.asarray(x_test)

		print(x_train1.shape,len(train_sel_idx),len(test_sel_idx))

		feature_idx = self.feature_idx
		f_id1 = np.asarray(range(0,len(feature_idx)))
		f_id2 = np.asarray(range(len(feature_idx),len(feature_idx)+self.t_phyloP.shape[1]))
		f_id3 = np.asarray(range(f_id2[-1]+1,feature_dim_kmer[0]+f_id2[-1]+1))
		if feature_dim_kmer[1]>0:
			f_id4 = np.asarray(range(f_id3[-1]+1,feature_dim_kmer[1]+f_id3[-1]+1))

		feature_type = self.feature_type
		if feature_type==0: # GC
			temp1 = f_id1
		elif feature_type==1: # phyloP
			temp1 = f_id2
		elif feature_type==2: # K-mer5
			temp1 = f_id4
		elif feature_type==3: # K-mer6
			temp1 = f_id3
		elif feature_type==4: # GC+phyloP
			temp1 = np.hstack((f_id1,f_id2))
		elif feature_type==5: # K-mer6+phyloP
			temp1 = np.hstack((f_id2,f_id3))
		elif feature_type==6: # GC+K-mer6+phyloP
			temp1 = np.hstack((f_id1,f_id2,f_id3))
		else:
			pass
			
		if feature_type>-1:
			x_train1_ori, x_test_ori = x_train1_ori[:,temp1], x_test_ori[:,temp1]

		self.x_train1_ori = x_train1_ori
		self.x_test_ori = x_test_ori
		self.feature_dim_kmer = feature_dim_kmer
		self.train_sel_list = train_sel_list
		self.test_sel_list = test_sel_list
		feature_dim_kmer_1 = np.sum(feature_dim_kmer)
		# print(x_train1_ori.shape, x_test_ori.shape)

		return x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer_1

	# load signals by index for training and test data
	def load_signal(self,train_sel_idx,test_sel_idx):

		y_signal_train1 = self.signal[train_sel_idx]
		y_signal_test = self.signal[test_sel_idx]

		return y_signal_train1, y_signal_test
	
	# the mapped indices of selected regions
	def load_map_idx(self,ref_filename,filename1):

		temp1 = pd.read_csv(ref_filename,sep='\t')
		temp2 = pd.read_csv(filename1,sep='\t')
		colname1, colname2 = list(temp1), list(temp2)
		chrom1, start1, stop1, serial1 = temp1[colname1[0]], temp1[colname1[1]], temp2[colname2[2]], temp2[colname2[3]]
		chrom2, start2, stop2, serial2 = temp2[colname2[0]], temp2[colname2[1]], temp2[colname2[2]], temp2[colname2[3]]

		map_idx = mapping_Idx(serial1,serial2)

		return serial1, serial2, map_idx

	# load and save the predicted genomic signals
	def load_predicted_signal(self,filename1,output_filename,x_min=0,x_max=1):

		data1 = np.load(filename1)
		data2 = data1[()]
		y_predicted_valid = data2[0]['y_predicted_valid']
		y_valid = data2[0]['y_valid']
		y_predicted_test = data2[0]['y_predicted_test']
		y_test = data2[0]['y_test']
		valid_sel_list = data2['valid']
		test_sel_list = data2['test']

		valid_serial = valid_sel_list[:,1]
		test_serial = test_sel_list[:,1]

		id1 = mapping_Idx(self.serial,valid_serial)
		id2 = mapping_Idx(self.serial,test_serial)

		fields = ['chrom','start','stop','signal','predicted']
		valid1 = pd.DataFrame(columns=fields)
		valid1['chrom'], valid1['start'], valid1['stop'], valid1['signal'], valid1['predicted'] = self.chrom[id1], self.start[id1], self.stop[id1], y_valid, y_predicted_valid

		test1 = pd.DataFrame(columns=fields)
		test1['chrom'],test1['start'], test1['stop'], test1['signal'], test1['predicted'] = self.chrom[id2], self.start[id2], self.stop[id2], y_test, y_predicted_test
		
		valid1.to_csv('valid_%s.txt'%(output_filename),header=True,index=False,sep='\t')
		test1.to_csv('test_%s.txt'%(output_filename),header=True,index=False,sep='\t')

		return True

	def load_features_transform(self,file_path,file_prefix_kmer,prefix,feature_dim_motif,feature_dim_transform=[50,50]):

		self.feature_dim_motif = feature_dim_motif
		kmer_size1, kmer_size2 = self.kmer_size[0], self.kmer_size[1]

		prefix = self.filename_load
		save_filename_ori = '%s_%s_ori1.npy'%(prefix,self.cell_type)
		self.save_filename_ori = save_filename_ori

		if os.path.exists(save_filename_ori)==True:
			print("loading %s"%(save_filename_ori))
			x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer, feature_dim_motif = pickle.load(open(save_filename_ori, 'rb'))
			self.feature_dim_motif = feature_dim_motif
		else:
			print("loading...")
			x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer = self.load_features(file_path,file_prefix_kmer,kmer_size1,kmer_size2)
			d1 = (x_train1_ori,x_test_ori,train_sel_list,test_sel_list,train_sel_idx,test_sel_idx,feature_dim_kmer, self.feature_dim_motif)
			pickle.dump(d1, open(save_filename_ori, 'wb'), protocol=4)

		y_signal_train1, y_signal_test = self.load_signal(train_sel_idx, test_sel_idx)
		y_signal_train1_ori = self.signal_normalize(y_signal_train1,[0,1])
		y_signal_test_ori = self.signal_normalize(y_signal_test,[0,1])
		# print(x_train1_ori.shape, y_signal_train1.shape, x_test_ori.shape, y_signal_test.shape)

		feature_dim = feature_dim_transform
		sub_sample_ratio = 1
		shuffle = 0
		normalize = 0
		train_sel_list, test_sel_list = np.asarray(train_sel_list), np.asarray(test_sel_list)
		self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list
		tol = self.tol
		L = self.flanking
		# print(feature_dim_kmer, feature_dim_transform)
		self.feature_dim_kmer = feature_dim_kmer

		for type_id2 in self.t_list:
			print("feature transform")
			prefix = self.filename_load
			# 1: PCA; 2: SVD for motif
			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim[0],feature_dim[1])

			x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim, shuffle, 
																sub_sample_ratio, type_id2, normalize)
			np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
									train_sel_list,test_sel_list),allow_pickle=True)

		return x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list



