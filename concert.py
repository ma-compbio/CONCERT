import pandas as pd
import numpy as np
import processSeq
import sys

import keras
keras.backend.image_data_format()
from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense, Reshape, Lambda, Conv1D, Flatten, MaxPooling1D, UpSampling1D, GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional
from keras.layers import BatchNormalization, Dropout, Concatenate, Embedding
from keras.layers import Activation,Dot,dot
from keras.models import Model, clone_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.constraints import unitnorm
from keras_layer_normalization import LayerNormalization
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,accuracy_score,matthews_corrcoef
from processSeq import load_seq_1, kmer_dict, load_signal_1, load_seq_2, load_seq_2_kmer, load_seq_altfeature_1
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost
import pickle
# from utility1_1 import mapping_Idx

import os.path
from optparse import OptionParser
import genUnlabelData_1,genLabelData_1,genVecs_1
from sklearn.svm import SVR

import sklearn as sk
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import time

from utility_1 import sample_select2, sample_select2a, sample_select2a_3
from utility_1 import dimension_reduction1, feature_transform1, get_model2a1, get_model2a_sequential, get_model2a1_sequential, get_model2a1_attention_sequential
from utility_1 import get_model2a_1, get_model2a_2
from utility_1 import get_model2a1_attention1_sequential, get_model2a1_attention2_sequential
from utility_1 import get_model2a1_attention1_2_sequential, get_model2a1_attention2_2_sequential
from utility_1 import get_model2a_attention1_sequential, get_model2a_attention2_sequential, get_model2a2_attention
from utility_1 import read_predict, read_predict_weighted
from utility_1 import sample_select2a_pre
# from utility_1 import get_model2a2_attention, get_model2a2_1, get_model2a2_1_predict
from utility_1 import get_model2a1_attention, get_model2a1_sequential
from utility_1 import get_model2a1_attention_1, get_model2a1_attention_2, get_model2a_sequential
from utility_1 import search_region_include, search_region, aver_overlap_value
from processSeq import load_seq_altfeature

def mapping_Idx(serial1,serial2):

	if len(np.unique(serial1))<len(serial1):
		print("error! ref_serial not unique")
		return

	unique_flag = 1
	t_serial2 = np.unique(serial2,return_inverse=True)
	if len(t_serial2[0])<len(serial2):
		serial2_ori = serial2.copy()	
		serial2 = t_serial2[0]
		unique_flag = 0

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
	print('mapping',len(ref_serial),len(map_serial))
	print(len(vec1),ref_serial[-1],map_serial[-1])
	print(len(b),len(b1),len(idx))
	idx1[map_sortedIdx[b1]] = idx

	if unique_flag==0:
		idx1 = idx1[t_serial2[1]]

	return np.int64(idx1)

def signal_normalize(signal, scale):

	s1, s2 = scale[0], scale[1]

	s_min, s_max = np.min(signal), np.max(signal)
	scaled_signal = s1+(signal-s_min)*1.0/(s_max-s_min)*(s2-s1)

	return scaled_signal

def signal_normalize_query(query_point, scale_ori, scale):

	s1, s2 = scale[0], scale[1]
	s_min, s_max = scale_ori[0], scale_ori[1]
	scaled_signal = s1+(query_point-s_min)*1.0/(s_max-s_min)*(s2-s1)

	return scaled_signal

def balance_data(X,t,y):
	pos_index = np.where(y == 1)[0]
	neg_index = np.where(y == 0)[0]
	np.random.shuffle(neg_index)
	neg_index = neg_index[:len(pos_index)]

	X = np.concatenate((X[pos_index], X[neg_index]), axis=0)
	y = np.concatenate((y[pos_index], y[neg_index]), axis=0)
	t = np.concatenate((t[pos_index], t[neg_index]), axis=0)

	return X,t,y

def build_classweight(y):
	dict1 = {}
	for l in np.unique(y):
		dict1[l] = len(y) * 1.0 / np.sum(y == l)
	print (dict1)

	return dict1

def score_function(y_test, y_pred, y_proba):

	auc = roc_auc_score(y_test,y_proba)
	aupr = average_precision_score(y_test,y_proba)
	precision = precision_score(y_test,y_pred)
	recall = recall_score(y_test,y_pred)
	accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))

	# print(auc,aupr,precision,recall)
	
	return accuracy, auc, aupr, precision, recall

def score_2a(y, y_predicted):

	score1 = mean_squared_error(y, y_predicted)
	score2 = pearsonr(y, y_predicted)
	score3 = explained_variance_score(y, y_predicted)
	score4 = mean_absolute_error(y, y_predicted)
	score5 = median_absolute_error(y, y_predicted)
	score6 = r2_score(y, y_predicted)
	vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6]

	return vec1

class roc_callback(keras.callbacks.Callback):
	def __init__(self,X,y):
		self.x = X
		self.y = y

	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		y_proba = model.predict(self.x,batch_size = BATCH_SIZE)
		y_pred = ((y_proba > 0.5) * 1.0).reshape((-1))
		y_test = self.y
		print("accuracy", np.sum(y_test == y_pred) / len(y_test))

		print("roc", roc_auc_score(y_test, y_proba))
		print("aupr", average_precision_score(y_test, y_proba))

		print("precision", precision_score(y_test, y_pred))
		print("recall", recall_score(y_test, y_pred))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return

def read_phyloP(species_name):

	path1 = '/volume01/yy3/seq_data/dl/replication_timing'
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
			# if cnt==0:
			# 	b1 = np.where((start>=start_ori[i])&(stop<stop_ori[i]))[0]
			# else:
			# 	t1 = b1[-1]+1
			# 	while start[t1]<start_ori[i]:
			# 		t1 += 1
			# 	if start[t1]>=start_ori[i]:
			# 		id1 = t1
			# 		while start[t1]<start_ori[i]:
			# 			t1 += 1

			t1 = b1[-1]+1
			b1 = np.where((start[t1:]>=start_ori[i])&(stop[t1:]<stop_ori[i]))[0]+t1
			# b1 = []
			# while start[t1]<start_ori[i]:
			# 	t1 += 1
			# if start[t1]>=start_ori[i]:
			# 	id1 = t1
			# 	while stop[t1]<stop_ori[i]:
			# 		t1 += 1
			# 	id2 = t1
			# 	b1 = np.asarray(range(id1,id2))
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

def read_phyloP_single(species_name,chrom_id):

	path1 = '/volume01/yy3/seq_data/dl/replication_timing'
	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path1,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	num_sample = len(chrom_ori)
	# chrom_vec = np.unique(chrom_ori)
	chrom_vec = [chrom_id]
	n_level, offset, magnitude = 15, 10, 2

	for chrom_id in chrom_vec:
		filename1 = '%s/phyloP/hg19.phyloP100way.%s.bedGraph'%(path1,chrom_id)
		data1 = pd.read_csv(filename1,header=None,sep='\t')
		chrom, start, stop, score = data1[0], data1[1], data1[2], data1[3]
		len1 = stop-start
		b = np.where(chrom_ori==chrom_id)[0]
		num_sample1 = len(b)
		vec1 = np.zeros((num_sample1,n_level+4))
		print(chrom_id,len(chrom),len(b))
		cnt = 0
		pre_b1 = [-1]
		b1 = pre_b1
		for i in b:
			# if cnt==0:
			# 	b1 = np.where((start>=start_ori[i])&(stop<stop_ori[i]))[0]
			# else:
			# 	t1 = b1[-1]+1
			# 	while start[t1]<start_ori[i]:
			# 		t1 += 1
			# 	if start[t1]>=start_ori[i]:
			# 		id1 = t1
			# 		while start[t1]<start_ori[i]:
			# 			t1 += 1

			t1 = pre_b1[-1]+1
			b1 = np.where((start[t1:]>=start_ori[i])&(stop[t1:]<stop_ori[i]))[0]+t1
			# b1 = []
			# while start[t1]<start_ori[i]:
			# 	t1 += 1
			# if start[t1]>=start_ori[i]:
			# 	id1 = t1
			# 	while stop[t1]<stop_ori[i]:
			# 		t1 += 1
			# 	id2 = t1
			# 	b1 = np.asarray(range(id1,id2))
			if len(b1)==0:
				continue

			t_len1, t_score = np.asarray(len1[b1]), np.asarray(score[b1])
			s1 = 0
			s2 = np.sum(t_len1)
			i1 = cnt
			for j in range(0,n_level):
				temp1 = (j-offset)*magnitude
				b2 = np.where((t_score<temp1+magnitude)&(t_score>=temp1))[0]
				# print(b2)
				vec1[i1,j] = np.sum(t_len1[b2])*1.0/s2
				s1 = s1+temp1*vec1[i1,j]

			vec1[i1,n_level:n_level+4] = [s1,np.median(t_score),np.max(t_score),np.min(t_score)]
			
			cnt += 1
			pre_b1 = b1
			if cnt%1000==0:
				print(chrom_id,cnt,len(b1),s2,vec1[i1,12:16])
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

def read_phyloP_pre(file_path):

	# file1 = pd.read_csv(ref_filename,sep='\t')	
	# # col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	# colnames = list(file1)
	# col1, col2, col3 = colnames[0], colnames[1], colnames[2]
	# chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	# num_sample = len(chrom_ori)
	# chrom_vec = np.unique(chrom_ori)
	chrom_vec = [1,2,3,4,5]
	# n_level, offset, magnitude = 15, 10, 2

	for chrom_id in chrom_vec:
		# filename1 = '%s/hg19.phyloP100way.%s.bedGraph'%(file_path,chrom_id)
		filename1 = '%s/chr%s.phyloP100way.bedGraph'%(file_path,chrom_id)
		data1 = pd.read_csv(filename1,header=None,sep='\t')
		chrom, start, stop, score = data1[0], data1[1], data1[2], data1[3]
		print(len(chrom),np.max(score),np.min(score))

	return True

def read_phyloP_1_ori(ref_filename,header,file_path,chrom_vec,n_level=15,offset=10,magnitude=2):

	# path1 = '/volume01/yy3/seq_data/dl/replication_timing'
	# filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path1,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
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
	region_len = stop_ori-start_ori

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
		pre_b1 = [-1]
		b1 = pre_b1
		pre_stop = 0
		for i in b:
			t1 = pre_b1[-1]+1
			window1 = region_len[i] + start_ori[i] - pre_stop + 10
			b1 = np.where((start[t1:t1+window1]>=start_ori[i])&(stop[t1:t1+window1]<stop_ori[i]))[0]+t1
			pre_stop = stop_ori[i]
			if len(b1)==0:
				continue

			t_len1, t_score = np.asarray(len1[b1]), np.asarray(score[b1])
			t_score[t_score>score_max] = score_max-1e-04
			s1 = 0
			s2 = np.sum(t_len1)
			i1 = cnt
			for j in range(0,n_level):
				temp1 = (j-offset)*magnitude
				b2 = np.where((t_score<temp1+magnitude)&(t_score>=temp1))[0]
				# print(b2)
				vec1[i1,j] = np.sum(t_len1[b2])*1.0/s2
				s1 = s1+temp1*vec1[i1,j]

			vec1[i1,n_level:n_level+4] = [s1,np.median(t_score),np.max(t_score),np.min(t_score)]
			
			cnt += 1
			pre_b1 = b1
			if cnt%1000==0:
				print(chrom_id,cnt,len(b1),s2,vec1[i1,12:16])
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

def read_phyloP_1(ref_filename,header,file_path,chrom_vec,n_level=15,offset=10,magnitude=2):

	file1 = pd.read_csv(ref_filename,header=header,sep='\t')
	
	# col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	colnames = list(file1)
	col1, col2, col3, col4 = colnames[0], colnames[1], colnames[2], colnames[3]
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1[col4])
	num_sample = len(chrom_ori)
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
				b1, start_idx = search_region_include(position, start, stop, m_idx, start_idx)

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

def generate_serial_start(filename1,chrom,start,stop):

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
	
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	ref_chrom, chrom_size = np.asarray(data1[0]), np.asarray(data1[1])
	serial_start = 0
	serial_vec = np.zeros(len(chrom))
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

class Reader(object):
	# Implementation of N-step Advantage Actor Critic.
	# This class inherits the Reinforce class, so for example, you can reuse
	# generate_episode() here.

	def __init__(self, ref_filename, feature_idvec = [1,1,1,1]):
		# Initializes RepliSeq
		self.ref_filename = ref_filename
		self.feature_idvec = feature_idvec
		
	def generate_serial(self,filename1,filename2,output_filename,header=None):

		# filename1 = '/volume01/yy3/seq_data/genome/hg38.chrom.sizes'

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

	def merge_file(self,ref_filename,output_filename,header=None):

		celltype_vec = ['H1','HCT116','H9']
		path1 = '.'
		
		if header==None:
			file1 = pd.read_csv(ref_filename,header=None,sep='\t')
		else:
			file1 = pd.read_csv(ref_filename,sep='\t')
		colnames = list(file1)		
		# col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
		col1, col2, col3, col_serial = colnames[0], colnames[1], colnames[2], colnames[3]
		chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1[col_serial])
		print('load ref serial', serial_ori.shape)
		num_ratio = 16
		for t_celltype in celltype_vec:
			print(t_celltype)
			replicate = 'R1'
			serial_list = []
			serial_vec = []
			signal_vec = []
			for i in range(1,num_ratio+1):
				# filename2 = '%s/%s/%s/%s_%s.S%d.ratio.sorted.bedGraph'%(path1,t_celltype,replicate,t_celltype,replicate,i)
				filename1 = '%s/%s/%s/%s_%s.S%d.ratio.sorted.bed'%(path1,t_celltype,replicate,t_celltype,replicate,i)
				data1 = pd.read_csv(filename1,header=None,sep='\t')
				serial = np.asarray(data1[3])
				serial_list = np.union1d(serial,serial_list)
				serial_vec.append(serial)
				signal_vec.append(np.asarray(data1[4]))
		
			idx = mapping_Idx(serial_ori,serial_list)
			data1 = file1.loc[idx,colnames]
			data1.to_csv('idx.txt',header=False,index=False,sep='\t')
			data1 = pd.read_csv(filename2,header=None,sep='\t')
			mtx1 = np.zeros((len(serial_list),num_ratio))
			for i in range(0,num_ratio):
				id1 = mapping_Idx(serial_list,serial_vec[i])
				mtx1[id1,i] = signal_vec[i]

			data2 = pd.DataFrame(columns=list(range(1,num_ratio+1)),data=mtx1)
			data3 = pd.concat([data1,data2], axis=1, join='outer', ignore_index=True, 
								keys=None, levels=None, names=None, verify_integrity=False, copy=True)

			data3.to_csv('%s_%s'%(t_celltype,output_filename),header=False,index=False,sep='\t')
		
		return data3

	def run_1(self):

		filename1 = '/volume01/yy3/seq_data/genome/hg38.chrom.sizes'

		celltype_vec = ['H1','HCT116','H9']
		# celltype_vec = ['H1']
		# path1 = '/volume01/yy3/seq_data/dl/replication_timing3/data_2'
		path1 = '/volume01/yy3/seq_data/dl/replication_timing2'
		num_ratio = 16
		header = None
		for t_celltype in celltype_vec:
			print(t_celltype)
			replicate = 'R1'
			for i in range(1,num_ratio+1):
				filename2 = '%s/%s/%s/%s_%s.S%d.ratio.sorted.bedGraph'%(path1,t_celltype,replicate,t_celltype,replicate,i)
				output_filename = '%s/%s/%s/%s_%s.S%d.ratio.sorted.bed'%(path1,t_celltype,replicate,t_celltype,replicate,i)
				self.generate_serial(filename1,filename2,output_filename,header)

	def run_2(self):

		file_path = '/volume01/yy3/seq_data/dl/replication_timing3/phyloP'
		read_phyloP_pre(file_path)


	def run_3(self,chrom_idvec):

		ref_filename = '/volume01/yy3/seq_data/dl/replication_timing3/hg38_10k_serial.bed'
		# file1 = pd.read_csv(ref_filename,sep='\t')
		header = None
		n_level, offset, magnitude = 15, 10, 2	
		file_path = '/volume01/yy3/seq_data/dl/replication_timing3/phyloP'
		read_phyloP_1_ori(ref_filename,header,file_path,chrom_idvec,n_level,offset,magnitude)

	def run_5(self,resolution):

		ref_filename = '/volume01/yy3/seq_data/dl/replication_timing3/hg38_%s_serial.bed'%(resolution)
		header = None
		filename = '/volume01/yy3/seq_data/dl/replication_timing3/hg38_%s_seq'%(resolution)
		output_filename = 'test_gc_%s.txt'%(resolution)
		read_gc_1(ref_filename,header,filename,output_filename)

	def load_motif_ori(self,filename1,motif_filename,output_filename):

		mtx1, chrom, start, stop, colnames = read_motif_1(motif_filename)

		serial_vec, start_vec = generate_serial_start(filename1,chrom,start,stop)

		if output_filename!=None:
			colnames2 = colnames[0:3]+['serial']+colnames[3:]
			data2 = pd.DataFrame(columns=colnames2)
			data2['chrom'], data2['start'], data2['stop'], data2['serial'] = chrom, start, stop, serial_vec

			num1 = len(colnames)
			for i in range(3,num1):
				print(colnames[i])
				data2[colnames[i]] = mtx1[:,i-3]

			data2.to_csv(output_filename,header=True,index=False,sep='\t')

		return True

	def load_motif(self,filename1,motif_filename,output_filename):

		mtx1, chrom, start, stop, colnames = read_motif_1(motif_filename)

		serial_vec, start_vec = generate_serial_start(filename1,chrom,start,stop)

		if output_filename!=None:
			colnames2 = ['chrom','start','stop','serial']
			data2 = pd.DataFrame(columns=colnames2)
			data2['chrom'], data2['start'], data2['stop'], data2['serial'] = chrom, start, stop, serial_vec

			data3 = pd.DataFrame(columns=colnames[3:],data=mtx1)

			data1 = pd.concat([data2,data3], axis=1, join='outer', ignore_index=True, 
								keys=None, levels=None, names=None, verify_integrity=False, copy=True)
			# num1 = len(colnames)
			# for i in range(3,num1):
			# 	print(colnames[i])
			# 	data2[colnames[i]] = mtx1[:,i-3]

			data1.to_csv(output_filename,header=True,index=False,sep='\t')
			print('data1',data1.shape)

		return True

class RepliSeq(object):
	# Implementation of N-step Advantage Actor Critic.
	# This class inherits the Reinforce class, so for example, you can reuse
	# generate_episode() here.

	def __init__(self, chromosome,run_id,generate,chromvec,test_chromvec,n_epochs,species_id,
					featureid,type_id,cell,method,ftype,ftrans,tlist,flanking,normalize,
					hidden_unit,batch_size,lr=0.001,step=5,
					activation='relu',min_delta=0.001,
					attention=1,fc1=0,fc2=0,kmer_size=[6,5],tol=5):
		# Initializes RepliSeq
		self.chromosome = chromosome
		self.run_id = run_id
		self.generate = generate
		self.train_chromvec = chromvec
		print('test_chromvec',test_chromvec)
		self.test_chromvec = test_chromvec
		self.n_epochs = n_epochs
		self.species_id = species_id
		self.type_id = type_id
		self.cell_type = cell
		self.method = method
		self.ftype = ftype
		self.ftrans = ftrans[0]
		self.ftrans1 = ftrans[1]
		self.t_list = tlist
		self.flanking = flanking
		self.flanking1 = 3
		self.normalize = normalize
		self.batch_size = batch_size
		config = dict(output_dim=hidden_unit,fc1_output_dim=fc1,fc2_output_dim=fc2,n_epochs=n_epochs,batch_size=batch_size)
		self.config = config
		self.tol = tol
		self.attention = attention
		self.lr = lr
		self.step = step
		self.feature_type = -1
		self.kmer_size = kmer_size
		self.activation = activation
		self.min_delta = min_delta
		self.chromvec_sel = chromvec
		
		if method>10:
			self.predict_context = 1
		else:
			self.predict_context = 0

		if ftype[0]==-5:
			self.feature_idx1= -5 # full dimensions
		elif ftype[0]==-6:
			self.feature_idx1 = -6	# frequency dimensions
		else:
			self.feature_idx1 = ftype

		# self.feature_idx = [0,2]
		self.feature_idx = featureid

		path2 = '.'
		species_name = 'hg38'
		print(self.test_chromvec)
		filename1 = '%s_chr%s-chr%s_chr%s-chr%s'%(self.cell_type, self.train_chromvec[0], self.train_chromvec[-1], self.test_chromvec[0], self.test_chromvec[-1])
		self.filename_load = filename1
		print(self.filename_load,self.method,self.predict_context,self.attention)
		self.set_generate(generate,filename1)

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
		y_signal = signal_normalize(y_signal_ori,[0,1])
		threshold = signal_normalize_query(0,[np.min(y_signal_ori),np.max(y_signal_ori)],[0,1])

		return x_mtx, y_signal, y_label, threshold

	def load_samples_kmer(chrom_vec,chrom,seq,kmer_size,kmer_dict1,path_1):

		x_mtx_vec, y_label_vec, y_group_vec, y_signal_ori_vec = [], [], [], []
		for chrom_id in chrom_vec:
			chrom_id1 = 'chr%s'%(chrom_id)
			sel_idx = np.where(chrom==chrom_id1)[0]
			print(('sel_idx:%d')%(len(sel_idx)))

			x_kmer = load_seq_2_kmer(seq,kmer_size,kmer_dict1,chrom_id1,sel_idx)
			np.save('%s/training2_kmer_%s'%(path_1,chrom_id),x_kmer)

		return True

	def dimension_reduction(self,x_ori,feature_dim,shuffle,sub_sample,type_id):

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

		# # save transfrom model
		# filename1 = '%s_%d_dimensions.h5'%(self.filename_load,feature_dim)
		# # np.save(filename1, self.dimension_model)
		# pickle.dump(self.dimension_model, open(filename1, 'wb'))
		# # self.dimension_model = pickle.load(open(filename1, 'rb'))

		return x

	def feature_transform_single(self, x_ori1, feature_dim_transform, shuffle, sub_sample_ratio, type_id, normalize=0):
		
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

		# num_train, num_test = x_train.shape[0], x_test.shape[0]
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
		# x_train1, x_test1 = x1[0:num_train], x1[num_train:num_train+num_test]
		# print(x_train.shape,x_train1.shape,x_test.shape,x_test1.shape)
		print('feature transform single',x1.shape)

		return x1

	def load_features_ori(self,chromvec):

		x_vec1 = []
		idx_sel_list1 = []
		for t_chrom in chromvec:
			cell_type1 = 'GM12878'
			filename1 = '%s_chr%s-chr%s_chr10-chr10'%(cell_type1, t_chrom, t_chrom)
			save_filename_ori = '%s_ori1.npy'%(filename1)
			print(save_filename_ori)
			if os.path.exists(save_filename_ori)==True:
				print("loading %s"%(save_filename_ori))
				# x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer, feature_dim_motif = np.load(save_filename_ori,allow_pickle=True)
				x_train1_ori,train_sel_list,train_sel_idx,feature_dim_kmer,feature_dim_motif = pickle.load(open(save_filename_ori, 'rb'))
				# self.feature_dim_motif = feature_dim_motif
				x_vec1.extend(x_train1_ori)
				idx_sel_list1.append(train_sel_list)
			else:
				print("file does not exist!")

		x_vec1 = np.asarray(x_vec1)
		return x_vec1, idx_sel_list1

	def test_predict(self,model_path,train_chromvec,test_chromvec,feature_dim_transform,output_filename):

		# x_train, train_sel_list1 = self.load_features_ori(train_chromvec)
		# print('x_train', x_train.shape)
		# vec1 = self.load_features_transform_predict(model_path,x_train,test_chromvec,feature_dim_transform)
		vec1 = self.load_features_transform_predict_1(model_path,test_chromvec,feature_dim_transform,output_filename)

	def test_predict_1(self,filename1,filename2,output_filename):

		# x_train, train_sel_list1 = self.load_features_ori(train_chromvec)
		# print('x_train', x_train.shape)
		# vec1 = self.load_features_transform_predict(model_path,x_train,test_chromvec,feature_dim_transform)
		# vec1 = self.load_features_transform_predict_1(model_path,test_chromvec,feature_dim_transform,output_filename)

		self.load_local_serial(filename1)
		# filename2 = 'attention1_%s.npy'%(output_filename)
		data1 = np.load(filename2,allow_pickle=True)
		data1 = data1[()]
		print(data1.keys())

		attention1 = data1['attention1']
		vec1 = data1['vec1']
		vec_local1 = data1['vec1_local']
		serial_list = np.unique(vec1)
		print(attention1.shape, vec1.shape, vec_local1.shape)

		attention1 = np.swapaxes(attention1,1,2)
		attention1 = attention1.reshape(attention1.shape[0]*attention1.shape[1],attention1.shape[-1])
		
		vec1 = np.ravel(vec1)
		num1 = len(serial_list)
		list1 = np.zeros((num1,4))
		t1 = np.quantile(attention1,[0.1,0.5,0.75,0.9,0.95])
		print(t1)
		for i in range(0,num1):
			t_serial = serial_list[i]
			b1 = np.where(vec1==t_serial)[0]
			temp1 = np.ravel(attention1[b1])
			m1,m2,m_1,m_2 = np.max(temp1), np.min(temp1), np.median(temp1), np.mean(temp1)
			list1[i] = [m1,m2,m_1,m_2]

		id1 = mapping_Idx(self.serial,serial_list)
		fields = ['chrom','start','stop','serial','max','min','median','mean']
		data1 = pd.DataFrame(columns=fields)
		data1[fields[0]], data1[fields[1]], data1[fields[2]], data1[fields[3]] = self.chrom[id1], self.start[id1], self.stop[id1], self.serial[id1]
		for i in range(4,8):
			data1[fields[i]] = list1[:,i-4]

		data1.to_csv(output_filename,header=True,index=False,sep='\t')

		return t1

	def load_features_transform_predict(self,model_path,x_train,test_chromvec, feature_dim_transform):

		# x_train_list1, train_sel_list1 = self.load_features_ori(train_chromvec)
		feature_dim_motif = 769
		# feature_dim_transform = [50,25]
		tol = self.tol
		L = self.flanking
		run_id = self.run_id
		flanking1 = self.flanking1
		vec1 = []

		context_size = 2*L+1
		# config = dict(fc1_output_dim=5,fc2_output_dim=0,n_epochs=10)
		# config = dict(feature_dim=x_train.shape[-1],output_dim=32,fc1_output_dim=0,fc2_output_dim=0,n_epochs=100,batch_size=128)
		config = self.config
		# config['feature_dim'] = x_train.shape[-1]
		config['feature_dim'] = np.sum(feature_dim_transform)+21
		config['lr'] = self.lr
		config['activation'] = self.activation

		model = get_model2a1_attention(context_size, config)
		print('loading model...')
		model.load_weights(model_path)

		# self.load_local_serial(filename1)
		# local_serial, local_signal = self.load_local_serial(filename1)			
		for t_chrom in test_chromvec:
			x_test1, test_sel_list1 = self.load_features_transform_single(x_train,[t_chrom],feature_dim_motif,feature_dim_transform)
			t_chrom1 = 'chr%s'%(t_chrom)
			id1 = np.where(self.chrom==t_chrom1)[0]
			t_serial = self.serial[id1]
			y_signal_test = self.signal[id1]
			id2 = mapping_Idx(test_sel_list1[:,1],t_serial)		
			y_test_temp1 = np.ones(x_test1.shape[0])

			x_test, y_test_temp, vec_test, vec_test_local = sample_select2a(x_test1, y_test_temp1, test_sel_list1, tol, L)
			y_predicted_test = model.predict(x_test)

			y_predicted_test_ori = read_predict(y_predicted_test, vec_test_local, [], flanking1)
			y_predicted_test = y_predicted_test_ori[id2]
			print('predict', y_predicted_test_ori.shape, y_predicted_test.shape)
			
			# y_signal_train1, y_signal_test = self.load_signal(train_sel_idx, test_sel_idx)
			# y_signal_train1_ori = signal_normalize(y_signal_train1,[0,1])
			y_test = signal_normalize(y_signal_test,[0,1])

			temp1 = score_2a(y_test, y_predicted_test)
			vec1.append(temp1)
			print(temp1)

		return vec1

	def load_features_transform_predict_1(self,model_path,test_chromvec, feature_dim_transform,output_filename):

		# x_train_list1, train_sel_list1 = self.load_features_ori(train_chromvec)
		feature_dim_motif = 769
		# feature_dim_transform = [50,25]
		tol = self.tol
		L = self.flanking
		run_id = self.run_id
		flanking1 = self.flanking1
		vec1 = []

		context_size = 2*L+1
		# config = dict(fc1_output_dim=5,fc2_output_dim=0,n_epochs=10)
		# config = dict(feature_dim=x_train.shape[-1],output_dim=32,fc1_output_dim=0,fc2_output_dim=0,n_epochs=100,batch_size=128)
		config = self.config
		# config['feature_dim'] = x_train.shape[-1]
		config['feature_dim'] = np.sum(feature_dim_transform)+21
		config['lr'] = self.lr
		config['activation'] = self.activation

		model = get_model2a1_attention(context_size, config)
		print('loading model...')
		model.load_weights(model_path)
		type_id2 = 0
		feature_dim = feature_dim_transform

		# self.load_local_serial(filename1)
		# local_serial, local_signal = self.load_local_serial(filename1)			
		for t_chrom in test_chromvec:

			prefix = '%s_chr1-chr5_chr%s-chr%s'%(self.cell_type,str(t_chrom),str(t_chrom))
			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim[0],feature_dim[1])
			# x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim, shuffle, 
			# 										sub_sample_ratio, type_id2, normalize)
			# np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
			#						train_sel_list,test_sel_list),allow_pickle=True)
			x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list = np.load(filename1,allow_pickle=True)

			x_test, y_test, vec_test, vec_test_local = sample_select2a(x_test1_trans, y_signal_test_ori, test_sel_list, tol, L)
			# x_test, y_test_temp, vec_test, vec_test_local = sample_select2a(x_test1, y_test_temp1, test_sel_list1, tol, L)
			print(x_test.shape,y_test.shape)
			y_predicted_test = model.predict(x_test)

			layer_name = 'attention1'
			intermediate_layer = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
			feature1, attention = intermediate_layer.predict(x_test)
			print('attention1',attention.shape)
			dict1 = dict()
			dict1['attention1'] = attention
			dict1['vec1'] = vec_test
			dict1['vec1_local'] = vec_test_local
			np.save('attention1_%s.npy'%(output_filename),dict1,allow_pickle=True)

			# y_predicted_test_ori = read_predict(y_predicted_test, vec_test_local, [], flanking1)
			# y_predicted_test = y_predicted_test_ori[id2]
			# print('predict', y_predicted_test_ori.shape, y_predicted_test.shape)
			y_predicted_test = read_predict(y_predicted_test, vec_test_local, [], flanking1)
			
			y_test = np.ravel(y_test[:,L])

			temp1 = score_2a(y_test, y_predicted_test)
			vec1.append(temp1)
			print(temp1)

		vec1 = np.asarray(vec1)
		np.savetxt('test_3.txt',vec1,fmt='%.7f',delimiter='\t')

		return vec1

	def load_features_transform_predict_2(self,model_path,test_chromvec, feature_dim_transform):

		# x_train_list1, train_sel_list1 = self.load_features_ori(train_chromvec)
		feature_dim_motif = 769
		# feature_dim_transform = [50,25]
		tol = self.tol
		L = self.flanking
		run_id = self.run_id
		flanking1 = self.flanking1
		vec1 = []

		context_size = 2*L+1
		# config = dict(fc1_output_dim=5,fc2_output_dim=0,n_epochs=10)
		# config = dict(feature_dim=x_train.shape[-1],output_dim=32,fc1_output_dim=0,fc2_output_dim=0,n_epochs=100,batch_size=128)
		config = self.config
		# config['feature_dim'] = x_train.shape[-1]
		config['feature_dim'] = np.sum(feature_dim_transform)+21
		config['lr'] = self.lr
		config['activation'] = self.activation

		model = get_model2a1_attention(context_size, config)
		print('loading model...')
		model.load_weights(model_path)
		type_id2 = 0
		feature_dim = feature_dim_transform

		# self.load_local_serial(filename1)
		# local_serial, local_signal = self.load_local_serial(filename1)			
		for t_chrom in test_chromvec:
			prefix = 'chr1-chr5_chr%s-chr%s'%(str(t_chrom),str(t_chrom))
			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim[0],feature_dim[1])

			# x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim, shuffle, 
			# 										sub_sample_ratio, type_id2, normalize)
			# np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
			#						train_sel_list,test_sel_list),allow_pickle=True)
			x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list = np.load(filename1,allow_pickle=True)

			x_test, y_test_temp, vec_test, vec_test_local = sample_select2a(x_test1_trans, y_signal_test_ori, test_sel_list, tol, L)
			# x_test, y_test_temp, vec_test, vec_test_local = sample_select2a(x_test1, y_test_temp1, test_sel_list1, tol, L)
			y_predicted_test = model.predict(x_test)

			y_predicted_test_ori = read_predict(y_predicted_test, vec_test_local, [], flanking1)

			t_chrom1 = 'chr%s'%(t_chrom)
			id1 = np.where(self.chrom==t_chrom1)[0]
			t_serial = self.serial[id1]
			y_signal_test = self.signal[id1]

			id1_1 = np.where(self.chrom_ori==t_chrom1)[0]
			id2 = mapping_Idx(self.serial_ori[id1_1],t_serial)		

			y_predicted_test = y_predicted_test_ori[id2]
			print('predict', x_test.shape, y_predicted_test_ori.shape, y_predicted_test.shape)
			
			# y_signal_train1, y_signal_test = self.load_signal(train_sel_idx, test_sel_idx)
			# y_signal_train1_ori = signal_normalize(y_signal_train1,[0,1])
			y_test = signal_normalize(y_signal_test,[0,1])

			b1 = np.isnan(y_test)
			b2 = np.isnan(y_predicted_test)
			b1_1 = np.isinf(y_test)
			b2_1 = np.isinf(y_predicted_test)
			b1_2 = np.isfinite(y_test)
			b2_2 = np.isfinite(y_predicted_test)
			print(y_signal_test_ori)
			print(y_test)
			print(y_predicted_test)
			print(np.sum(b1),np.sum(b2),np.sum(b1_1),np.sum(b2_1),np.sum(b1_2),np.sum(b2_2))

			temp1 = score_2a(y_test, y_predicted_test)
			print(t_chrom)
			vec1.append(temp1)
			print(temp1)

		vec1 = np.asarray(vec1)
		np.savetxt('test_1.txt',vec1,fmt='%.7f',delimiter='\t')

		return vec1

	def load_features_transform_single(self,x_train_list1,test_chromvec,feature_dim_motif,feature_dim_transform=[50,50]):

		self.feature_dim_motif = feature_dim_motif
		# kmer_size1, kmer_size2 = self.kmer_size[0], self.kmer_size[1]

		# prefix = self.filename_load
		# save_filename_ori = '%s_%s_ori1.npy'%(prefix,self.cell_type)
		# self.save_filename_ori = save_filename_ori
		# x_train_list1, train_sel_list1 = self.load_features_ori(train_chromvec)
		x_test_list1, test_sel_list1 = self.load_features_ori(test_chromvec)
		x_train, x_test = np.asarray(x_train_list1), np.asarray(x_test_list1)
		num_train, num_test = x_train.shape[0], x_test.shape[0]
		x_ori1 = np.vstack((x_train,x_test))
		print('x_ori1', x_ori1.shape)

		dim1 = x_ori1.shape[1]
		# feature_dim_kmer = np.sum(self.feature_dim_kmer)
		feature_dim_kmer = 5120
		dim2 = dim1-feature_dim_kmer-self.feature_dim_motif
		feature_dim = feature_dim_transform[0]
		feature_dim1 = feature_dim_transform[1]
		
		print("feature_dim_kmer",feature_dim_kmer,dim2)
		feature_kmer_idx = np.asarray(range(dim2,dim2+feature_dim_kmer))
		x_ori = x_ori1[:,feature_kmer_idx]
		feature_motif_idx = np.asarray(range(dim2+feature_dim_kmer,dim1))
		x_ori_motif = x_ori1[:,feature_motif_idx]
		num_train, num_test = x_train.shape[0], x_test.shape[0]

		vec1 = ['PCA','Incremental PCA','Kernel PCA','Sparse PCA','SVD','GRP','SRP','MDS','ISOMAP','Minibatch','ICA','tSNE','LLE','Encoder']

		shuffle = 0
		sub_sample = -1
		type_id = 0
		start = time.time()
		x = self.dimension_reduction(x_ori,feature_dim,shuffle,sub_sample,type_id)
		stop = time.time()
		print("feature transform %s"%(vec1[type_id]),stop - start)

		# save transform model
		feature_dim, feature_dim1 = feature_dim_transform[0], feature_dim_transform[1]
		# filename1 = '%s_chr%s_%d_dimensions1.h5'%(self.filename_load,test_chromvec[0],feature_dim)
		filename1 = 'chr%s_%d_dimensions1.h5'%(test_chromvec[0],feature_dim)
		# pickle.dump(self.dimension_model, open(filename1, 'wb'))

		start = time.time()
		motif_type_id = 4 # Truncated SVD
		x_motif = self.dimension_reduction(x_ori_motif,feature_dim1,shuffle,sub_sample,motif_type_id)
		stop = time.time()
		print("feature transform motif %s"%(vec1[motif_type_id]),stop - start)
		x1 = np.hstack((x_ori1[:,0:dim2],x,x_motif))

		# save transfrom model
		filename1 = '%s_chr%s_%d_dimensions2.2.h5'%(self.filename_load,test_chromvec[0],feature_dim1)
		pickle.dump(self.dimension_model, open(filename1, 'wb'))

		# if normalize>=2:
		# 	sc = StandardScaler()
		# 	x1 = sc.fit_transform(x1)
		x_train1, x_test1 = x1[0:num_train], x1[num_train:num_train+num_test]
		print(x_train.shape,x_train1.shape,x_test.shape,x_test1.shape)

		print("feature transform")
			# filename1 = 'test_chr10_%d_1_5_3.npy'%(type_id2)
		# prefix = self.filename_load
		# 1: PCA; 2: SVD for motif
		filename1 = 'chr%s_%d_%d_%d.npy'%(test_chromvec[0],type_id,feature_dim[0],feature_dim[1])

		test_sel_list1 = np.asarray(test_sel_list1)
		np.save(filename1,(x_test1,test_sel_list1),allow_pickle=True)

		return x_test1, test_sel_list1

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
				
			# if feature_type>-1:
			# 	x_train1_ori, x_test_ori = x_train1_ori[:,temp1], x_test_ori[:,temp1]

			if feature_type>-1:
				x_train1_ori = x_train1_ori[:,temp1]

			self.x_train1_ori = x_train1_ori
			# self.y_signal_train1 = y_signal_train1
			# self.x_test_ori = x_test_ori
			# self.y_signal_test = y_signal_test
			self.feature_dim_kmer = feature_dim_kmer
			self.train_sel_list = train_sel_list
			# self.test_sel_list = test_sel_list
			feature_dim_kmer_1 = np.sum(feature_dim_kmer)

			# prefix = self.filename_load
			# save_filename_ori = '%s_%s_ori1.npy'%(prefix,self.cell_type)

			filename1 = '%s_chr%s-chr%s_chr%s-chr%s'%(self.cell_type, t_chrom, t_chrom, self.test_chromvec[0], self.test_chromvec[-1])
			save_filename_ori = '%s_ori1.npy'%(filename1)
			print(save_filename_ori)
			
			d1 = (x_train1_ori,train_sel_list,train_sel_idx,feature_dim_kmer,self.feature_dim_motif)
			pickle.dump(d1, open(save_filename_ori, 'wb'), protocol=4)

		return True
	
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
		# x_train_sub = sc.fit_transform(x_ori[0:num_train,:])
		# x_test_sub = sc.transform(x_ori[num_train+num_test,:])
		# x_train_sub = sc.fit_transform(x_ori[0:num_train,:])
		# x_test_sub = sc.transform(x_ori[num_train+num_test,:])
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

	def compare_single(self,x_train,y_train,x_valid,y_valid,x_test,y_test,type_id=0):

		# x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.2, random_state=42)
		if type_id==0:
			self.model_single = LinearRegression().fit(x_train, y_train)
		else:
			print("xgboost regression")
			self.model_single = xgboost.XGBRegressor(colsample_bytree=1,
				 gamma=0,    
				 n_jobs=20,             
				 learning_rate=0.1,
				 max_depth=5,
				 min_child_weight=1,
				 n_estimators=500,                                                                    
				 reg_alpha=0,
				 reg_lambda=1,
				 objective='reg:squarederror',
				 subsample=1,
				 seed=42)
			print("fitting model...")
			self.model_single.fit(x_train, y_train)
		
		y_predicted_valid = self.model_single.predict(x_valid)
		y_predicted_test = self.model_single.predict(x_test)

		vec1 = []
		score1 = score_2a(y_valid,y_predicted_valid)
		score2 = score_2a(y_test,y_predicted_test)
		vec1.append(score1)
		vec1.append(score2)

		return vec1

	# load features: kmer, gc, phyloP score
	def load_phyloP_score(self,filename,query_serial,feature_idx):

		# load phyloP scores
		# filename3 = '%s/phyloP_chr%s.txt'%(file_path,chrom_id)
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

	def load_gc_feature(self,signal_ori,serial_ori,query_serial,feature_idx):

		# filename = '%s/training2_gc.txt'%(file_path,species_name)
		# file2 = pd.read_csv(filename,sep='\t')
		# gc_signal = np.asarray(file2)

		trans_id1a = mapping_Idx(serial_ori,query_serial)	# mapped index
		t_gc = signal_ori[trans_id1a]
		t_gc = t_gc[:,feature_idx]

		return t_gc

	def load_kmer_feature(self,filename,kmer_size,sel_idx):

		# filename2 = '%s/Kmer%d/training_kmer_%s.npy'%(file_path,kmer_size,chrom_id)
		file1 = np.load(filename)
		t_signal_ori1 = np.asarray(file1)
		t_signal1 = t_signal_ori1[sel_idx]
		# feature_dim_kmer1 = t_signal_ori1.shape[1]

		# temp1 = serial[id1]
		# n1 = len(temp1)
		# temp2 = np.vstack(([int(chrom_id)]*n1,temp1)).T
		# train_sel_list.extend(temp2)	# chrom_id, serial

		return t_signal1

	def load_feature_1(self,signal_ori,serial_ori,query_serial,feature_idx=-1):

		# filename2 = '%s/Kmer%d/training_kmer_%s.npy'%(file_path,kmer_size,chrom_id)
		# file1 = np.load(filename)
		# t_signal_ori1 = np.asarray(file1)
		trans_id1a = mapping_Idx(serial_ori,query_serial)	# mapped index
		signal1 = signal_ori[trans_id1a]
		if feature_idx!=-1:
			signal1 = signal_ori[:,feature_idx]

		# temp1 = serial[id1]
		# n1 = len(temp1)
		# temp2 = np.vstack(([int(chrom_id)]*n1,temp1)).T
		# train_sel_list.extend(temp2)	# chrom_id, serial

		return signal1

	def load_feature_2(self,signal_ori,sel_idx,feature_idx=-1):

		signal1 = signal_ori[sel_idx]
		if feature_idx!=-1:
			signal1 = signal1[:,feature_idx]

		return signal1

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
		# train_sel_list.extend(temp2)	# chrom_id, serial

		# load kmer features
		file_path = self.file_path
		kmer_size = kmer_size1
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		filename2 = '%s/Kmer%d/training_kmer_%s.npy'%(file_path,kmer_size,chrom_id)
		t_signal1 = self.load_kmer_feature(filename2,kmer_size,trans_id1)
		feature_dim_kmer1 = t_signal1.shape[1]

		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
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
		# train_sel_list.extend(temp2)	# chrom_id, serial

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
		# self.y_signal_train1 = y_signal_train1
		self.x_test_ori = x_test_ori
		# self.y_signal_test = y_signal_test
		self.feature_dim_kmer = feature_dim_kmer
		self.train_sel_list = train_sel_list
		self.test_sel_list = test_sel_list
		feature_dim_kmer_1 = np.sum(feature_dim_kmer)

		# print(x_train1_ori.shape, y_signal_train1.shape, x_test_ori.shape, y_signal_test.shape)
		print(x_train1_ori.shape, x_test_ori.shape)

		# return x_train1_ori, x_test_ori, y_signal_train1, y_signal_test, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer_1

		return x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer_1

	# load signals
	def load_signal(self,train_sel_idx,test_sel_idx):

		y_signal_train1 = self.signal[train_sel_idx]
		y_signal_test = self.signal[test_sel_idx]

		return y_signal_train1, y_signal_test
	
	# the mapped indices of selected regions
	def load_map_idx(self, ref_filename, filename1):

		# path1 = '/volume01/yy3/seq_data/dl/replication_timing'
		# filename3 = '%s/estimate_rt/estimate_rt_%s.txt'%(path1,species_name)
		# filename3a = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path1,species_name)
		temp1 = pd.read_csv(ref_filename,sep='\t')
		temp2 = pd.read_csv(filename1,sep='\t')
		colname1, colname2 = list(temp1), list(temp2)
		chrom2, start2, stop2, serial2 = temp1[colname1[0]], temp1[colname1[1]], temp2[colname2[2]], temp2[colname2[3]]
		chrom2, start2, stop2, serial2 = temp2[colname2[0]], temp2[colname2[1]], temp2[colname2[2]], temp2[colname2[3]]

		map_idx = mapping_Idx(serial1,serial2)

		return serial1, serial2, map_idx

	def load_predicted_signal(self,filename1,output_filename,x_min=0,x_max=1):

		# filename1 = 'feature_transform_0_50_600.npy'
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
		# valid2 = pd.DataFrame(columns=fields)
		# valid2['chrom'], valid2['start'], valid2['stop'], valid2['signal'] = self.chrom[id1], self.start[id1], self.stop[id1], y_predicted_valid

		test1 = pd.DataFrame(columns=fields)
		test1['chrom'],test1['start'], test1['stop'], test1['signal'], test1['predicted'] = self.chrom[id2], self.start[id2], self.stop[id2], y_test, y_predicted_test
		# test2 = pd.DataFrame(columns=fields)
		# test2['chrom'],test2['start'], test2['stop'], test2['signal'] = self.chrom[id2], self.start[id2], self.stop[id2], y_predicted_test

		valid1.to_csv('valid_%s.txt'%(output_filename),header=True,index=False,sep='\t')
		test1.to_csv('test_%s.txt'%(output_filename),header=True,index=False,sep='\t')

		return True

	def load_ref_serial(self, ref_filename, header=None):

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
	def load_local_serial(self, filename1, header=None):

		if header==None:
			file2 = pd.read_csv(filename1,header=header,sep='\t')
		else:
			file2 = pd.read_csv(filename1,sep='\t')	
		colnames = list(file2)
		col1, col2, col3, col_serial = colnames[0], colnames[1], colnames[2], colnames[3]
		self.chrom, self.start, self.stop, self.serial = np.asarray(file2[col1]), np.asarray(file2[col2]), np.asarray(file2[col3]), np.asarray(file2[col_serial])

		# label = np.asarray(file1['label'])
		# group_label = np.asarray(file1['group_label'])
		if len(colnames)>=5:
			col_signal = colnames[4]
			self.signal = np.asarray(file2[col_signal])
		else:
			self.signal = np.ones(file2.shape[0])
		# print(self.signal.shape)
		print('load local serial', self.serial.shape, self.signal.shape)

		return self.serial, self.signal

	def set_species_id(self,species_id,resolution):

		self.species_id = species_id
		self.resolution = resolution
		print('species_id, resolution', species_id, resolution)

	def load_features_transform(self,file_path,file_prefix_kmer,prefix,feature_dim_motif,feature_dim_transform=[50,50]):

		self.feature_dim_motif = feature_dim_motif
		kmer_size1, kmer_size2 = self.kmer_size[0], self.kmer_size[1]

		prefix = self.filename_load
		save_filename_ori = '%s_%s_ori1.npy'%(prefix,self.cell_type)
		self.save_filename_ori = save_filename_ori

		if os.path.exists(save_filename_ori)==True:
			print("loading %s"%(save_filename_ori))
			# x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer, feature_dim_motif = np.load(save_filename_ori,allow_pickle=True)
			x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer, feature_dim_motif = pickle.load(open(save_filename_ori, 'rb'))
			self.feature_dim_motif = feature_dim_motif
		else:
			print("loading...")
			x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer = self.load_features(file_path,file_prefix_kmer,kmer_size1,kmer_size2)
			# np.save(self.save_filename_ori,(x_train1_ori,x_test_ori,train_sel_list,test_sel_list,train_sel_idx,test_sel_idx,feature_dim_kmer, self.feature_dim_motif),allow_pickle=True)
			d1 = (x_train1_ori,x_test_ori,train_sel_list,test_sel_list,train_sel_idx,test_sel_idx,feature_dim_kmer, self.feature_dim_motif)
			pickle.dump(d1, open(save_filename_ori, 'wb'), protocol=4)

		y_signal_train1, y_signal_test = self.load_signal(train_sel_idx, test_sel_idx)
		y_signal_train1_ori = signal_normalize(y_signal_train1,[0,1])
		y_signal_test_ori = signal_normalize(y_signal_test,[0,1])

		print(x_train1_ori.shape, y_signal_train1.shape, x_test_ori.shape, y_signal_test.shape)

		feature_dim = feature_dim_transform
		sub_sample_ratio = 1
		shuffle = 0
		normalize = 0
		vec2 = dict()
		m_corr, m_explain = [0,0], [0,0]
		# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
		train_sel_list, test_sel_list = np.asarray(train_sel_list), np.asarray(test_sel_list)
		self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list
		tol = self.tol
		L = self.flanking
		print(train_sel_list[0:5])
		print(test_sel_list[0:5])
		print(feature_dim_kmer, feature_dim_transform)
		self.feature_dim_kmer = feature_dim_kmer

		for type_id2 in self.t_list:
			# np.save(filename1)
			print("feature transform")
			# filename1 = 'test_chr10_%d_1_5_3.npy'%(type_id2)
			prefix = self.filename_load
			# 1: PCA; 2: SVD for motif
			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim[0],feature_dim[1])

			x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim, shuffle, 
													sub_sample_ratio, type_id2, normalize)
			np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
									train_sel_list,test_sel_list),allow_pickle=True)

		return x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list

	def load_features_transform_1(self,file_path,file_prefix_kmer,prefix,feature_dim_motif,feature_dim_transform=[50,50]):

		self.feature_dim_motif = feature_dim_motif
		kmer_size1, kmer_size2 = self.kmer_size[0], self.kmer_size[1]

		prefix = self.filename_load
		save_filename_ori = '%s_%s_ori1.npy'%(prefix,self.cell_type)
		self.save_filename_ori = save_filename_ori

		if os.path.exists(save_filename_ori)==True:
			print("loading %s"%(save_filename_ori))
			# x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer, feature_dim_motif = np.load(save_filename_ori,allow_pickle=True)
			x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer, feature_dim_motif = pickle.load(open(save_filename_ori, 'rb'))
			self.feature_dim_motif = feature_dim_motif
		else:
			print("loading...")
			x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer = self.load_features(file_path,file_prefix_kmer,kmer_size1,kmer_size2)
			# np.save(self.save_filename_ori,(x_train1_ori,x_test_ori,train_sel_list,test_sel_list,train_sel_idx,test_sel_idx,feature_dim_kmer, self.feature_dim_motif),allow_pickle=True)
			d1 = (x_train1_ori,x_test_ori,train_sel_list,test_sel_list,train_sel_idx,test_sel_idx,feature_dim_kmer, self.feature_dim_motif)
			pickle.dump(d1, open(save_filename_ori, 'wb'), protocol=4)

		
		y_signal_train1, y_signal_test, serial1, serial2 = self.load_signal_1(self.train_chromvec, self.test_chromvec)
		id1 = mapping_Idx(train_sel_list[:,1],serial1)
		x_train1_ori = x_train1_ori[id1]

		id2 = mapping_Idx(test_sel_list[:,1],serial2)
		x_test_ori = x_test_ori[id1]

		#y_signal_train1, y_signal_test = self.load_signal(train_sel_idx, test_sel_idx)
		y_signal_train1_ori = signal_normalize(y_signal_train1,[0,1])
		y_signal_test_ori = signal_normalize(y_signal_test,[0,1])

		print(x_train1_ori.shape, y_signal_train1.shape, x_test_ori.shape, y_signal_test.shape)

		feature_dim = feature_dim_transform
		sub_sample_ratio = 1
		shuffle = 0
		normalize = 0
		vec2 = dict()
		m_corr, m_explain = [0,0], [0,0]
		# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
		train_sel_list, test_sel_list = np.asarray(train_sel_list), np.asarray(test_sel_list)
		self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list
		tol = self.tol
		L = self.flanking
		print(train_sel_list[0:5])
		print(test_sel_list[0:5])
		print(feature_dim_kmer, feature_dim_transform)
		self.feature_dim_kmer = feature_dim_kmer

		for type_id2 in self.t_list:
			# np.save(filename1)
			print("feature transform")
			# filename1 = 'test_chr10_%d_1_5_3.npy'%(type_id2)
			prefix = self.filename_load
			# 1: PCA; 2: SVD for motif
			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim[0],feature_dim[1])

			x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim, shuffle, 
													sub_sample_ratio, type_id2, normalize)
			np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
									train_sel_list,test_sel_list),allow_pickle=True)

		return x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list
	
	def set_generate(self,generate,filename=None):

		self.generate = generate
		if filename!=None:
			self.filename_load = str(filename)

		return True

	def set_featuredim_motif(self,feature_dim_motif):

		self.feature_dim_motif = feature_dim_motif

		return True

	# context feature
	def kmer_compare_single2a2_6_weighted1(self, feature_dim_transform):

		# serial_ori = self.load_ref_serial(ref_filename)
		# serial = self.load_local_serial(filename1)
		self.feature_dim_transform = feature_dim_transform
		# map_idx = mapping_Idx(serial_ori,serial)

		if self.generate == 1:
			x_train1_ori, x_test_ori, y_signal_train1, y_signal_test, train_sel_list, test_sel_list, _, _, feature_dim_kmer = self.load_features(kmer_size1,kmer_size2)
			y_signal_train1_ori = signal_normalize(y_signal_train1,[0,1])
			y_signal_test_ori = signal_normalize(y_signal_test,[0,1])
			train_sel_list, test_sel_list = np.asarray(train_sel_list), np.asarray(test_sel_list)
			self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list

		sub_sample_ratio = 1
		shuffle = 0
		normalize, flanking, attention, run_id = self.normalize, self.flanking, self.attention, self.run_id
		config = self.config
		vec2 = dict()
		m_corr, m_explain = [0,0], [0,0]
		# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
		tol = self.tol
		L = flanking
		
		for type_id2 in self.t_list:
			# np.save(filename1)
			print("feature transform")
			# filename1 = 'test_chr10_%d_1.npy'%(type_id2)
			# filename1 = self.filename_load
			prefix = self.filename_load
			print(self.filename_load)
			filename1 = '%s_%d_1.npy'%(prefix,type_id2)

			if self.generate==1:
				x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim_transform, shuffle, 
																			sub_sample_ratio, type_id2, normalize)
				np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
									train_sel_list,test_sel_list),allow_pickle=True)
			else:
				if os.path.exists(filename1)==True:
					print("loading data...")
					x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori,train_sel_list,test_sel_list = np.load(filename1,allow_pickle=True)
					self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list

					# x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori,train_sel_list,test_sel_list = x_train1_trans[0:5000], x_test1_trans[0:5000], y_signal_train1_ori[0:5000], y_signal_test_ori[0:5000],train_sel_list[0:5000],test_sel_list[0:5000]
				else:
					print(filename1)
					print("data not found!")
					return

			print(train_sel_list[0:5])
			print(test_sel_list[0:5])

			train_num = x_train1_trans.shape[0]
			id1 = range(0,train_num)
			num1 = int(train_num*0.8)
			num2 = int(train_num*0.9)
			idx_train = id1[0:num1]
			idx_valid = id1[num1:num2]
			idx_test = id1[num2:]
			print(x_train1_trans.shape, x_test1_trans.shape)

			print(type_id2)
			print("training...")
			dict1 = self.training(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,idx_train,idx_valid,idx_test,type_id2)

			vec2[type_id2] = dict1
			vec1 = dict1['vec1']
			temp1 = vec1[2]
			if temp1[1]>m_corr[1]:
				m_corr = [type_id2,temp1[1]]
			if temp1[2]>m_explain[1]:
				m_explain = [type_id2,temp1[2]]

		print(m_corr,m_explain)
		np.save('feature_transform_%d_%d_%d.npy'%(self.t_list[0],feature_dim_transform[0],run_id),vec2,allow_pickle=True)
		print(vec2)

		# y_proba = data1['yprob']
		# print(y_test.shape,y_proba.shape)
		# corr1 = pearsonr(y_test, np.ravel(y_proba))
		# print(corr1)
		self.output_vec2(vec2)

		return vec1,dict1

	def generate_index(self, train_sel_list, test_sel_list, ratio=0.8, type_id=0):

		if type_id==0:
			train_num = len(train_sel_list)
			id1 = range(0,train_num)
			num1 = int(train_num*ratio)
			num2 = int(train_num*0.9)
			idx_train = id1[0:num1]
			idx_valid = id1[num1:num2]
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
				num2 = int(train_num*0.9)
				idx_train.extend(id1[0:num1])
				idx_valid.extend(id1[num1:num2])
				# idx_test.extend(id1[num2:])

		return idx_train, idx_valid, idx_test

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
				# idx_test.extend(id1[num2:])

		return idx_train, idx_valid, idx_test

	def generate_index_2(self, x_train, train_sel_list, test_sel_list, chrom_vec1, ratio=0.9, type_id=0):

		idx_test = []
		train_sel_list1 = []
		test_sel_list1 = []
		chrom = train_sel_list[:,0]
		chrom_vec = np.unique(chrom)
		print(chrom_vec)
		idx_train, idx_valid, idx_test = [], [], []
		id_vec1 = []
		for chrom_id in chrom_vec1:
			id1 = np.where(chrom==chrom_id)[0]
			train_sel_list1.extend(train_sel_list[id1])
			id_vec1.extend(id1)

		id_vec1 = np.asarray(id_vec1)
		x_train = x_train[id_vec1]

		train_num = len(train_sel_list1)
		id1 = range(0,train_num)
		ratio = self.ratio
		num1 = int(train_num*ratio)
		idx_train = id1[0:num1]
		idx_valid = id1[num1:]

		return x_train, train_sel_list1, test_sel_list1, idx_train, idx_valid, idx_test

	# context feature
	def kmer_compare_single2a2_6_weighted2(self,file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform):

		# serial_ori = self.load_ref_serial(ref_filename)
		# serial = self.load_local_serial(filename1)
		self.feature_dim_transform = feature_dim_transform
		# map_idx = mapping_Idx(serial_ori,serial)

		sub_sample_ratio = 1
		shuffle = 0
		normalize, flanking, attention, run_id = self.normalize, self.flanking, self.attention, self.run_id
		config = self.config
		vec2 = dict()
		m_corr, m_explain = [0,0], [0,0]
		# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
		tol = self.tol
		L = flanking
		
		for type_id2 in self.t_list:
			# np.save(filename1)
			print("feature transform")
			# filename1 = 'test_chr10_%d_1.npy'%(type_id2)
			# filename1 = self.filename_load
			prefix = self.filename_load
			print(self.filename_load)
			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim_transform[0],feature_dim_transform[1])

			if self.generate==1:
				# x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim_transform, shuffle, 
				# 															sub_sample_ratio, type_id2, normalize)
				# np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
				# 					train_sel_list,test_sel_list),allow_pickle=True)
				x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list = self.load_features_transform(file_path,file_prefix_kmer,prefix,feature_dim_motif,feature_dim_transform)
			else:
				if os.path.exists(filename1)==True:
					print("loading data...")
					x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori, train_sel_list, test_sel_list = np.load(filename1,allow_pickle=True)
					self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list

					# x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori,train_sel_list,test_sel_list = x_train1_trans[0:5000], x_test1_trans[0:5000], y_signal_train1_ori[0:5000], y_signal_test_ori[0:5000],train_sel_list[0:5000],test_sel_list[0:5000]
				else:
					print(filename1)
					print("data not found!")
					return

			if self.feature_dim_motif<=0:
				x_train1_trans = x_train1_trans[:,0:-feature_dim_transform[1]]
				x_test1_trans = x_test1_trans[:,0:-feature_dim_transform[1]]

			print(train_sel_list[0:5])
			print(test_sel_list[0:5])

			print(x_train1_trans.shape, x_test1_trans.shape)

			idx_train, idx_valid, idx_test = self.generate_index(train_sel_list, test_sel_list)

			print(type_id2)
			print("training...")
			dict1 = self.training(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,idx_train,idx_valid,idx_test,type_id2)

			vec2[type_id2] = dict1
			vec1 = dict1['vec1']
			temp1 = vec1[2]
			if temp1[1]>m_corr[1]:
				m_corr = [type_id2,temp1[1]]
			if temp1[2]>m_explain[1]:
				m_explain = [type_id2,temp1[2]]

		print(m_corr,m_explain)

		vec2['valid'], vec2['test1'], vec2['test'] = train_sel_list[idx_valid], train_sel_list[idx_test], test_sel_list
		np.save('feature_transform_%d_%d_%d.npy'%(self.t_list[0],feature_dim_transform[0],run_id),vec2,allow_pickle=True)
		print(vec2)

		self.output_vec2(vec2,self.t_list)

		return vec1,dict1

	# context feature
	def kmer_compare_single2a2_6_weighted3(self,file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform):

		# serial_ori = self.load_ref_serial(ref_filename)
		# serial = self.load_local_serial(filename1)
		self.feature_dim_transform = feature_dim_transform
		# map_idx = mapping_Idx(serial_ori,serial)

		sub_sample_ratio = 1
		shuffle = 0
		normalize, flanking, attention, run_id = self.normalize, self.flanking, self.attention, self.run_id
		config = self.config
		vec2 = dict()
		m_corr, m_explain = [0,0], [0,0]
		# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
		tol = self.tol
		L = flanking
		
		for type_id2 in self.t_list:
			# np.save(filename1)
			print("feature transform")
			# filename1 = 'test_chr10_%d_1.npy'%(type_id2)
			# filename1 = self.filename_load
			prefix = self.filename_load
			print(self.filename_load)
			# filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim_transform[0],feature_dim_transform[1])
			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim_transform[0],feature_dim_transform[1])

			if self.generate==1:
				# x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim_transform, shuffle, 
				# 															sub_sample_ratio, type_id2, normalize)
				# np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
				# 					train_sel_list,test_sel_list),allow_pickle=True)
				x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list = self.load_features_transform(file_path,file_prefix_kmer,prefix,feature_dim_motif,feature_dim_transform)
			else:
				if os.path.exists(filename1)==True:
					print("loading data...")
					x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori, train_sel_list, test_sel_list = np.load(filename1,allow_pickle=True)
					self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list

					# x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori,train_sel_list,test_sel_list = x_train1_trans[0:5000], x_test1_trans[0:5000], y_signal_train1_ori[0:5000], y_signal_test_ori[0:5000],train_sel_list[0:5000],test_sel_list[0:5000]
				else:
					print(filename1)
					print("data not found!")
					return

			if self.feature_dim_motif<=0:
				x_train1_trans = x_train1_trans[:,0:-feature_dim_transform[1]]
				x_test1_trans = x_test1_trans[:,0:-feature_dim_transform[1]]
			elif self.feature_dim_motif==1:
				x_train1_trans = x_train1_trans[:,0:2]
				x_test1_trans = x_test1_trans[:,0:2]
			elif self.feature_dim_motif==2:
				x_train1_trans = x_train1_trans[:,0:21]
				x_test1_trans = x_test1_trans[:,0:21]
			elif self.feature_dim_motif==3:
				x_train1_trans = x_train1_trans[:,2:21]
				x_test1_trans = x_test1_trans[:,2:21]
			elif self.feature_dim_motif==4:
				x_train1_trans = x_train1_trans[:,21:-feature_dim_transform[1]]
				x_test1_trans = x_test1_trans[:,21:-feature_dim_transform[1]]
			elif self.feature_dim_motif==5:
				dim1 = x_train1_trans.shape[1]-feature_dim_transform[1]
				sel_idx = [0,1]+list(range(21,dim1))
				x_train1_trans = x_train1_trans[:,sel_idx]
				x_test1_trans = x_test1_trans[:,sel_idx]
			else:
				pass

			print(train_sel_list[0:5])
			print(test_sel_list[0:5])

			print(x_train1_trans.shape, x_test1_trans.shape)

			idx_train, idx_valid, idx_test = self.generate_index_1(train_sel_list, test_sel_list)

			print(type_id2)
			print("training...")

			dict1 = self.training_1(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,idx_train,idx_valid,idx_test,type_id2)

			vec2[type_id2] = dict1
			vec1 = dict1['vec1']
			temp1 = vec1[1]
			if temp1[1]>m_corr[1]:
				m_corr = [type_id2,temp1[1]]
			if temp1[2]>m_explain[1]:
				m_explain = [type_id2,temp1[2]]

		print(m_corr,m_explain)

		vec2['valid'], vec2['test'] = train_sel_list[idx_valid], test_sel_list
		np.save('feature_transform_%d_%d_%d.npy'%(self.t_list[0],feature_dim_transform[0],run_id),vec2,allow_pickle=True)
		print(vec2)

		self.output_vec2(vec2,self.t_list)

		return vec1,dict1

	# context feature
	def kmer_compare_single2a2_6_weighted3_single(self,file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform):

		# serial_ori = self.load_ref_serial(ref_filename)
		# serial = self.load_local_serial(filename1)
		self.feature_dim_transform = feature_dim_transform
		# map_idx = mapping_Idx(serial_ori,serial)

		sub_sample_ratio = 1
		shuffle = 0
		normalize, flanking, attention, run_id = self.normalize, self.flanking, self.attention, self.run_id
		config = self.config
		vec2 = dict()
		m_corr, m_explain = [0,0], [0,0]
		# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
		tol = self.tol
		L = flanking
		
		type_id2 = 0
		# for type_id2 in self.t_list:
		for t_chrom in self.test_chromvec:
			# np.save(filename1)
			print("feature transform")
			# filename1 = 'test_chr10_%d_1.npy'%(type_id2)
			# filename1 = self.filename_load
			prefix = self.filename_load
			print(self.filename_load)
			# filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim_transform[0],feature_dim_transform[1])
			# prefix = 'chr1-chr5_chr%s-chr%s'%(t_chrom,t_chrom)
			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim_transform[0],feature_dim_transform[1])

			if self.generate==1:
				# x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim_transform, shuffle, 
				# 															sub_sample_ratio, type_id2, normalize)
				# np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
				# 					train_sel_list,test_sel_list),allow_pickle=True)
				x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list = self.load_features_transform(file_path,file_prefix_kmer,prefix,feature_dim_motif,feature_dim_transform)
			else:
				if os.path.exists(filename1)==True:
					print("loading data...")
					x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori, train_sel_list, test_sel_list = np.load(filename1,allow_pickle=True)
					self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list

					# x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori,train_sel_list,test_sel_list = x_train1_trans[0:5000], x_test1_trans[0:5000], y_signal_train1_ori[0:5000], y_signal_test_ori[0:5000],train_sel_list[0:5000],test_sel_list[0:5000]
				else:
					print(filename1)
					print("data not found!")
					return

			if self.feature_dim_motif<=0:
				x_train1_trans = x_train1_trans[:,0:-feature_dim_transform[1]]
				x_test1_trans = x_test1_trans[:,0:-feature_dim_transform[1]]
			elif self.feature_dim_motif==1:
				x_train1_trans = x_train1_trans[:,0:2]
				x_test1_trans = x_test1_trans[:,0:2]
			elif self.feature_dim_motif==2:
				x_train1_trans = x_train1_trans[:,0:21]
				x_test1_trans = x_test1_trans[:,0:21]
			elif self.feature_dim_motif==3:
				x_train1_trans = x_train1_trans[:,2:21]
				x_test1_trans = x_test1_trans[:,2:21]
			elif self.feature_dim_motif==4:
				x_train1_trans = x_train1_trans[:,21:-feature_dim_transform[1]]
				x_test1_trans = x_test1_trans[:,21:-feature_dim_transform[1]]
			elif self.feature_dim_motif==5:
				dim1 = x_train1_trans.shape[1]-feature_dim_transform[1]
				sel_idx = [0,1]+list(range(21,dim1))
				x_train1_trans = x_train1_trans[:,sel_idx]
				x_test1_trans = x_test1_trans[:,sel_idx]
			else:
				pass

			print(train_sel_list[0:5])
			print(test_sel_list[0:5])

			print(x_train1_trans.shape, x_test1_trans.shape)

			idx_train, idx_valid, idx_test = self.generate_index_1(train_sel_list, test_sel_list)

			print(type_id2)
			print("training...")
			# dict1 = self.training_1(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,idx_train,idx_valid,idx_test,type_id2)

			x_train, x_valid = x_train1_trans[idx_train], x_train1_trans[idx_valid]
			x_test = x_test1_trans
			y_train, y_valid = y_signal_train1_ori[idx_train], y_signal_train1_ori[idx_valid] 
			y_test = y_signal_test_ori

			type_id3 = 1
			dict1 = dict()
			vec1 = self.compare_single(x_train,y_train,x_valid,y_valid,x_test,y_test,type_id3)
			dict1['vec1'] = vec1
			# vec2[type_id2] = dict1
			vec2[t_chrom] = dict1
			print(vec1)

		# self.output_vec2(vec2,self.t_list)
		self.output_vec2(vec2,self.test_chromvec)

		return vec1

	def set_chromvec_training(self,chrom_vec,run_id=-1,ratio=0.9):

		self.chromvec_sel = chrom_vec
		if run_id>=0:
			self.run_id = run_id
		self.ratio = ratio

	def load_features_transform_3(self,file_path,file_prefix_kmer,prefix,feature_dim_motif,feature_dim_transform=[50,50]):

		# train_sel_idx, test_sel_idx = [], []
		# train_sel_list, test_sel_list = [], []
		# data_vec = []
		# x_train1, x_test = [], []
		# if type_id==0 or type_id==2:
		# 	ratio = 1
		# else:
		# 	ratio = 0.5
		self.feature_dim_motif = feature_dim_motif
		kmer_size1, kmer_size2 = self.kmer_size[0], self.kmer_size[1]

		prefix = self.filename_load
		# save_filename_ori = '%s_%s_ori1.npy'%(prefix,self.cell_type)
		save_filename_ori = '%s_ori1.npy'%(prefix)
		self.save_filename_ori = save_filename_ori
		print(save_filename_ori)

		if os.path.exists(save_filename_ori)==True:
			print("loading %s"%(save_filename_ori))
			# x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer, feature_dim_motif = np.load(save_filename_ori,allow_pickle=True)
			x_train1_ori, x_test_ori, train_sel_list_ori, test_sel_list_ori, train_sel_idx_ori, test_sel_idx_ori, feature_dim_kmer, feature_dim_motif = pickle.load(open(save_filename_ori, 'rb'))
			self.feature_dim_motif = feature_dim_motif
		else:
			print("loading...")
			x_train1_ori, x_test_ori, train_sel_list, test_sel_list, train_sel_idx, test_sel_idx, feature_dim_kmer = self.load_features(file_path,file_prefix_kmer,kmer_size1,kmer_size2)
			# np.save(self.save_filename_ori,(x_train1_ori,x_test_ori,train_sel_list,test_sel_list,train_sel_idx,test_sel_idx,feature_dim_kmer, self.feature_dim_motif),allow_pickle=True)
			d1 = (x_train1_ori,x_test_ori,train_sel_list,test_sel_list,train_sel_idx,test_sel_idx,feature_dim_kmer, self.feature_dim_motif)
			pickle.dump(d1, open(save_filename_ori, 'wb'), protocol=4)
		
		y_signal_train1_ori, y_signal_test_ori, serial1, serial2, train_sel_list, test_sel_list = self.load_signal_3(self.chrom, self.train_chromvec, self.test_chromvec)
		
		# train_sel_list based on the ref serial
		train_sel_list_ori, test_sel_list_ori = np.asarray(train_sel_list_ori), np.asarray(test_sel_list_ori)
		id1_train = mapping_Idx(train_sel_list_ori[:,1],serial1)
		x_train1_ori = x_train1_ori[id1_train]

		id1_test = mapping_Idx(test_sel_list_ori[:,1],serial2)
		x_test_ori = x_test_ori[id1_test]

		train_sel_list, test_sel_list = np.asarray(train_sel_list), np.asarray(test_sel_list)
		self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list

		# y_signal_train1, y_signal_test = self.load_signal(train_sel_idx, test_sel_idx)
		# y_signal_train1_ori = signal_normalize(y_signal_train1,[0,1])
		# y_signal_test_ori = signal_normalize(y_signal_test,[0,1])

		print(x_train1_ori.shape, y_signal_train1_ori.shape, x_test_ori.shape, y_signal_test_ori.shape)

		feature_dim = feature_dim_transform
		sub_sample_ratio = 1
		shuffle = 0
		normalize = 0
		vec2 = dict()
		m_corr, m_explain = [0,0], [0,0]
		# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
		tol = self.tol
		L = self.flanking
		print(train_sel_list[0:5])
		print(test_sel_list[0:5])
		print(feature_dim_kmer, feature_dim_transform)
		self.feature_dim_kmer = feature_dim_kmer

		for type_id2 in self.t_list:
			# np.save(filename1)
			print("feature transform")
			# filename1 = 'test_chr10_%d_1_5_3.npy'%(type_id2)
			prefix = self.filename_load
			# 1: PCA; 2: SVD for motif
			# filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim[0],feature_dim[1])

			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim[0],feature_dim[1])
			print(filename1)

			if os.path.exists(filename1)==True:
				x_train1_trans,x_test1_trans,y_signal_train1_temp,y_signal_test_temp,train_sel_list_ori,test_sel_list_ori = np.load(filename1,allow_pickle=True)
				x_train1_trans = x_train1_trans[id1_train]
				x_test1_trans = x_test1_trans[id1_test]
			else:
				x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim, shuffle, 
													sub_sample_ratio, type_id2, normalize)
				np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
									train_sel_list,test_sel_list),allow_pickle=True)

		print(x_train1_trans.shape,x_test1_trans.shape,len(self.train_sel_list),len(self.test_sel_list))
		
		return x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list

	# context feature
	def kmer_compare_single2a2_6_weighted5(self,file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform):

		# serial_ori = self.load_ref_serial(ref_filename)
		# serial = self.load_local_serial(filename1)
		self.feature_dim_transform = feature_dim_transform
		# map_idx = mapping_Idx(serial_ori,serial)

		sub_sample_ratio = 1
		shuffle = 0
		normalize, flanking, attention, run_id = self.normalize, self.flanking, self.attention, self.run_id
		config = self.config
		vec2 = dict()
		m_corr, m_explain = [0,0], [0,0]
		# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
		tol = self.tol
		L = flanking
		
		for type_id2 in self.t_list:
			# np.save(filename1)
			print("feature transform")
			# filename1 = 'test_chr10_%d_1.npy'%(type_id2)
			# filename1 = self.filename_load
			prefix = self.filename_load
			print(self.filename_load)
			filename1 = '%s_%d_%d_%d_1.npy'%(prefix,type_id2,feature_dim_transform[0],feature_dim_transform[1])

			self.generate = 1
			
			if self.generate==1:
				# x_train1_trans, x_test1_trans = self.feature_transform(x_train1_ori, x_test_ori, feature_dim_transform, shuffle, 
				# 															sub_sample_ratio, type_id2, normalize)
				# np.save(filename1,(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,
				# 					train_sel_list,test_sel_list),allow_pickle=True)
				x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,train_sel_list,test_sel_list = self.load_features_transform_3(file_path,file_prefix_kmer,prefix,feature_dim_motif,feature_dim_transform)
			else:
				if os.path.exists(filename1)==True:
					print("loading data...")
					x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori, train_sel_list, test_sel_list = np.load(filename1,allow_pickle=True)
					self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list

					# x_train1_trans, x_test1_trans, y_signal_train1_ori, y_signal_test_ori,train_sel_list,test_sel_list = x_train1_trans[0:5000], x_test1_trans[0:5000], y_signal_train1_ori[0:5000], y_signal_test_ori[0:5000],train_sel_list[0:5000],test_sel_list[0:5000]
				else:
					print(filename1)
					print("data not found!")
					return

			if self.feature_dim_motif<=0:
				x_train1_trans = x_train1_trans[:,0:-feature_dim_transform[1]]
				x_test1_trans = x_test1_trans[:,0:-feature_dim_transform[1]]

			# print(train_sel_list[0:5])
			# print(test_sel_list[0:5])

			# train_num = x_train1_trans.shape[0]
			# id1 = range(0,train_num)
			# num1 = int(train_num*0.8)
			# num2 = int(train_num*0.9)
			# idx_train = id1[0:num1]
			# idx_valid = id1[num1:num2]
			# idx_test = id1[num2:]

			print('signal')
			print(x_train1_trans.shape, x_test1_trans.shape)
			print(y_signal_train1_ori.shape, y_signal_test_ori.shape)

			# idx_train, idx_valid, idx_test = self.generate_index_1(train_sel_list, test_sel_list)
			chrom_vec1 = self.chromvec_sel
			ratio = 0.95
			if chrom_vec1 != self.train_chromvec:
				x_train1_trans, train_sel_list, test_sel_list, idx_train, idx_valid, idx_test = self.generate_index_2(x_train1_trans, train_sel_list, test_sel_list, chrom_vec1, ratio)
			else:
				idx_train, idx_valid, idx_test = self.generate_index_1(train_sel_list, test_sel_list, ratio)

			self.train_sel_list, self.test_sel_list = train_sel_list, test_sel_list
			print(train_sel_list[0:5])
			print(test_sel_list[0:5])

			print(type_id2)
			print("training...")
			dict1 = self.training_3(x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,idx_train,idx_valid,idx_test,type_id2)

			vec2[type_id2] = dict1
			vec1 = dict1['vec1']
			temp1 = vec1[-1]
			if temp1[1]>m_corr[1]:
				m_corr = [type_id2,temp1[1]]
			if temp1[3]>m_explain[1]:
				m_explain = [type_id2,temp1[3]]

		print(m_corr,m_explain)

		vec2['valid'], vec2['test'] = train_sel_list[idx_valid], test_sel_list
		np.save('feature_transform_%d_%d_%d.npy'%(self.t_list[0],feature_dim_transform[0],run_id),vec2,allow_pickle=True)
		print(vec2)

		# y_proba = data1['yprob']
		# print(y_test.shape,y_proba.shape)
		# corr1 = pearsonr(y_test, np.ravel(y_proba))
		# print(corr1)
		self.output_vec2(vec2,self.t_list)

		return vec1,dict1

	def output_vec2(self,vec2,tlist):

		# type_idvec = list(vec2.keys())
		temp1 = []
		for type_id2 in tlist:
			dict1 = vec2[type_id2]
			vec1 = dict1['vec1']
			temp1.extend(vec1)

		temp1 = np.asarray(temp1)
		
		filename1 = 'test_vec2_%d.txt'%(self.run_id)
		np.savetxt(filename1,temp1,fmt='%.7f',delimiter='\t')

		return True

	def training(self,x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,idx_train,idx_valid,idx_test,type_id2):
		
		train_sel_list, test_sel_list = self.train_sel_list, self.test_sel_list
		tol = self.tol
		L = self.flanking
		run_id = self.run_id
			
		x_test, y_test, vec_test, vec_test_local = sample_select2a(x_test1_trans, y_signal_test_ori, test_sel_list, tol, L)
		x_train, y_train, vec_train, vec_train_local = sample_select2a(x_train1_trans[idx_train], y_signal_train1_ori[idx_train], train_sel_list[idx_train], tol, L)
		x_valid, y_valid, vec_valid, vec_valid_local = sample_select2a(x_train1_trans[idx_valid], y_signal_train1_ori[idx_valid], train_sel_list[idx_valid], tol, L)
		x_test_1, y_test_1, vec_test1, vec_test1_local = sample_select2a(x_train1_trans[idx_test], y_signal_train1_ori[idx_test], train_sel_list[idx_test], tol, L)

		print(x_train.shape,x_valid.shape)

		context_size = x_train.shape[1]
		# config = dict(fc1_output_dim=5,fc2_output_dim=0,n_epochs=10)
		# config = dict(feature_dim=x_train.shape[-1],output_dim=32,fc1_output_dim=0,fc2_output_dim=0,n_epochs=100,batch_size=128)
		config = self.config
		config['feature_dim'] = x_train.shape[-1]
		config['lr'] = self.lr
		config['activation'] = self.activation
		BATCH_SIZE = config['batch_size']
		n_epochs = config['n_epochs']
		# earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=self.step, verbose=0, mode='auto')
		earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=0, mode='auto')
		MODEL_PATH = 'test_%d'%(run_id)
		checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
		# roc_cb = roc_callback([x_test,t_test],y_test)
		# model = get_model2(x_mtx.shape[-2])
		# if attention==1:
		# 	# model = get_model2a1_attention_sequential(context_size,config)
		# 	if self.predict_context==1:
		# 		self.model = get_model2a1_attention(context_size,config)
		# else:
		# 	# model = get_model2a1(context_size,config)
		# 	self.model = get_model2a1_sequential(context_size,config)

		attention = self.attention
		print(self.predict_context, self.attention)
		if self.predict_context==1:
			if attention==1:
				print('get_model2a1_attention')
				model = get_model2a1_attention(context_size,config)		# context self-attention model and prediction per position
			else:
				print('get_model2a1_sequential')
				model = get_model2a1_sequential(context_size,config)	# context without attention and prediction per position
		else:
			if attention==1:
				print('get_model2a1_attention_1')
				model = get_model2a1_attention_1(context_size,config)	# context with attention from intermedicate layer
			elif attention==2:
				print('get_model2a1_attention_2')
				model = get_model2a1_attention_2(context_size,config)	# context with attention from input
			else:
				print('get_model2a_sequential')
				model = get_model2a_sequential(context_size,config)		# context without attention
			
		# model.fit(X_train,y_train,epochs = 100,batch_size = BATCH_SIZE,validation_data = [X_test,y_test],class_weight=build_classweight(y_train),callbacks=[earlystop,checkpointer,roc_cb])
		# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_test,y_test],callbacks=[earlystop,checkpointer])
		model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
		# model.load_weights(MODEL_PATH)

		model_path = 'model_%d_%d_%d.h5'%(run_id,type_id2,context_size)
		model.save(model_path)

		vec1 = []
		# y_predicted_train = model.predict(x_train)
		self.model = model
		y_predicted_valid = model.predict(x_valid)
		y_predicted_test = model.predict(x_test)
		y_predicted_test_1 = model.predict(x_test_1)

		print(y_predicted_valid.shape, y_predicted_test_1.shape, y_predicted_test.shape)
		id1 = self.flanking
		# y_predicted_train, y_predicted_valid, y_predicted_test = y_predicted_train[:,id1], y_predicted_valid[:,id1], y_predicted_test[:,id1]
		# y_predicted_train, y_predicted_valid, y_predicted_test = np.ravel(y_predicted_train), np.ravel(y_predicted_valid), np.ravel(y_predicted_test)

		flanking1 = self.flanking1
		# y_predicted_train = read_predict(y_predicted_train, vec_train_local, [], flanking1)

		if self.predict_context==1:
			y_predicted_valid = read_predict(y_predicted_valid, vec_valid_local, [], flanking1)
			y_predicted_test_1 = read_predict(y_predicted_test_1, vec_test1_local, [], flanking1)
			y_predicted_test = read_predict(y_predicted_test, vec_test_local, [], flanking1)

			y_train, y_valid, y_test_1, y_test = y_train[:,id1], y_valid[:,id1], y_test_1[:,id1], y_test[:,id1]
		else:
			y_predicted_valid, y_predicted_test_1, y_predicted_test = np.ravel(y_predicted_valid), np.ravel(y_predicted_test_1), np.ravel(y_predicted_test)

		y_train, y_valid, y_test_1, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test_1), np.ravel(y_test)

		temp1 = score_2a(y_valid, y_predicted_valid)
		vec1.append(temp1)
		print(temp1)

		temp1 = score_2a(y_test_1, y_predicted_test_1)
		vec1.append(temp1)
		print(temp1)
		
		temp1 = score_2a(y_test, y_predicted_test)
		vec1.append(temp1)
		print(temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		dict1['y_valid'], dict1['y_test1'], dict1['y_test'] = y_valid, y_test_1, y_test
		dict1['y_predicted_valid'], dict1['y_predicted_test_1'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test_1, y_predicted_test

		return dict1

	def training_1(self,x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,idx_train,idx_valid,idx_test,type_id2):
		
		train_sel_list, test_sel_list = self.train_sel_list, self.test_sel_list
		tol = self.tol
		L = self.flanking
		run_id = self.run_id
			
		x_test, y_test, vec_test, vec_test_local = sample_select2a(x_test1_trans, y_signal_test_ori, test_sel_list, tol, L)
		x_train, y_train, vec_train, vec_train_local = sample_select2a(x_train1_trans[idx_train], y_signal_train1_ori[idx_train], train_sel_list[idx_train], tol, L)
		x_valid, y_valid, vec_valid, vec_valid_local = sample_select2a(x_train1_trans[idx_valid], y_signal_train1_ori[idx_valid], train_sel_list[idx_valid], tol, L)
		# x_test_1, y_test_1, vec_test1, vec_test1_local = sample_select2a(x_train1_trans[idx_test], y_signal_train1_ori[idx_test], train_sel_list[idx_test], tol, L)

		print(x_train.shape,x_valid.shape)

		context_size = x_train.shape[1]
		# config = dict(fc1_output_dim=5,fc2_output_dim=0,n_epochs=10)
		# config = dict(feature_dim=x_train.shape[-1],output_dim=32,fc1_output_dim=0,fc2_output_dim=0,n_epochs=100,batch_size=128)
		config = self.config
		config['feature_dim'] = x_train.shape[-1]
		config['lr'] = self.lr
		config['activation'] = self.activation
		BATCH_SIZE = config['batch_size']
		n_epochs = config['n_epochs']
		# earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=self.step, verbose=0, mode='auto')
		earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=0, mode='auto')
		MODEL_PATH = 'test_%d'%(run_id)
		checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
		# roc_cb = roc_callback([x_test,t_test],y_test)
		# model = get_model2(x_mtx.shape[-2])
		# if attention==1:
		# 	# model = get_model2a1_attention_sequential(context_size,config)
		# 	if self.predict_context==1:
		# 		self.model = get_model2a1_attention(context_size,config)
		# else:
		# 	# model = get_model2a1(context_size,config)
		# 	self.model = get_model2a1_sequential(context_size,config)

		attention = self.attention
		print(self.predict_context, self.attention)
		if self.predict_context==1:
			if attention==1:
				print('get_model2a1_attention')
				model = get_model2a1_attention(context_size,config)		# context self-attention model and prediction per position
			else:
				print('get_model2a1_sequential')
				model = get_model2a1_sequential(context_size,config)	# context without attention and prediction per position
		else:
			if attention==1:
				print('get_model2a1_attention_1')
				model = get_model2a1_attention_1(context_size,config)	# context with attention from intermedicate layer
			elif attention==2:
				print('get_model2a1_attention_2')
				model = get_model2a1_attention_2(context_size,config)	# context with attention from input
			else:
				print('get_model2a_sequential')
				model = get_model2a_sequential(context_size,config)		# context without attention
			
		# model.fit(X_train,y_train,epochs = 100,batch_size = BATCH_SIZE,validation_data = [X_test,y_test],class_weight=build_classweight(y_train),callbacks=[earlystop,checkpointer,roc_cb])
		# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_test,y_test],callbacks=[earlystop,checkpointer])
		model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
		# model.load_weights(MODEL_PATH)

		model_path = 'model_%d_%d_%d.h5'%(run_id,type_id2,context_size)
		model.save(model_path)

		vec1 = []
		# y_predicted_train = model.predict(x_train)
		self.model = model
		y_predicted_valid = model.predict(x_valid)
		y_predicted_test = model.predict(x_test)
		# y_predicted_test_1 = model.predict(x_test_1)

		print(y_predicted_valid.shape, y_predicted_test.shape)
		id1 = self.flanking
		# y_predicted_train, y_predicted_valid, y_predicted_test = y_predicted_train[:,id1], y_predicted_valid[:,id1], y_predicted_test[:,id1]
		# y_predicted_train, y_predicted_valid, y_predicted_test = np.ravel(y_predicted_train), np.ravel(y_predicted_valid), np.ravel(y_predicted_test)

		flanking1 = self.flanking1
		# y_predicted_train = read_predict(y_predicted_train, vec_train_local, [], flanking1)

		if self.predict_context==1:
			# y_predicted_valid = read_predict(y_predicted_valid, vec_valid_local, [], flanking1)
			# y_predicted_test_1 = read_predict(y_predicted_test_1, vec_test1_local, [], flanking1)
			# y_predicted_test = read_predict(y_predicted_test, vec_test_local, [], flanking1)

			y_predicted_valid = read_predict(y_predicted_valid, vec_valid_local, [], flanking1)
			# y_predicted_test_1 = read_predict(y_predicted_test_1, vec_test1_local, [], flanking1)
			y_predicted_test = read_predict(y_predicted_test, vec_test_local, [], flanking1)

			# y_train, y_valid, y_test_1, y_test = y_train[:,id1], y_valid[:,id1], y_test_1[:,id1], y_test[:,id1]
			y_train, y_valid, y_test = y_train[:,id1], y_valid[:,id1], y_test[:,id1]
		else:
			# y_predicted_valid, y_predicted_test_1, y_predicted_test = np.ravel(y_predicted_valid), np.ravel(y_predicted_test_1), np.ravel(y_predicted_test)
			y_predicted_valid, y_predicted_test = np.ravel(y_predicted_valid), np.ravel(y_predicted_test)

		# y_train, y_valid, y_test_1, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test_1), np.ravel(y_test)
		y_train, y_valid, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test)

		temp1 = score_2a(y_valid, y_predicted_valid)
		vec1.append(temp1)
		print(temp1)

		# temp1 = score_2a(y_test_1, y_predicted_test_1)
		# vec1.append(temp1)
		# print(temp1)
		
		temp1 = score_2a(y_test, y_predicted_test)
		vec1.append(temp1)
		print(temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		# dict1['y_valid'], dict1['y_test1'], dict1['y_test'] = y_valid, y_test_1, y_test
		dict1['y_valid'], dict1['y_test'] = y_valid, y_test
		# dict1['y_predicted_valid'], dict1['y_predicted_test_1'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test_1, y_predicted_test
		dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test

		return dict1

	def training_2(self,x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,idx_train,idx_valid,idx_test,type_id2):
		
		train_sel_list, test_sel_list = self.train_sel_list, self.test_sel_list
		tol = self.tol
		L = self.flanking
		run_id = self.run_id
			
		x_test, y_test, vec_test, vec_test_local = sample_select2a(x_test1_trans, y_signal_test_ori, test_sel_list, tol, L)
		x_train, y_train, vec_train, vec_train_local = sample_select2a(x_train1_trans[idx_train], y_signal_train1_ori[idx_train], train_sel_list[idx_train], tol, L)
		x_valid, y_valid, vec_valid, vec_valid_local = sample_select2a(x_train1_trans[idx_valid], y_signal_train1_ori[idx_valid], train_sel_list[idx_valid], tol, L)
		# x_test_1, y_test_1, vec_test1, vec_test1_local = sample_select2a(x_train1_trans[idx_test], y_signal_train1_ori[idx_test], train_sel_list[idx_test], tol, L)

		print(x_train.shape,x_valid.shape)

		context_size = x_train.shape[1]
		# config = dict(fc1_output_dim=5,fc2_output_dim=0,n_epochs=10)
		# config = dict(feature_dim=x_train.shape[-1],output_dim=32,fc1_output_dim=0,fc2_output_dim=0,n_epochs=100,batch_size=128)
		config = self.config
		config['feature_dim'] = x_train.shape[-1]
		config['lr'] = self.lr
		config['activation'] = self.activation
		BATCH_SIZE = config['batch_size']
		n_epochs = config['n_epochs']
		# earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=self.step, verbose=0, mode='auto')
		earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=0, mode='auto')
		MODEL_PATH = 'test_%d'%(run_id)
		checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
		# roc_cb = roc_callback([x_test,t_test],y_test)
		# model = get_model2(x_mtx.shape[-2])
		# if attention==1:
		# 	# model = get_model2a1_attention_sequential(context_size,config)
		# 	if self.predict_context==1:
		# 		self.model = get_model2a1_attention(context_size,config)
		# else:
		# 	# model = get_model2a1(context_size,config)
		# 	self.model = get_model2a1_sequential(context_size,config)

		attention = self.attention
		print(self.predict_context, self.attention)
		if self.predict_context==1:
			if attention==1:
				print('get_model2a1_attention')
				model = get_model2a1_attention(context_size,config)		# context self-attention model and prediction per position
			else:
				print('get_model2a1_sequential')
				model = get_model2a1_sequential(context_size,config)	# context without attention and prediction per position
		else:
			if attention==1:
				print('get_model2a1_attention_1')
				model = get_model2a1_attention_1(context_size,config)	# context with attention from intermedicate layer
			elif attention==2:
				print('get_model2a1_attention_2')
				model = get_model2a1_attention_2(context_size,config)	# context with attention from input
			else:
				print('get_model2a_sequential')
				model = get_model2a_sequential(context_size,config)		# context without attention
			
		# model.fit(X_train,y_train,epochs = 100,batch_size = BATCH_SIZE,validation_data = [X_test,y_test],class_weight=build_classweight(y_train),callbacks=[earlystop,checkpointer,roc_cb])
		# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_test,y_test],callbacks=[earlystop,checkpointer])
		model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
		# model.load_weights(MODEL_PATH)

		model_path = 'model_%d_%d_%d.h5'%(run_id,type_id2,context_size)
		model.save(model_path)

		vec1 = []
		# y_predicted_train = model.predict(x_train)
		self.model = model
		y_predicted_valid = model.predict(x_valid)
		y_predicted_test = model.predict(x_test)
		# y_predicted_test_1 = model.predict(x_test_1)

		print(y_predicted_valid.shape, y_predicted_test.shape)
		id1 = self.flanking
		# y_predicted_train, y_predicted_valid, y_predicted_test = y_predicted_train[:,id1], y_predicted_valid[:,id1], y_predicted_test[:,id1]
		# y_predicted_train, y_predicted_valid, y_predicted_test = np.ravel(y_predicted_train), np.ravel(y_predicted_valid), np.ravel(y_predicted_test)

		flanking1 = self.flanking1
		# y_predicted_train = read_predict(y_predicted_train, vec_train_local, [], flanking1)

		if self.predict_context==1:
			# y_predicted_valid = read_predict(y_predicted_valid, vec_valid_local, [], flanking1)
			# y_predicted_test_1 = read_predict(y_predicted_test_1, vec_test1_local, [], flanking1)
			# y_predicted_test = read_predict(y_predicted_test, vec_test_local, [], flanking1)

			y_predicted_valid = read_predict(y_predicted_valid, vec_valid_local, [], flanking1)
			# y_predicted_test_1 = read_predict(y_predicted_test_1, vec_test1_local, [], flanking1)
			y_predicted_test = read_predict(y_predicted_test, vec_test_local, [], flanking1)

			# y_train, y_valid, y_test_1, y_test = y_train[:,id1], y_valid[:,id1], y_test_1[:,id1], y_test[:,id1]
			y_train, y_valid, y_test = y_train[:,id1], y_valid[:,id1], y_test[:,id1]
		else:
			# y_predicted_valid, y_predicted_test_1, y_predicted_test = np.ravel(y_predicted_valid), np.ravel(y_predicted_test_1), np.ravel(y_predicted_test)
			y_predicted_valid, y_predicted_test = np.ravel(y_predicted_valid), np.ravel(y_predicted_test)

		# y_train, y_valid, y_test_1, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test_1), np.ravel(y_test)
		y_train, y_valid, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test)

		temp1 = score_2a(y_valid, y_predicted_valid)
		vec1.append(temp1)
		print(temp1)

		# temp1 = score_2a(y_test_1, y_predicted_test_1)
		# vec1.append(temp1)
		# print(temp1)
		
		temp1 = score_2a(y_test, y_predicted_test)
		vec1.append(temp1)
		print(temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		# dict1['y_valid'], dict1['y_test1'], dict1['y_test'] = y_valid, y_test_1, y_test
		dict1['y_valid'], dict1['y_test'] = y_valid, y_test
		# dict1['y_predicted_valid'], dict1['y_predicted_test_1'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test_1, y_predicted_test
		dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test

		return dict1

	def training_3(self,x_train1_trans,x_test1_trans,y_signal_train1_ori,y_signal_test_ori,idx_train,idx_valid,idx_test,type_id2):
		
		train_sel_list, test_sel_list = self.train_sel_list, self.test_sel_list
		tol = self.tol
		L = self.flanking
		run_id = self.run_id
			
		x_test, y_test, vec_test, vec_test_local = sample_select2a_3(x_test1_trans, y_signal_test_ori, test_sel_list, tol, L)
		x_train, y_train, vec_train, vec_train_local = sample_select2a_3(x_train1_trans[idx_train], y_signal_train1_ori[idx_train], train_sel_list[idx_train], tol, L)
		x_valid, y_valid, vec_valid, vec_valid_local = sample_select2a_3(x_train1_trans[idx_valid], y_signal_train1_ori[idx_valid], train_sel_list[idx_valid], tol, L)
		# x_test_1, y_test_1, vec_test1, vec_test1_local = sample_select2a(x_train1_trans[idx_test], y_signal_train1_ori[idx_test], train_sel_list[idx_test], tol, L)

		print("training_3")
		print(x_train.shape,x_valid.shape,x_test.shape,y_train.shape,y_valid.shape,y_test.shape)

		context_size = x_train.shape[1]
		# config = dict(fc1_output_dim=5,fc2_output_dim=0,n_epochs=10)
		# config = dict(feature_dim=x_train.shape[-1],output_dim=32,fc1_output_dim=0,fc2_output_dim=0,n_epochs=100,batch_size=128)
		config = self.config
		config['feature_dim'] = x_train.shape[-1]
		config['lr'] = self.lr
		config['activation'] = self.activation
		BATCH_SIZE = config['batch_size']
		n_epochs = config['n_epochs']
		n_ratio = 16
		config['output_dim2'] = 16
		config['ltype'] = self.ltype
		config['batch_norm'] = self.batch_norm
		# earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=self.step, verbose=0, mode='auto')
		earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=0, mode='auto')
		MODEL_PATH = 'test_%d'%(run_id)
		checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
		# roc_cb = roc_callback([x_test,t_test],y_test)
		# model = get_model2(x_mtx.shape[-2])
		# if attention==1:
		# 	# model = get_model2a1_attention_sequential(context_size,config)
		# 	if self.predict_context==1:
		# 		self.model = get_model2a1_attention(context_size,config)
		# else:
		# 	# model = get_model2a1(context_size,config)
		# 	self.model = get_model2a1_sequential(context_size,config)

		attention = self.attention
		print(self.predict_context, self.attention)
		if self.predict_context==1:
			if attention==1:
				print('get_model2a1_attention')
				model = get_model2a1_attention_3(context_size,config)		# context self-attention model and prediction per position
			else:
				print('get_model2a1_sequential')
				model = get_model2a1_sequential(context_size,config)	# context without attention and prediction per position
		# else:
		# 	if attention==1:
		# 		print('get_model2a1_attention_1')
		# 		model = get_model2a1_attention_1(context_size,config)	# context with attention from intermedicate layer
		# 	elif attention==2:
		# 		print('get_model2a1_attention_2')
		# 		model = get_model2a1_attention_2(context_size,config)	# context with attention from input
		# 	else:
		# 		print('get_model2a_sequential')
		# 		model = get_model2a_sequential(context_size,config)		# context without attention
			
		# model.fit(X_train,y_train,epochs = 100,batch_size = BATCH_SIZE,validation_data = [X_test,y_test],class_weight=build_classweight(y_train),callbacks=[earlystop,checkpointer,roc_cb])
		# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_test,y_test],callbacks=[earlystop,checkpointer])
		model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
		# model.load_weights(MODEL_PATH)

		model_path = 'model_%d_%d_%d.h5'%(run_id,type_id2,context_size)
		model.save(model_path)

		vec1 = []
		y_predicted_train = model.predict(x_train)
		self.model = model
		y_predicted_valid = model.predict(x_valid)
		y_predicted_test = model.predict(x_test)
		# y_predicted_test_1 = model.predict(x_test_1)

		print(y_predicted_valid.shape, y_predicted_test.shape)
		id1 = self.flanking
		# y_predicted_train, y_predicted_valid, y_predicted_test = y_predicted_train[:,id1], y_predicted_valid[:,id1], y_predicted_test[:,id1]
		# y_predicted_train, y_predicted_valid, y_predicted_test = np.ravel(y_predicted_train), np.ravel(y_predicted_valid), np.ravel(y_predicted_test)

		flanking1 = self.flanking1
		# y_predicted_train = read_predict(y_predicted_train, vec_train_local, [], flanking1)

		if self.predict_context==1:
			# y_predicted_valid = read_predict(y_predicted_valid, vec_valid_local, [], flanking1)
			# y_predicted_test_1 = read_predict(y_predicted_test_1, vec_test1_local, [], flanking1)
			# y_predicted_test = read_predict(y_predicted_test, vec_test_local, [], flanking1)

			# y_predicted_train = read_predict_3(y_predicted_train, vec_train_local, [], flanking1)
			# y_predicted_valid = read_predict_3(y_predicted_valid, vec_valid_local, [], flanking1)
			# # y_predicted_test_1 = read_predict(y_predicted_test_1, vec_test1_local, [], flanking1)
			# y_predicted_test = read_predict_3(y_predicted_test, vec_test_local, [], flanking1)

			y_predicted_train = y_predicted_train[:,id1]
			y_predicted_valid = y_predicted_valid[:,id1]
			# y_predicted_test_1 = read_predict(y_predicted_test_1, vec_test1_local, [], flanking1)
			y_predicted_test = y_predicted_test[:,id1]

			# y_train, y_valid, y_test_1, y_test = y_train[:,id1], y_valid[:,id1], y_test_1[:,id1], y_test[:,id1]
			y_train, y_valid, y_test = y_train[:,id1], y_valid[:,id1], y_test[:,id1]
		else:
			# y_predicted_valid, y_predicted_test_1, y_predicted_test = np.ravel(y_predicted_valid), np.ravel(y_predicted_test_1), np.ravel(y_predicted_test)
			# y_predicted_valid, y_predicted_test = np.ravel(y_predicted_valid), np.ravel(y_predicted_test)
			pass

		# y_train, y_valid, y_test_1, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test_1), np.ravel(y_test)
		# y_train, y_valid, y_test = np.ravel(y_train), np.ravel(y_valid), np.ravel(y_test)

		num1 = y_valid.shape[-1]
		for i in range(0,num1):
			temp1 = score_2a(y_train[:,i], y_predicted_train[:,i])
			vec1.append(temp1)
			print(i+1,temp1)

		for i in range(0,num1):
			temp1 = score_2a(y_valid[:,i], y_predicted_valid[:,i])
			vec1.append(temp1)
			print(i+1,temp1)
			
		for i in range(0,num1):
			temp1 = score_2a(y_test[:,i], y_predicted_test[:,i])
			vec1.append(temp1)
			print(i+1,temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		# dict1['y_valid'], dict1['y_test1'], dict1['y_test'] = y_valid, y_test_1, y_test
		dict1['y_valid'], dict1['y_test'] = y_valid, y_test
		# dict1['y_predicted_valid'], dict1['y_predicted_test_1'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test_1, y_predicted_test
		dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test
		dict1['y_train'], dict1['y_predicted_train'] = y_train, y_predicted_train

		return dict1

	# compare using kmer features
	def kmer_compare_single1(self, species_vec1, train_chromvec, test_chromvec, feature_idx, type_id):

		species_name = species_vec1[0]
		# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
		# data_vec.append(data1_sub)
		serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
		print("map_idx",map_idx.shape)

		path2 = '/volume01/yy3/seq_data/dl/replication_timing'

		filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
		# filename2a = 'test_seq_%s.1.txt'%(species_name)
		file1 = pd.read_csv(filename1,sep='\t')
		
		col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
		chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])

		filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
		# filename2a = 'test_seq_%s.1.txt'%(species_name)
		file1 = pd.read_csv(filename1,sep='\t')
		
		col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
		chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
		label = np.asarray(file1['label'])
		group_label = np.asarray(file1['group_label'])
		signal = np.asarray(file1['signal'])
		print(signal.shape)
		print(feature_idx)

		train_sel_idx, test_sel_idx = [], []
		data_vec = []
		x_train1, x_test = [], []
		if type_id==0 or type_id==2:
			ratio = 1
		else:
			ratio = 0.5
		for chrom_id in train_chromvec:
			chrom_id1 = 'chr%s'%(chrom_id)
			id1 = np.where(chrom==chrom_id1)[0]
			num1 = int(len(id1)*ratio)
			id1 = id1[0:num1]
			train_sel_idx.extend(id1)
			id1_ori = np.where(chrom_ori==chrom_id1)[0]
			filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
			file2 = np.load(filename2)
			t_signal = np.asarray(file2)
			trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
			print("trans_id1", trans_id1.shape)
			t_signal = t_signal[trans_id1]
			x_train1.extend(t_signal)

		# test in one species
		for chrom_id in test_chromvec:
			chrom_id1 = 'chr%s'%(chrom_id)
			id2 = np.where(chrom==chrom_id1)[0]
			test_sel_idx.extend(id2)
			id2_ori = np.where(chrom_ori==chrom_id1)[0]
			filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
			file2 = np.load(filename2)
			t_signal = np.asarray(file2)
			trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
			print("trans_id2", trans_id2.shape)
			t_signal = t_signal[trans_id2]
			x_test.extend(t_signal)

		print(len(train_sel_idx),len(test_sel_idx))
		# type_id = 1 or type_id = 2: add new species
		if type_id==1 or type_id==2:
			train_sel_idx1 = map_idx[train_sel_idx]
			test_sel_idx1 = map_idx[test_sel_idx]
			t_signal = []
			num1 = len(species_vec1)

			# train in multiple species
			for i in range(1,num1):
				species_name = species_vec1[i]
				filename1 = '%s/training2_kmer_%s.npy'%(path2,species_name)
				data1 = np.load(filename1)
				data1_sub = data1[train_sel_idx1]
				# data_vec.append(data1_sub)
				x_train1.extend(np.asarray(data1_sub))

				filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
				# filename2a = 'test_seq_%s.1.txt'%(species_name)
				file1 = pd.read_csv(filename1,sep='\t')
				signal = np.asarray(file1['signal'])
				t_signal.extend(signal[train_sel_idx])

			t_signal = np.asarray(t_signal)
			
		x_train1, x_test = np.asarray(x_train1), np.asarray(x_test)
		# x_train1 = x_train1[:,feature_idx]
		# x_test = x_test[:,feature_idx]
		print(x_train1.shape,x_test.shape)

		if type_id==1 or type_id==2:
			y_signal_train1 = np.hstack((signal[train_sel_idx],t_signal))
		else:
			y_signal_train1 = signal[train_sel_idx]

		y_signal_test = signal[test_sel_idx]
		print(x_train1.shape,y_signal_train1.shape)

		x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.2, random_state=42)

		vec1 = []
		print("LR")
		reg = LinearRegression().fit(x_train, y_train)
		y_predicted_valid = reg.predict(x_valid)
		y_predicted_test = reg.predict(x_test)
		print(reg.coef_,reg.intercept_)

		score1, score2 = score_2(y_valid, y_predicted_valid)
		vec1.append([score1,score2])
		print(score1,score2)
	
		score1, score2 = score_2(y_signal_test, y_predicted_test)
		vec1.append([score1,score2])
		print(score1,score2)

		print(vec1)

		return vec1

	# select sample
	def sample_select(self, x_mtx, idx_sel_list, tol=5, L=5):

		num1 = len(idx_sel_list)
		feature_dim = x_mtx.shape[1]
		# L = 5
		size1 = 2*L+1
		vec1_list = np.zeros((num1,size1))
		feature_list = np.zeros((num1,size1*feature_dim))
		for i in range(0,num1):
			temp1 = idx_sel_list[i]
			t_chrom, t_serial = temp1[0], temp1[1]
			id1 = []
			for k in range(-L,L+1):
				id2 = np.min((np.max((i+k,0)),num1-1))
				id1.append(id2)
			# print(id1)
			
			vec1 = []
			start1 = t_serial
			t_id = i
			for k in range(1,L+1):
				id2 = id1[L-k]
				if (idx_sel_list[id2,0]==t_chrom) and (idx_sel_list[id2,1]>=start1-tol):
					vec1.append(id2)
					t_id = id2
					start1 = idx_sel_list[id2,1]
				else:
					vec1.append(t_id)
			vec1 = vec1[::-1]
			start1 = t_serial
			t_id = i
			vec1.append(t_id)
			for k in range(1,L+1):
				id2 = id1[L+k]
				if (idx_sel_list[id2,0]==t_chrom) and (idx_sel_list[id2,1]<=start1+tol):
					vec1.append(id2)
					t_id = id2
					start1 = idx_sel_list[id2,1]
				else:
					vec1.append(t_id)

			t_feature = x_mtx[vec1]

			vec1_list[i] = idx_sel_list[vec1,1]
			feature_list[i] = np.ravel(t_feature)

			if i%10000==0:
				print(i,t_feature.shape,vec1,vec1_list[i])

		return feature_list, vec1_list

# Generate unlabeled data
def seq2sentence(filename1, output_filename):
	word = int(word)
	if not os.path.isfile(filename1):
		print("Generating sentences...")
		outfile = open(output_filename, "w")
		file = open(filename1, 'r')
		gen_seq = ""
		lines = file.readlines()
		for line in lines:
			line = line.strip()
			sentence = processSeq.DNA2Sentence(line,word)
			outfile.write(sentence+"\n")

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="0", help="experiment id")
	parser.add_option("-f","--chromosome", default="1", help="Chromosome name")
	parser.add_option("-g","--generate", default="0", help="whether to generate feature vector: 1: generate; 0: not generate")
	parser.add_option("-n","--n_epochs", default="100", help="number of epochs")
	parser.add_option("-c","--chromvec",default="1",help="chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-t","--testchromvec",default="10",help="test chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-i","--species",default="0",help="species id")
	parser.add_option("-j","--featureid",default="0",help="feature idx")
	parser.add_option("-u","--typeid",default="0",help="type id")
	parser.add_option("-b","--cell",default="GM",help="cell type")
	parser.add_option("-l","--method",default="0",help="network")
	parser.add_option("-m","--ftype",default="-5",help="feature type phyloP")
	parser.add_option("-p","--ftrans",default="50,50",help="transform feature dimension")
	parser.add_option("-q","--tlist",default="3,1,2",help="feature transform method")
	parser.add_option("-v","--normalize",default="0",help="normalize feature")
	parser.add_option("-w","--flanking",default="5",help="flanking region")
	parser.add_option("--b1",default="128",help="batch size")
	parser.add_option("-h","--unit",default="32",help="hidden unit")
	parser.add_option("-a","--attention",default="0",help="use attention or not")
	parser.add_option("--fc1",default="0",help="fc1 output dim")
	parser.add_option("--fc2",default="0",help="fc2 output dim")
	parser.add_option("--lr",default="0.001",help="learning rate")
	parser.add_option("--step",default="5",help="tolerate steps")
	parser.add_option("--delta",default="0.001",help="delta")
	parser.add_option("--activation",default="relu",help="activation")

	(opts, args) = parser.parse_args()
	return opts

def run1():

	ref_filename = 'hg38.5k.bed'
	feature_idvec = [1,1,1,1]

	reader = Reader(ref_filename,feature_idvec)

	reader.run_1()

	return True

def run2():

	ref_filename = 'hg38.5k.bed'
	feature_idvec = [1,1,1,1]

	reader = Reader(ref_filename,feature_idvec)

	reader.run_2()

	return True

def run3(chromosome):

	ref_filename = 'hg38.5k.bed'
	feature_idvec = [1,1,1,1]
	reader = Reader(ref_filename,feature_idvec)
	reader.run_3(chromosome)

	return True

def run5(resolution):

	ref_filename = 'hg38.5k.bed'
	feature_idvec = [1,1,1,1]
	reader = Reader(ref_filename,feature_idvec)
	# resolution = '5k'
	reader.run_5(str(resolution))
	
	return True

def run6(chromosome,run_id,generate,chromvec,test_chromvec,n_epochs,species_id,
		featureid,type_id,cell,method,ftype,ftrans,tlist,flanking,normalize,unit,
		batch,lr,step,activation,delta,attention,fc1,fc2):

	chrom_id_ori = str(chromosome)
	run_id = int(run_id)
	generate = int(generate)
	chrom_vec = str(chromvec)
	test_chromvec = str(test_chromvec)
	ftype = str(ftype)
	tlist = str(tlist)
	normalize = int(normalize)
	flanking = int(flanking)
	hidden_unit = int(unit)
	batch_size = int(batch)
	fc1_output_dim = int(fc1)
	fc2_output_dim = int(fc2)
	n_epochs = int(n_epochs)
	attention1 = int(attention)
	lr = float(lr)
	step = int(step)
	activation = str(activation)
	delta = float(delta)
	if activation=="None":
		activation = ''

	temp1 = chrom_vec.split(',')
	chrom_vec = [str(chrom_id) for chrom_id in temp1]
	temp1 = test_chromvec.split(',')
	test_chromvec = [str(chrom_id) for chrom_id in temp1]
	n_epochs = int(n_epochs)
	temp1 = featureid.split(',')
	feature_idx = [int(f_id) for f_id in temp1]

	temp1 = ftype.split(',')
	ftype = [int(f_id) for f_id in temp1]

	temp1 = tlist.split(',')
	t_list = [int(t_id) for t_id in temp1]

	temp1 = ftrans.split(',')
	ftrans = [int(t_id) for t_id in temp1]

	# feature_dim_transform = int(ftrans[0])
	feature_dim_transform = ftrans

	# species_id = 'hg19'
	species_idvec = ['hg19','chimp','orangutan','gibbon','greenmonkey']
	species_idvec1 = ['hg19','panTro4','ponAbe2','nomLeu3','chlSab1']
	species_id = int(species_id)
	species_id1 = species_idvec1[species_id]
	species_name = species_idvec[species_id]
	print(species_id1,species_name)
	train_chromvec = chrom_vec

	cell = str(cell)
	method = int(method)

	kmer_size = [6,5]
	t_repli_seq= RepliSeq(chromosome,run_id,generate,chrom_vec,test_chromvec,n_epochs,species_id,
					feature_idx,type_id,cell,method,ftype,ftrans,t_list,flanking,normalize,
					hidden_unit,batch_size,lr,step,activation,delta,attention,fc1,fc2,kmer_size)
	# t_replic_seq.load_features_transform()
	# t_repli_seq.set_generate(generate)
	# t_repli_seq.kmer_compare_single2a2_6_weighted1(feature_dim_transform)

	file_path = '/volume01/yy3/seq_data/dl/replication_timing3'
	species_id = 'hg38'
	resolution = '5k'
	ref_filename = '%s/hg38_%s_serial.bed'%(file_path,resolution)
	# filename1 = '%s/data_2/%s.smooth.sorted.bed'%(file_path,cell)
	filename1 = ref_filename
	t_repli_seq.load_ref_serial(ref_filename)
	t_repli_seq.load_local_serial(filename1)
	# feature_dim_transform = [50,50]
	file_prefix_kmer = 'test'
	prefix = 'test'
	t_repli_seq.set_species_id(species_id,resolution)
	feature_dim_motif = 769
	t_repli_seq.load_features_transform(file_path,file_prefix_kmer,prefix,feature_dim_motif,feature_dim_transform)

def run7(chromosome,run_id,generate,chromvec,test_chromvec,n_epochs,species_id,
		featureid,type_id,cell,method,ftype,ftrans,tlist,flanking,normalize,unit,
		batch,lr,step,activation,delta,attention,fc1,fc2,feature_dim_motif=769,select=1):

	chrom_id_ori = str(chromosome)
	run_id = int(run_id)
	generate = int(generate)
	chrom_vec = str(chromvec)
	test_chromvec = str(test_chromvec)
	ftype = str(ftype)
	tlist = str(tlist)
	normalize = int(normalize)
	flanking = int(flanking)
	hidden_unit = int(unit)
	batch_size = int(batch)
	fc1_output_dim = int(fc1)
	fc2_output_dim = int(fc2)
	n_epochs = int(n_epochs)
	attention1 = int(attention)
	lr = float(lr)
	step = int(step)
	activation = str(activation)
	delta = float(delta)
	if activation=="None":
		activation = ''

	temp1 = chrom_vec.split(',')
	chrom_vec = [str(chrom_id) for chrom_id in temp1]
	temp1 = test_chromvec.split(',')
	test_chromvec = [str(chrom_id) for chrom_id in temp1]
	n_epochs = int(n_epochs)
	temp1 = featureid.split(',')
	feature_idx = [int(f_id) for f_id in temp1]

	temp1 = ftype.split(',')
	ftype = [int(f_id) for f_id in temp1]

	temp1 = tlist.split(',')
	t_list = [int(t_id) for t_id in temp1]

	temp1 = ftrans.split(',')
	ftrans = [int(t_id) for t_id in temp1]

	# feature_dim_transform = int(ftrans[0])
	feature_dim_transform = ftrans

	# species_id = 'hg19'
	species_idvec = ['hg19','chimp','orangutan','gibbon','greenmonkey']
	species_idvec1 = ['hg19','panTro4','ponAbe2','nomLeu3','chlSab1']
	species_id = int(species_id)
	species_id1 = species_idvec1[species_id]
	species_name = species_idvec[species_id]
	print(species_id1,species_name)
	train_chromvec = chrom_vec

	cell = str(cell)
	method = int(method)

	file_path = '/volume01/yy3/seq_data/dl/replication_timing3'
	species_id = 'hg38'
	resolution = '5k'
	ref_filename = '%s/hg38_%s_serial.bed'%(file_path,resolution)
	filename1 = '%s/data_2/%s.smooth.sorted.bed'%(file_path,cell)
	# feature_dim_transform = [50,50]
	print(feature_dim_transform)
	file_prefix_kmer = 'test'
	prefix = 'test'	
	#feature_dim_motif = 769

	t_repli_seq= RepliSeq(chromosome,run_id,generate,chrom_vec,test_chromvec,n_epochs,species_id,
					featureid,type_id,cell,method,ftype,ftrans,t_list,flanking,normalize,
					hidden_unit,batch_size,lr,step,activation,delta,attention,fc1,fc2)

	t_repli_seq.set_species_id(species_id,resolution)

	t_repli_seq.set_generate(generate)
	t_repli_seq.set_featuredim_motif(feature_dim_motif)

	if select>0:
		chromvec = [1,2,3,4]
		t_repli_seq.set_chromvec_training(chromvec)

		# t_repli_seq.kmer_compare_single2a2_6_weighted2(file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform)
		t_repli_seq.kmer_compare_single2a2_6_weighted5(file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform)

		chromvec = [1,2,3,4]
		t_repli_seq.set_chromvec_training(chromvec,run_id+20,0.95)

		# t_repli_seq.kmer_compare_single2a2_6_weighted2(file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform)
		t_repli_seq.kmer_compare_single2a2_6_weighted5(file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform)

	else:
		t_repli_seq.kmer_compare_single2a2_6_weighted5(file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform)


def run_initialize(chromosome,run_id,generate,chromvec,test_chromvec,n_epochs,species_id,
		featureid,type_id,cell,method,ftype,ftrans,tlist,flanking,normalize,unit,
		batch,lr,step,activation,delta,attention,fc1,fc2,feature_dim_motif=769):
	
	chrom_id_ori = str(chromosome)
	run_id = int(run_id)
	generate = int(generate)
	chrom_vec = str(chromvec)
	test_chromvec = str(test_chromvec)
	ftype = str(ftype)
	tlist = str(tlist)
	normalize = int(normalize)
	flanking = int(flanking)
	hidden_unit = int(unit)
	batch_size = int(batch)
	fc1_output_dim = int(fc1)
	fc2_output_dim = int(fc2)
	n_epochs = int(n_epochs)
	attention1 = int(attention)
	lr = float(lr)
	step = int(step)
	activation = str(activation)
	delta = float(delta)
	if activation=="None":
		activation = ''

	temp1 = chrom_vec.split(',')
	chrom_vec = [str(chrom_id) for chrom_id in temp1]
	temp1 = test_chromvec.split(',')
	test_chromvec = [str(chrom_id) for chrom_id in temp1]
	n_epochs = int(n_epochs)
	temp1 = featureid.split(',')
	feature_idx = [int(f_id) for f_id in temp1]

	temp1 = ftype.split(',')
	ftype = [int(f_id) for f_id in temp1]

	temp1 = tlist.split(',')
	t_list = [int(t_id) for t_id in temp1]

	temp1 = ftrans.split(',')
	ftrans = [int(t_id) for t_id in temp1]

	# feature_dim_transform = int(ftrans[0])
	feature_dim_transform = ftrans

	# species_id = 'hg19'
	species_idvec = ['hg19','chimp','orangutan','gibbon','greenmonkey']
	species_idvec1 = ['hg19','panTro4','ponAbe2','nomLeu3','chlSab1']
	species_id = int(species_id)
	species_id1 = species_idvec1[species_id]
	species_name = species_idvec[species_id]
	print(species_id1,species_name)
	train_chromvec = chrom_vec

	cell = str(cell)
	method = int(method)

	file_path = '/volume01/yy3/seq_data/dl/replication_timing3'
	# file_path = '.'
	species_id = 'hg38'
	resolution = '5k'
	ref_filename = '%s/hg38_%s_serial.bed'%(file_path,resolution)
	filename1 = '%s/data_2/%s.smooth.sorted.bed'%(file_path,cell)
	# filename1 = '%s/%s.smooth.sorted.bed'%(file_path,cell)
	# feature_dim_transform = [50,50]
	print(feature_dim_transform)
	file_prefix_kmer = 'test'
	prefix = 'test'	
	#feature_dim_motif = 769
	# t_repli_seq= RepliSeq(chromosome,run_id,generate,chrom_vec,test_chromvec,n_epochs,species_id,
	# 				featureid,type_id,cell,method,ftype,ftrans,t_list,flanking,normalize,
	# 				hidden_unit,batch_size,lr,step,attention,fc1,fc2)
	t_repli_seq= RepliSeq(chromosome,run_id,generate,chrom_vec,test_chromvec,n_epochs,species_id,
					feature_idx,type_id,cell,method,ftype,ftrans,t_list,flanking,normalize,
					hidden_unit,batch_size,lr,step,activation,delta,attention,fc1,fc2)

	t_repli_seq.load_ref_serial(ref_filename)
	t_repli_seq.load_local_serial(filename1)
	# t_repli_seq.load_local_serial(ref_filename)
	t_repli_seq.set_species_id(species_id,resolution)
	t_repli_seq.set_featuredim_motif(feature_dim_motif)

	return t_repli_seq

def run8(t_repli_seq, feature_dim_motif=769, feature_dim_transform=[50,50]):

	file_path = '/volume01/yy3/seq_data/dl/replication_timing3'
	file_prefix_kmer = 'test'
	kmer_size1 = 6
	kmer_size2 = 5
	t_repli_seq.load_features_single(file_path,file_prefix_kmer,kmer_size1,kmer_size2)

def run(chromosome,run_id,generate,chromvec,test_chromvec,n_epochs,species_id,
		featureid,type_id,cell,method,ftype,ftrans,tlist,flanking,normalize,unit,
		batch,lr,step,activation,delta,attention,fc1,fc2,feature_dim_motif=769):

	chrom_id_ori = str(chromosome)
	run_id = int(run_id)
	generate = int(generate)
	chrom_vec = str(chromvec)
	test_chromvec = str(test_chromvec)
	ftype = str(ftype)
	tlist = str(tlist)
	normalize = int(normalize)
	flanking = int(flanking)
	hidden_unit = int(unit)
	batch_size = int(batch)
	fc1_output_dim = int(fc1)
	fc2_output_dim = int(fc2)
	n_epochs = int(n_epochs)
	attention1 = int(attention)
	lr = float(lr)
	step = int(step)
	activation = str(activation)
	delta = float(delta)
	if activation=="None":
		activation = ''

	temp1 = chrom_vec.split(',')
	chrom_vec = [str(chrom_id) for chrom_id in temp1]
	temp1 = test_chromvec.split(',')
	test_chromvec = [str(chrom_id) for chrom_id in temp1]
	n_epochs = int(n_epochs)
	temp1 = featureid.split(',')
	feature_idx = [int(f_id) for f_id in temp1]

	temp1 = ftype.split(',')
	ftype = [int(f_id) for f_id in temp1]

	temp1 = tlist.split(',')
	t_list = [int(t_id) for t_id in temp1]

	temp1 = ftrans.split(',')
	ftrans = [int(t_id) for t_id in temp1]

	# feature_dim_transform = int(ftrans[0])
	feature_dim_transform = ftrans

	# species_id = 'hg19'
	species_idvec = ['hg19','chimp','orangutan','gibbon','greenmonkey']
	species_idvec1 = ['hg19','panTro4','ponAbe2','nomLeu3','chlSab1']
	species_id = int(species_id)
	species_id1 = species_idvec1[species_id]
	species_name = species_idvec[species_id]
	print(species_id1,species_name)
	train_chromvec = chrom_vec

	cell = str(cell)
	method = int(method)

	file_path = '/volume01/yy3/seq_data/dl/replication_timing3'
	species_id = 'hg38'
	resolution = '5k'
	ref_filename = '%s/hg38_%s_serial.bed'%(file_path,resolution)
	filename1 = '%s/data_2/%s.smooth.sorted.bed'%(file_path,cell)
	# feature_dim_transform = [50,50]
	print(feature_dim_transform)
	file_prefix_kmer = 'test'
	prefix = 'test'	
	#feature_dim_motif = 769

	# t_repli_seq= RepliSeq(chromosome,run_id,generate,chrom_vec,test_chromvec,n_epochs,species_id,
	# 				featureid,type_id,cell,method,ftype,ftrans,t_list,flanking,normalize,
	# 				hidden_unit,batch_size,lr,step,attention,fc1,fc2)
	t_repli_seq= RepliSeq(chromosome,run_id,generate,chrom_vec,test_chromvec,n_epochs,species_id,
					featureid,type_id,cell,method,ftype,ftrans,t_list,flanking,normalize,
					hidden_unit,batch_size,lr,step,activation,delta,attention,fc1,fc2)
	# t_replic_seq.load_features_transform()
	# t_repli_seq.set_generate(generate)
	# t_repli_seq.kmer_compare_single2a2_6_weighted1(feature_dim_transform)

	# t_repli_seq.load_ref_serial(ref_filename)
	# t_repli_seq.load_local_serial(filename1)
	t_repli_seq.set_species_id(species_id,resolution)

	t_repli_seq.set_generate(generate)
	t_repli_seq.set_featuredim_motif(feature_dim_motif)

	# t_repli_seq.kmer_compare_single2a2_6_weighted2(file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform)
	t_repli_seq.kmer_compare_single2a2_6_weighted3(file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform)
	# t_repli_seq.kmer_compare_single2a2_6_weighted3_single(file_path,file_prefix_kmer,feature_dim_motif,feature_dim_transform)

if __name__ == '__main__':

	opts = parse_args()
	run(opts.chromosome,opts.run_id,opts.generate,opts.chromvec,opts.testchromvec,
		opts.n_epochs,opts.species,opts.featureid,opts.typeid,opts.cell,
		opts.method,opts.ftype,opts.ftrans,opts.tlist,opts.flanking,opts.normalize,
		opts.unit,opts.b1,opts.lr,opts.step,opts.activation,opts.delta,opts.attention,opts.fc1,opts.fc2)

