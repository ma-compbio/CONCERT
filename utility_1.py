import pandas as pd
import numpy as np

import processSeq
import sys

import tensorflow as tf
import keras
keras.backend.image_data_format()
from keras import backend as K
from keras import regularizers
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention
from keras.engine.topology import Layer
from keras_layer_normalization import LayerNormalization
from keras.layers import Input, Dense, Average, Reshape, Lambda, Conv1D, Flatten, MaxPooling1D, UpSampling1D, GlobalMaxPooling1D
# from keras.layers import LSTM, Bidirectional, BatchNormalization, Dropout, Concatenate, Embedding, Activation, Dot, dot
from keras.layers import BatchNormalization, Dropout, Concatenate, Embedding, Activation,Dot,dot
from keras.layers import TimeDistributed, RepeatVector, Permute, merge, Multiply
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU
from keras.models import Sequential, Model, clone_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.constraints import unitnorm

import sklearn as sk
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, TSNE
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD, FastICA, MiniBatchDictionaryLearning
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, matthews_corrcoef
import xgboost
from processSeq import load_seq_1, kmer_dict, load_seq_2, load_seq_2_kmer

from scipy import stats
from scipy.stats import skew, pearsonr, spearmanr, wilcoxon, mannwhitneyu,kstest,ks_2samp, chisquare
from scipy import signal
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences
from statsmodels.stats.multitest import multipletests

from timeit import default_timer as timer
import time

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
plt.switch_backend('Agg')

import seaborn as sns
import h5py

import os.path
from optparse import OptionParser
clipped_relu = lambda x: relu(x, max_value=1.0)

import multiprocessing as mp
import threading
import sys

n_epochs = 100
drop_out_rate = 0.5
learning_rate = 0.001
validation_split_ratio = 0.1
BATCH_SIZE = 128
NUM_DENSE_LAYER = 2
NUM_CONV_LAYER_2 = 2	# number of convolutional layers
MODEL_PATH = './test2_2.2'
READ_THRESHOLD = 100
CLIPNORM1 = 1000.0

def mapping_Idx(serial1,serial2):

	if len(np.unique(serial1))<len(serial1):
		print("error! ref_serial not unique", len(np.unique(serial1)), len(serial1))
		return

	unique_flag = 1
	t_serial2 = np.unique(serial2,return_inverse=True)
	if len(t_serial2[0])<len(serial2):
		# print("serial2 not unique!")
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
	idx1[map_sortedIdx[b1]] = idx

	if unique_flag==0:
		idx1 = idx1[t_serial2[1]]

	return np.int64(idx1)

def search_Idx(serial1,serial2):
	id1 = mapping_Idx(serial1,serial2)
	b2 = np.where(id1<0)[0]
	if len(b2)>0:
		print('error!',len(b2))
		return
	return id1

def smooth(x,window_len=11,window='hanning'):
	"""smooth the data using a window with requested size.
	"""

	# if x.ndim != 1:
	#     raise ValueError, "smooth only accepts 1 dimension arrays."

	# if x.size < window_len:
	#     raise ValueError, "Input vector needs to be bigger than window size."

	assert x.ndim==1
	assert x.size==window_len

	if window_len<3:
		return x

	flag = (window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'])
	assert flag==1

	s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	if window == 'flat': #moving average
		w=numpy.ones(window_len,'d')
	else:
		w=eval('numpy.'+window+'(window_len)')

	y=numpy.convolve(w/w.sum(),s,mode='valid')
	return y

def score_2a(y, y_predicted):

	score1 = mean_squared_error(y, y_predicted)
	score2 = pearsonr(y, y_predicted)
	score3 = explained_variance_score(y, y_predicted)
	score4 = mean_absolute_error(y, y_predicted)
	score5 = median_absolute_error(y, y_predicted)
	score6 = r2_score(y, y_predicted)
	score7, pvalue = spearmanr(y,y_predicted)
	vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6, score7, pvalue]

	return vec1

def score_2a_1(data1, data2, alternative='greater'):

	try:
		mannwhitneyu_statistic, mannwhitneyu_pvalue = mannwhitneyu(data1,data2,alternative=alternative)
	except:
		mannwhitneyu_statistic, mannwhitneyu_pvalue = -1, 1.1

	if alternative=='greater':
		alternative1 = 'less'
	else:
		alternative1 = alternative

	try:
		ks_statistic, ks_pvalue = ks_2samp(data1,data2,alternative=alternative1)
	except:
		ks_statistic, ks_pvalue = -1, 1.1

	vec1 = [[mannwhitneyu_pvalue,ks_pvalue],
			[mannwhitneyu_statistic,ks_statistic]]

	return vec1

def score_2a_2(y,y_ori,y_predicted,type_id=0,chrom=[],by_chrom=0):

	sample_num = len(y)
	middle_point = 0
	if np.min(y_ori)>-0.2:
		middle_point = 0.5
	thresh_1 = len(np.where(y_ori<middle_point)[0])/sample_num
	temp1 = np.quantile(y,thresh_1)
	temp3 = np.quantile(y_predicted,thresh_1)
	thresh_2 = 0.5
	temp2 = np.quantile(y_ori,thresh_2)
	print(sample_num,thresh_1,temp1,temp2,temp3)
	print(np.max(y_ori),np.min(y_ori),np.max(y),np.min(y),np.max(y_predicted),np.min(y_predicted))

	thresh = temp1
	y1 = np.zeros_like(y,dtype=np.int8)
	y2 = y1.copy()
	y1[y>thresh] = 1

	if type_id>0:
		thresh = temp3

	y2[y_predicted>thresh] = 1
	y_predicted_scale = stats.rankdata(y_predicted,'average')/len(y_predicted)
	
	accuracy, auc, aupr, precision, recall, F1 = score_function(y1,y2,y_predicted_scale)

	list1, list2 = [], []
	if by_chrom==1:
		assert len(chrom)>0
		chrom_vec = np.unique(chrom)
		for chrom_id in chrom_vec:
			b1 = np.where(chrom==chrom_id)[0]

			t_vec1 = score_function(y1[b1],y2[b1],y_predicted_scale[b1])
			list1.append(chrom_id)
			list2.append(t_vec1)

		list2 = np.asarray(list2)

	return (accuracy, auc, aupr, precision, recall, F1, list1, list2)

def writeToBED(filename1,filename2,color_vec):

	data1 = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1)

	fields = ['chrom','start','stop','name','score','strand','thickStart','thickEnd','itemRgb']

	data2 = pd.DataFrame(columns=fields)
	for i in range(3):
		data2[fields[i]] = data1[colnames[i]]

	num1 = data1.shape[0]
	data2['name'] = list(range(num1))
	data2['score'] = [600]*num1
	data2['strand'] = ['.']*num1
	data2['thickStart'], data2['thickEnd'] = data2['start'], data2['stop']
	color1 = np.asarray([color_vec[0]]*num1)
	color1[range(1,num1,2)] = color_vec[1]
	data2['itemRgb'] = color1

	data2.to_csv(filename2,header=False,index=False,sep='\t')

	return True

# reciprocal mapping
# input: filename_1: orignal position file on genome 1
#        filename1:  positions mapped from genome 1 to genome 2
#		 filename2:  positions mapped from genome 2 to genome 1
#        output_filename1: positions with reciprocal mapping on genome 2
#        output_filename1: positions with reciprocal mapping on genome 1
def remapping_serial(filename_1,filename1,filename2,output_filename1,output_filename_1=''):

	data_1 = pd.read_csv(filename_1,header=None,sep='\t')
	chrom_1, start_1, stop_1 = np.asarray(data_1[0]), np.asarray(data_1[1]), np.asarray(data_1[2])
	serial_1 =  np.asarray(data_1[3])

	data1 = pd.read_csv(filename1,header=None,sep='\t')
	serial1 =  np.asarray(data1[3])

	data2 = pd.read_csv(filename2,header=None,sep='\t')
	chrom2, start2, stop2 = np.asarray(data2[0]), np.asarray(data2[1]), np.asarray(data2[2])
	serial2 =  np.asarray(data2[3])

	id1 = mapping_Idx(serial_1,serial2)

	assert np.sum(id1<0)==0

	num1 = len(serial2)
	id2 = np.zeros(num1,dtype=np.int8)
	for i in range(num1):
		t_chrom2, t_start2, t_stop2 = chrom2[i], start2[i], stop2[i]
		t_chrom_1, t_start_1, t_stop_1 = chrom_1[id1[i]], start_1[id1[i]], stop_1[id1[i]]

		if (t_chrom_1==t_chrom2) and (t_start2<t_stop_1) and (t_stop2>t_start_1):
			id2[i] = 1

	b1 = np.where(id2>0)[0]
	serial_2 = serial2[b1]
	id_1 = mapping_Idx(serial_1,serial_2)
	id_2 = mapping_Idx(serial1,serial_2)

	data_1  = data_1.loc[id_1,:]
	data1 = data1.loc[id_2,:]

	if output_filename_1!='':
		data_1.to_csv(output_filename_1,index=False,header=False,sep='\t')
	data1.to_csv(output_filename1,index=False,header=False,sep='\t')

	return True

# co-binding
def binding_1(filename_list,output_filename_1,distance_thresh=10000):

	filename1 = filename_list[0]
	num1 = len(filename_list)

	data1 = pd.read_csv(filename1,header=None,sep='\t')
	region_num = len(data1)

	colnames = list(data1)
	col1, col2, col3 = colnames[0], colnames[1], colnames[2]
	chrom1, start1, stop1 = np.asarray(data1[col1]), np.asarray(data1[col2]), np.asarray(data1[col3])

	for i in range(1,num1):
		filename2 = filename1_list[i]
		data2 = pd.read_csv(filename2,header=None,sep='\t')
		region_num2 = len(data2)

		colnames = list(data1)
		col1, col2, col3 = colnames[0], colnames[1], colnames[2]
		chrom2, start2, stop2 = np.asarray(data1[col1]), np.asarray(data1[col2]), np.asarray(data1[col3])

def binding1(filename1,output_filename,output_filename1=''):

	data1 = pd.read_csv(filename1,header=None,sep='\t')
	chrom1 = np.asarray(data1[0])
	b1 = np.where((chrom1!='chrX')&(chrom1!='chrY'))[0]
	data1 = data1.loc[b1,:]
	data1.reset_index(drop=True,inplace=True)
	chrom1, start1, stop1 = np.asarray(data1[0]), np.asarray(data1[1]), np.asarray(data1[2])
	chrom2, start2, stop2 = np.asarray(data1[10]), np.asarray(data1[11]), np.asarray(data1[12])
	region_num = len(chrom1)
	t1 = np.min([start1,start2],axis=0)
	t2 = np.max([stop1,stop2],axis=0)
	fields = ['chrom','start','stop','start1','stop1','start2','stop2','region_len']
	region_len = t2-t1
	print(np.min(region_len),np.max(region_len),np.median(region_len))
	data1 = pd.DataFrame(columns=fields)
	data1['chrom'] = chrom1
	data1['start'], data1['stop'] = t1, t2
	data1['start1'], data1['stop1'] = start1, stop1
	data1['start2'], data1['stop2'] = start2, stop2
	data1['region_len'] = region_len

	data1.to_csv(output_filename,header=False,index=False,sep='\t')

	if output_filename1=='':
		b1 = output_filename.find('.txt')
		output_filename1 = output_filename[0:b1]+'.bed'
	
	data2 = data1.loc[:,['chrom','start','stop']]
	data2['serial'] = np.arange(region_num)+1
	data2.to_csv(output_filename1,header=False,index=False,sep='\t')

	return data1

def region_annotation(filename1):

	compare_with_regions_peak_search1(chrom,start,stop,serial,value,seq_list,thresh_vec=[0.9])

def peak_extend(chrom_ori,start_ori,serial_ori,chrom,start,serial,flanking=2):
	
	num1 = len(chrom)
	sample_num = len(chrom_ori)
	vec1 = np.zeros(sample_num,dtype=np.int64)
	print(num1,sample_num)
	t1 = serial+flanking
	chrom_vec = np.unique(chrom)
	size1 = 2*flanking+1
	label = np.arange(1,num1+1)

	for chrom_id in chrom_vec:
		b1 = np.where(chrom==chrom_id)[0]
		t_serial1 = serial[b1]
		num2 = len(t_serial1)
		t1 = np.outer(t_serial1,np.ones(size1))
		t2 = t1+np.outer(np.ones(num2),np.arange(-flanking,flanking+1))

		b2 = np.where(chrom_ori==chrom_id)[0]
		s1 = np.min(serial_ori[b2])
		s2 = np.max(serial_ori[b2])

		t2[t2<s1] = s1
		t2[t2>s2] = s2

		t_label1 = label[b1]
		label1 = -np.repeat(t_label1,size1)

		t2 = np.ravel(t2)
		id1 = mapping_Idx(serial_ori,t2)
		id2 = (id1>=0)
		vec1[id1[id2]] = label1[id2]

	b1 = mapping_Idx(serial_ori,serial)
	assert np.sum(b1<0)==0
	vec1[b1] = label

	return vec1

# select genomic loci with high estimated importance scores
def find_region_sub1_ori(filename_list1,output_filename1,output_filename2='',load=1,config={}):

	pd.set_option("display.precision", 8)
	type_id1, type_id2 = 0, 1
	filename_centromere = config['filename_centromere']
	if (load==0) or (os.path.exists(output_filename1)==False):
		print(filename_list1)
		print(output_filename1)
		data1, chrom_numList = select_region1_merge(filename_list1,output_filename1,
												type_id1=type_id1,type_id2=type_id2,
												filename_centromere=filename_centromere)
	else:
		data1 = pd.read_csv(output_filename1,sep='\t')
		if filename_centromere!='':
			chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
			id1 = select_idx_centromere(chrom,start,stop,filename_centromere)
			print('select_idx_centromere', len(chrom), len(id1), len(id1)/len(chrom))
			data1 = data1.loc[id1,:]
			data1.reset_index(drop=True,inplace=True)

	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	vec1 = ['Q1','Q2']
	type_id = 1
	sel_column = vec1[type_id]
	print(output_filename1,list(data1),data1.shape)
	attention1 = np.asarray(data1[sel_column])
	thresh = config['thresh']	# thresh = 0.95
	# test 1: predicted attention above threshold
	id1 = np.where(attention1>thresh)[0]
	print(len(id1),len(attention1),len(id1)/len(attention1))
	vec1 = peak_extend(chrom,start,serial,chrom[id1],start[id1],serial[id1],flanking=2)

	# test 2: predicted attention local peak above 0.95
	signal = np.asarray(data1['signal'])
	attention_1 = np.asarray(data1['predicted_attention'])
	print(data1.shape)

	value = np.column_stack((attention_1,attention1))
	seq_list = generate_sequences_chrom(chrom,serial)

	num1 = len(seq_list)
	cnt1 = 0
	for i in range(num1):
		cnt1 += seq_list[i][1]-seq_list[i][0]+1
	print(cnt1)

	if 'thresh_vec_pre' in config:
		thresh_vec_pre = config['thresh_vec_pre']
	else:
		thresh_vec_pre = [0.90,0.50,0.50]
	peak_thresh_1, peak_thresh_2, peak_thresh_3 = thresh_vec_pre

	if 'distance_thresh_vec' in config:
		distance_thresh_vec = config['distance_thresh_vec']
	else:
		distance_thresh_vec = [[-1,5],[0.25,1]]
	distance_thresh_1, distance_thresh_2 = distance_thresh_vec

	thresh_vec = [peak_thresh_1]
	config = {'thresh_vec':thresh_vec}
	config['peak_type'] = 0
	config['threshold'] = distance_thresh_1[0]
	config['distance_peak_thresh'] = distance_thresh_1[1]
	print('compare with regions peak search')
	dict2 = compare_with_regions_peak_search(chrom,start,stop,serial,value,seq_list,config)
	chrom1, start1, stop1, serial1, annot1 = dict2[thresh_vec[0]]
	vec2 = peak_extend(chrom,start,serial,chrom1,start1,serial1,flanking=2)

	# test 3: predicted attention local peak above distance of 0.25
	value = np.column_stack((attention1,attention1))

	thresh_vec = [peak_thresh_2]
	config = {'thresh_vec':thresh_vec}
	config['peak_type'] = 0
	config['threshold'] = distance_thresh_2[0]	# distance of value from peak to neighbors
	config['distance_peak_thresh'] = distance_thresh_2[1]	# update for peak distance
	print('compare with regions peak search')
	dict2_1 = compare_with_regions_peak_search(chrom,start,stop,serial,value,seq_list,config)
	chrom1, start1, stop1, serial1, annot1 = dict2_1[thresh_vec[0]]
	vec2_1 = peak_extend(chrom,start,serial,chrom1,start1,serial1,flanking=2)

	# test 4: predicted attention local peak wavelet transformation
	thresh_vec = [peak_thresh_3]
	value = np.column_stack((attention_1,attention1))
	config = {'thresh_vec':thresh_vec}
	config['peak_type'] = 1
	dict3 = compare_with_regions_peak_search(chrom,start,stop,serial,value,seq_list,config)
	chrom1, start1, stop1, serial1, annot1 = dict3[thresh_vec[0]]
	vec3 = peak_extend(chrom,start,serial,chrom1,start1,serial1,flanking=2)

	print(len(vec1),len(vec2),len(vec2_1),len(vec3))
	data1['sel1'], data1['sel2'], data1['sel2.1'], data1['sel3'] = vec1, vec2, vec2_1, vec3

	if output_filename2=='':
		b1 = output_filename1.find('.txt')
		output_filename2 = output_filename1[0:b1]+'.2.txt'

	print('find_region_sub1',data1.shape)
	print(output_filename2)
	data1.to_csv(output_filename2,index=False,sep='\t',float_format='%.6f')
	
	return data1

# select genomic loci with high estimated importance scores
def find_region_sub1(filename_list1,output_filename1,output_filename2='',load=1,config={}):

	pd.set_option("display.precision", 8)
	type_id1, type_id2 = 0, 1
	filename_centromere = config['filename_centromere']
	if (load==0) or (os.path.exists(output_filename1)==False):
		print(filename_list1)
		print(output_filename1)
		data1, chrom_numList = select_region1_merge(filename_list1,output_filename1,
												type_id1=type_id1,type_id2=type_id2,
												filename_centromere=filename_centromere)
	else:
		data1 = pd.read_csv(output_filename1,sep='\t')
		if filename_centromere!='':
			chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
			id1 = select_idx_centromere(chrom,start,stop,filename_centromere)
			print('select_idx_centromere', len(chrom), len(id1), len(id1)/len(chrom))
			data1 = data1.loc[id1,:]
			data1.reset_index(drop=True,inplace=True)

	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	vec1 = ['Q1','Q2']
	type_id = 1
	sel_column = vec1[type_id]
	print(output_filename1,list(data1),data1.shape)
	# result = pd.merge(left, right, left_index=True, right_index=True, how='outer')
	attention1 = np.asarray(data1[sel_column])
	thresh = config['thresh']	# thresh = 0.95
	# test 1: predicted attention above threshold
	id1 = np.where(attention1>thresh)[0]
	print(len(id1),len(attention1),len(id1)/len(attention1))
	vec1 = peak_extend(chrom,start,serial,chrom[id1],start[id1],serial[id1],flanking=2)

	# test 2: predicted attention local peak above 0.95
	signal = np.asarray(data1['signal'])
	attention_1 = np.asarray(data1['predicted_attention'])
	print(data1.shape)

	value = np.column_stack((attention_1,attention1))
	seq_list = generate_sequences_chrom(chrom,serial)

	num1 = len(seq_list)
	cnt1 = 0
	for i in range(num1):
		cnt1 += seq_list[i][1]-seq_list[i][0]+1
	print(cnt1)

	if 'thresh_vec_pre' in config:
		thresh_vec_pre = config['thresh_vec_pre']
	else:
		thresh_vec_pre = [0.90,0.50,0.50]
	peak_thresh_1, peak_thresh_2, peak_thresh_3 = thresh_vec_pre

	if 'distance_thresh_vec' in config:
		distance_thresh_vec = config['distance_thresh_vec']
	else:
		distance_thresh_vec = [[[-1,1],[-1,3],[-1,5]],[0.25,1]]
	distance_thresh_list1, distance_thresh_2 = distance_thresh_vec

	thresh_vec = [peak_thresh_1]
	config = {'thresh_vec':thresh_vec}
	config['peak_type'] = 0

	peak_list1 = []
	for distance_thresh_1 in distance_thresh_list1: 
		config['threshold'] = distance_thresh_1[0]
		config['distance_peak_thresh'] = distance_thresh_1[1]
		print(distance_thresh_1)
		print('compare with regions peak search')
		dict2 = compare_with_regions_peak_search(chrom,start,stop,serial,value,seq_list,config)
		# dict1[0] = dict2[thresh_vec[0]]
		chrom1, start1, stop1, serial1, annot1 = dict2[thresh_vec[0]]
		vec2 = peak_extend(chrom,start,serial,chrom1,start1,serial1,flanking=2)
		peak_list1.append(vec2)

	# test 3: predicted attention local peak above distance of 0.25
	value = np.column_stack((attention1,attention1))
	thresh_vec = [peak_thresh_2]
	config = {'thresh_vec':thresh_vec}
	config['peak_type'] = 0
	config['threshold'] = distance_thresh_2[0]	# distance of value from peak to neighbors
	config['distance_peak_thresh'] = distance_thresh_2[1]	# update for peak distance
	print('compare with regions peak search')
	dict2_1 = compare_with_regions_peak_search(chrom,start,stop,serial,value,seq_list,config)
	chrom1, start1, stop1, serial1, annot1 = dict2_1[thresh_vec[0]]
	vec2_1 = peak_extend(chrom,start,serial,chrom1,start1,serial1,flanking=2)

	# test 4: predicted attention local peak wavelet transformation
	thresh_vec = [peak_thresh_3]
	value = np.column_stack((attention_1,attention1))
	config = {'thresh_vec':thresh_vec}
	config['peak_type'] = 1
	dict3 = compare_with_regions_peak_search(chrom,start,stop,serial,value,seq_list,config)
	chrom1, start1, stop1, serial1, annot1 = dict3[thresh_vec[0]]
	vec3 = peak_extend(chrom,start,serial,chrom1,start1,serial1,flanking=2)

	data1['sel1'] = vec1
	cnt = len(peak_list1)
	for i in range(cnt):
		if i==0:
			t_colname = 'sel2'
		else:
			t_colname = 'sel2.0.%d'%(i)
		data1[t_colname] = peak_list1[i]

	data1['sel2.1'], data1['sel3'] = vec2_1, vec3

	if output_filename2=='':
		b1 = output_filename1.find('.txt')
		output_filename2 = output_filename1[0:b1]+'.2.txt'

	print('find_region_sub1',data1.shape)
	print(output_filename2)
	data1.to_csv(output_filename2,index=False,sep='\t',float_format='%.6f')

	return data1

# write genomie loci to bed file
# def find_region_sub2(filename1,output_filename1,config={}):

# 	data1 = pd.read_csv(filename1,sep='\t')
# 	colnames = list(data1)
# 	colnames_1 = ['sel1','sel2','sel2.1','sel3']
# 	label_1 = np.asarray(data1.loc[:,colnames_1])
# 	sel1, sel2, sel2_1 = label_1[:,0], label_1[:,1], label_1[:,2]
# 	b1, b2, b3 = np.where(sel1>0)[0], np.where(sel2>0)[0], np.where(sel2_1>0)[0]
# 	b_1, b_2, b_3 = np.where(sel1!=0)[0], np.where(sel2!=0)[0], np.where(sel2_1!=0)[0]

# 	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
# 	sel_column = ['Q2']
# 	predicted_attention, predicted_attention1 = np.asarray(data1['predicted_attention']), np.asarray(data1[sel_column])
# 	thresh = config['thresh_select']
# 	id1 = np.where(predicted_attention1[b3]>thresh)[0]
# 	id1 = b3[id1]

# 	id2 = np.union1d(b2,id1) 	# local peaks
# 	id3 = np.union1d(b1,id2)	# local peaks or high scores

# 	id5 = np.intersect1d(b2,id3) # local peaks
# 	id6 = np.intersect1d(b1,id5) # local peaks and high values
# 	print('select',len(b1),len(b2),len(b3),len(id1),len(id2),len(id3),len(id5),len(id6))

# 	sample_num = data1.shape[0]
# 	t_label = np.zeros(sample_num,dtype=np.int8)
# 	list1 = [b1,b2,id3,id6]
# 	for i in range(len(list1)):
# 		t_label[list1[i]] = i+1

# 	sel_id = np.where(t_label>0)[0]
# 	data1 = data1.loc[sel_id,['chrom','start','stop','serial','predicted_attention','Q2']]
# 	data1.reset_index(drop=True,inplace=True)
# 	data1['label'] = t_label[t_label>0]

# 	data1.to_csv(output_filename1,index=False,sep='\t')

# 	return True

# write genomie loci to bed file
# def find_region_sub2_1(filename1,output_filename1,config={}):

# 	data1 = pd.read_csv(filename1,sep='\t')
# 	colnames = list(data1)
	
# 	# high value, local peak (distance>=1), local peak with signal difference (distance>=1), wavelet local peak
# 	colnames_1 = ['sel1','sel2','sel2.1','sel3']
# 	num1 = len(colnames)
# 	# local peak with different distances
# 	for i in range(num1):
# 		t_colname = colnames[i]
# 		if t_colname.find('sel2.0')>=0:
# 			# print(1,t_colname)
# 			colnames_1.append(t_colname)
# 		# else:
# 		# 	print(0,t_colname)

# 	colnames_2 = colnames_1[0:3]+colnames_1[4:]
# 	num2 = len(colnames_2)
# 	print('colnames_2',colnames_2)

# 	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
# 	sel_column = 'Q2'
# 	predicted_attention, predicted_attention1 = np.asarray(data1['predicted_attention']), np.asarray(data1[sel_column])
# 	thresh = config['thresh_select']

# 	region_num = data1.shape[0]
# 	mask = np.zeros((region_num,num2),dtype=np.int8)
# 	thresh = config['thresh_select']
# 	thresh_2 = config['thresh_2']
# 	for i in range(num2):
# 		t_colname = colnames_2[i]
# 		t_column = np.asarray(data1[t_colname])
# 		b1 = np.where(t_column>0)[0]
# 		if t_colname=='sel2.1':
# 			id1 = np.where(predicted_attention1[b1]>thresh)[0]
# 			b1 = b1[id1]
# 		if t_colname=='sel1':
# 			id1 = np.where(predicted_attention1[b1]>thresh_2)[0]
# 			b1 = b1[id1]
# 		mask[b1,i] = 1

# 	label_value = np.zeros(region_num,dtype=np.int32)
# 	for i1 in range(num2):
# 		label_value = label_value + (10**i1)*mask[:,i1]

# 	t_label = label_value

# 	sel_id = np.where(t_label>0)[0]
# 	data1 = data1.loc[sel_id,['chrom','start','stop','serial','predicted_attention','Q2']]
# 	data1.reset_index(drop=True,inplace=True)
# 	data1['label'] = t_label[t_label>0]

# 	data1.to_csv(output_filename1,index=False,sep='\t')

# 	return True

# merge neighboring important genomic loci into regions
# def find_region_sub3(filename1,output_filename1,config={}):

# 	data1 = pd.read_csv(filename1,sep='\t')
# 	chrom = np.asarray(data1['chrom'])
# 	t_score = np.asarray(data1['Q2'])

# 	t_label = np.asarray(data1['label'])
# 	colnames = list(data1)

# 	thresh1 = config['thresh_select']
# 	b1 = np.where(t_score>thresh1)[0]

# 	b2 = np.where(t_label>=3)[0]
# 	b1 = np.intersect1d(b1,b2)

# 	data1 = data1.loc[b1,:]
# 	data1.reset_index(drop=True,inplace=True)
# 	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
# 	t_score = np.asarray(data1['Q2'])
# 	t_percentile = np.asarray(data1['label_1'])

# 	id1 = [int(chrom1[3:]) for chrom1 in chrom]
# 	idx_sel_list = np.column_stack((id1,serial))

# 	seq_list = generate_sequences(idx_sel_list,gap_tol=5)
# 	data_1 = output_generate_sequences(chrom,start,stop,serial,idx_sel_list,seq_list,output_filename='temp1.txt',save_mode=0)
# 	serial1, serial2 = np.asarray(data_1['serial1']), np.asarray(data_1['serial2'])
# 	num1 = len(serial1)
# 	list_1, list_2, list_3 = [], [], []
# 	for i in range(num1):
# 		b1 = np.where((serial<=serial2[i])&(serial>=serial1[i]))[0]
# 		list1 = [str(serial[i1]) for i1 in b1]
# 		list2 = ['%.4f'%(t_score1) for t_score1 in t_score[b1]]
# 		list3 = ['%.4f'%(t_percent1) for t_percent1 in t_percentile[b1]] # signal percentile
# 		d1 = ','
# 		str1 = d1.join(list1)
# 		str2 = d1.join(list2)
# 		list_1.append(str1)
# 		list_2.append(str2)
# 		list_3.append(d1.join(list3))
	
# 	data_1['loci'] = list_1
# 	data_1['score'] = list_2
# 	data_1['signal_percentile'] = list_3

# 	data_1.to_csv(output_filename1,index=False,sep='\t')

# 	return data_1

# generate serial for bed file
# def find_region_sub3_1(filename1,genome_file='',chrom_num=19):

# 	data1 = pd.read_csv(filename1,header=None,sep='\t')
# 	colnames = list(data1)

# 	if len(colnames)<5:
# 		chrom, start, stop, signal = np.asarray(data1[0]), np.asarray(data1[1]), np.asarray(data1[2]), np.asarray(data1[3])
		
# 		serial, start_vec = generate_serial_start(genome_file,chrom,start,stop,chrom_num=chrom_num,type_id=0)
# 		data1['serial'] = serial
# 		id1 = np.where(serial>=0)[0]
# 		data1 = data1.loc[id1,colnames[0:3]+['serial']+colnames[3:]]
# 		b1 = filename1.find('.bedGraph')
# 		output_filename = filename1[0:b1]+'.bed'
# 		data1 = data1.sort_values(by=['serial'])
# 		data1.to_csv(output_filename,header=False,index=False,sep='\t')
# 	# else:
# 	# 	chrom, start, stop, signal = np.asarray(data1[0]), np.asarray(data1[1]), np.asarray(data1[2]), np.asarray(data1[4])
# 	# 	serial = np.asarray(data1[3])

# 	return data1

# generate domain labels
# input: filename1: original RT data
# generate domain labels
def find_region_sub3_2(filename1):

	fields = ['chrom','start','stop','serial','signal','label',
				'q1','q2','q_1','q_2','local_peak1','local_peak2']
	num_fields = len(fields)

	data1_ori = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1_ori)
	t_num_fields = len(colnames)
	if t_num_fields>=num_fields:
		return

	if t_num_fields<10:
		data1_ori = region_quantile_signal(filename1)
		print('region_quantile_signal',data1_ori.shape)
		# return 

	colnames = list(data1_ori)
	t_num_fields = len(colnames)
	if t_num_fields<num_fields-1:
		data1_ori = region_local_signal(filename1)
		print('region_local_signal',data1_ori.shape)
		# return

	colnames = list(data1_ori)
	chrom = np.asarray(data1_ori[0])

	sample_num = len(chrom)
	label_ori = np.zeros(sample_num,dtype=np.int64)

	id1 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]
	data1 = data1_ori.loc[id1,:]
	data1.reset_index(drop=True,inplace=True)
	chrom, start, stop, signal = np.asarray(data1[0]), np.asarray(data1[1]), np.asarray(data1[2]), np.asarray(data1[4])
	serial = np.asarray(data1[3])

	sample_num1 = len(chrom)
	label1 = np.zeros(sample_num1,dtype=np.int64)

	if np.abs(np.median(signal))>0.35:
		thresh = 0.5
	else:
		thresh = 0
	b1 = np.where(signal>thresh)[0]
	b2 = np.where(signal<=thresh)[0]

	id_1 = [int(chrom1[3:]) for chrom1 in chrom[b1]]
	idx_sel_list1 = np.column_stack((id_1,serial[b1]))
	seq_list1 = generate_sequences(idx_sel_list1,gap_tol=5)

	id_2 = [int(chrom1[3:]) for chrom1 in chrom[b2]]
	idx_sel_list2 = np.column_stack((id_2,serial[b2]))
	seq_list2 = generate_sequences(idx_sel_list2,gap_tol=5)

	num1 = len(seq_list1)
	for i in range(num1):
		s1, s2 = seq_list1[i][0], seq_list1[i][1]
		t_id = b1[s1:(s2+1)]
		label1[t_id] = i+1

	num2 = len(seq_list2)
	for i in range(num2):
		s1, s2 = seq_list2[i][0], seq_list2[i][1]
		t_id = b2[s1:(s2+1)]
		label1[t_id] = -(i+1)

	label_ori[id1] = label1
	data1_ori['label'] = label_ori

	data1_ori = data1_ori.loc[:,colnames[0:5]+['label']+colnames[5:]]
	data1_ori.to_csv(filename1,header=False,index=False,sep='\t',float_format='%.6f')

	return data1_ori

# quantile of signals
# input: filename1: original RT signal
def region_quantile_signal(filename1):

	data1_ori = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1_ori)
	chrom, start, stop = np.asarray(data1_ori[0]), np.asarray(data1_ori[1]), np.asarray(data1_ori[2])

	sample_num = len(chrom)
	id1 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]
	data1 = data1_ori.loc[id1,:]
	data1.reset_index(drop=True,inplace=True)
	
	chrom, start, stop, signal = np.asarray(data1[0]), np.asarray(data1[1]), np.asarray(data1[2]), np.asarray(data1[4])
	serial = np.asarray(data1[3])

	sample_num1 = len(chrom)
	print(sample_num1)

	thresh = 0
	ranking = stats.rankdata(signal,'average')/len(signal)
	rank1 = np.zeros((sample_num1,4))
	rank1[:,0] = ranking

	b1 = np.where(signal>thresh)[0]
	rank1[b1,2]= stats.rankdata(signal[b1],'average')/len(b1)
	b2 = np.where(signal<=thresh)[0]
	rank1[b2,2]= -stats.rankdata(-signal[b2],'average')/len(b2)

	chrom_vec = np.unique(chrom)
	for chrom_id in chrom_vec:
		b1 = np.where(chrom==chrom_id)[0]
		rank1[b1,1]= stats.rankdata(signal[b1],'average')/len(b1)
		b2 = np.where(signal[b1]>thresh)[0]
		b_2 = b1[b2]
		rank1[b_2,3] = stats.rankdata(signal[b_2],'average')/len(b_2)
		b2 = np.where(signal[b1]<=thresh)[0]
		b_2 = b1[b2]
		rank1[b_2,3] = -stats.rankdata(-signal[b_2],'average')/len(b_2)

	rank_1 = np.zeros((sample_num,4))
	rank_1[id1] = rank1
	fields = ['q1','q2','q_1','q_2']
	num2 = len(fields)
	for i in range(num2):
		data1_ori[5+i] = rank_1[:,i]

	data1_ori.to_csv(filename1,index=False,header=False,sep='\t',float_format='%.7f')

	return data1_ori

# quantile of signals
# input: filename1: original RT signal
def region_local_signal(filename1):

	data1_ori = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1_ori)
	col1, col2, col3, col4, col5 = colnames[0], colnames[1], colnames[2], colnames[3], colnames[4]
	chrom = np.asarray(data1_ori[col1])

	fields = ['chrom','start','stop','serial','signal','q1','q2','q_1','q_2']
	data1_ori = data1_ori.loc[:,colnames[0:len(fields)]]

	sample_num = len(chrom)
	id1 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]
	data1 = data1_ori.loc[id1,:]
	data1.reset_index(drop=True,inplace=True)

	chrom, start, stop, signal = np.asarray(data1[col1]), np.asarray(data1[col2]), np.asarray(data1[col3]), np.asarray(data1[col5])
	serial = np.asarray(data1[col4])
	sample_num1 = len(serial)

	seq_list = generate_sequences_chrom(chrom,serial)
	value = signal
	min_value1 = np.min(value)-0.1
	print(sample_num1,np.max(value),np.min(value))
	config = {'prominence_thresh':0,'distance_thresh':20,'width_thresh':20}
	dict_2 = compare_with_regions_peak_search2(chrom,start,stop,serial,value,seq_list,thresh_vec=[min_value1],config=config)
	
	fields = ['chrom','start','stop','serial','signal','q1','q2','q_1','q_2']
	num1 = len(fields)
	key_1 = list(dict_2.keys())
	for i in range(2):
		chrom_local,start_local,stop_local,serial_local,annot_local = dict_2[key_1[0]][i] # using wavelet transformation to find peaks
		local_peak = [chrom_local,start_local,stop_local]

		id2 = mapping_Idx(serial,serial_local)
		assert np.sum(id2<0)==0
		label = np.zeros(sample_num,dtype=np.int32)
		n_local_peak = len(serial_local)
		label[id1[id2]] = np.arange(n_local_peak)+1
		data1_ori[num1+i] = label

	colnames = list(data1_ori)
	data1_ori = data1_ori.loc[:,colnames[0:(num1+2)]]
	data1_ori.to_csv(filename1,index=False,header=False,sep='\t',float_format='%.7f')

	return data1_ori

# filter regions by signal
# input: filename1: orignal RT data
#		 filename_list: region file
#		 filename_list1: RT estimation file
def region_filter_1(filename1,filename_list,filename_list1):

	fields = ['chrom','start','stop','serial','signal','label',
				'q1','q2','q_1','q_2','local_peak1','local_peak2']
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	print('region_filter_1',filename1)
	print(list(data1))
	if len(list(data1))<len(fields):
		print('generate domain labels')
		data1 = find_region_sub3_2(filename1)
	print(filename1)
	print(filename_list,filename_list1)
	colnames = list(data1)
	print(colnames)
	n_column = len(colnames)
	if n_column>len(fields):
		fields = fields + list(range(n_column-len(fields)))
	data1.columns = fields
	print(list(data1))

	num1 = len(filename_list)
	serial = np.asarray(data1['serial'])
	signal_percentile = np.asarray(data1['q_1'])
	local_peak = np.asarray(data1['local_peak2'])
	print(np.max(local_peak))
	peak_serial = serial[local_peak>0]
	thresh1 = 0.1
	thresh2 = 20

	for i in range(num1):
		filename2 = filename_list[i]
		filename2_1 = filename_list1[i]
		print(filename2,filename2_1)
		data2 = pd.read_csv(filename2,sep='\t')
		chrom1, serial1 = np.asarray(data2['chrom']), np.asarray(data2['serial'])
		start1 = np.asarray(data2['start'])
		sample_num1 = len(chrom1)

		data2_1 = pd.read_csv(filename2_1,sep='\t')
		chrom2, serial2 = np.asarray(data2_1['chrom']), np.asarray(data2_1['serial'])
		signal, predicted_signal = np.asarray(data2_1['signal']), np.asarray(data2_1['predicted_signal'])

		id1 = mapping_Idx(serial,serial1)
		t_id1 = np.where(id1<0)[0]
		print(len(t_id1),serial1[t_id1],chrom1[t_id1],start1[t_id1])
		assert np.sum(id1<0)==0
		t_id2 = np.where(id1>=0)[0]
		id1 = id1[t_id2]

		t1 = signal_percentile[id1]
		# b1 = np.where((t1>0))[0]
		b2 = np.where((t1<thresh1))[0]
		num2 = len(b2)
		# vec1 = np.zeros(num2)
		vec1 = np.zeros(sample_num1)
		for i1 in range(sample_num1):
			vec1[i1] = np.min(np.abs(peak_serial-serial1[i1]))

		id_1 = np.where(vec1>thresh2)[0]
		id_2 = np.intersect1d(id_1,b2)
		temp1 = np.asarray([num2,len(id_1),len(id_2)])
		print(temp1,temp1/sample_num1)
		label_1 = -2*np.ones(sample_num1,dtype=np.float32)
		label_3 = np.zeros(sample_num1,dtype=np.int32)
		label_2 = np.ones(sample_num1,dtype=np.float32)

		# label_1[b2] = t1[b2]
		label_1[t_id2] = t1
		label_3= vec1

		id2 = mapping_Idx(serial2,serial1)
		assert np.sum(id2<0)==0
		rank1 = stats.rankdata(signal[id2],'average')/sample_num1
		rank2 = stats.rankdata(predicted_signal[id2],'average')/sample_num1

		label_2 = rank2-rank1

		data2['label_1'] = label_1
		data2['distance'] = np.int32(label_3)
		data2['difference'] = label_2 
		data2.to_csv(filename2,index=False,sep='\t',float_format='%.7f')

	return True

# overlaping of regions
# find overlapping regions
# input: data1: position file 1
#        data2: position file 2
#        mode: 0, for each position in file 1, find all positions in file 2 overlapping with this position
#		 mode: 1, for each posiiton in file 1, find position in flie 2 that has the longest overlap with this position
def overlapping_with_regions_sub1(data1,data2,tol=0,mode=0):

	colnames1 = list(data1)
	chrom1, start1, stop1 = np.asarray(data1[colnames1[0]]), np.asarray(data1[colnames1[1]]), np.asarray(data1[colnames1[2]])
	num1 = len(chrom1)

	colnames2 = list(data2)
	chrom2, start2, stop2 = np.asarray(data2[colnames2[0]]), np.asarray(data2[colnames2[1]]), np.asarray(data2[colnames2[2]])
	num2 = len(chrom2)

	chrom_vec1 = np.unique(chrom1)
	chrom_vec2 = np.unique(chrom2)
	dict1, dict2 = dict(), dict()
	for chrom_id in chrom_vec1:
		dict1[chrom_id] = np.where(chrom1==chrom_id)[0]

	for chrom_id in chrom_vec2:
		dict2[chrom_id] = np.where(chrom2==chrom_id)[0]

	id_vec1, id_vec2 = [], []
	region_len1 = stop1-start1
	region_len2 = stop2-start2
	cnt = 0
	for chrom_id in chrom_vec1:
		if not(chrom_id in chrom_vec2):
			continue
		id1 = dict1[chrom_id]
		id2 = dict2[chrom_id]
		print(chrom_id,len(id1),len(id2))

		for t_id in id1:
			t_chrom1, t_start1, t_stop1 = chrom1[t_id], start1[t_id], stop1[t_id]
			t_start1, t_stop1 = t_start1-tol, t_stop1+tol

			b2 = np.where((start2[id2]<t_stop1)&(stop2[id2]>t_start1))[0]

			if len(b2)>0:
				id_vec1.append(t_id)
				t1 = id2[b2]
				if mode==0:
					id_vec2.append(t1)
				else:
					overlap = []
					for t_id2 in t1:
						temp1 = np.min([t_stop1-start2[t_id2],stop2[t_id2]-t_start1,region_len1[t_id],region_len2[t_id2]])
						overlap.append(temp1)
					id_1 = np.argmax(overlap)
					id_vec2.append(t1[id_1])
					
	id_vec1 = np.asarray(id_vec1)
	id_vec2 = np.asarray(id_vec2)

	return id_vec1, id_vec2

# overlaping of different runs
def find_region_sub3_overlapping(filename_list,tol=0, thresh1=2):
	
	filename1 = filename_list[0]
	data1 = pd.read_csv(filename1,sep='\t')
	label1 = np.asarray(data1['label1'])
	id1 = np.where(label1>0)[0]
	data1 = data1.loc[id1,:]
	data1.reset_index(drop=True,inplace=True)
	chrom1, start1, stop1 = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop'])
	region_num1 = len(id1)

	num1 = len(filename_list)
	bin_size = 5000
	tol1 = tol*bin_size
	list1, list2 = [], []
	label = np.zeros((region_num1,num1-1),dtype=np.int8)

	for i in range(1,num1):
		filename2 = filename_list[i]
		data2 = pd.read_csv(filename2,sep='\t')

		label2 = np.asarray(data2['label1'])
		id2 = np.where(label2>0)[0]
		data2 = data2.loc[id2,:]
		data2.reset_index(drop=True,inplace=True)
		chrom2, start2, stop2 = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop'])
		region_num2 = len(id2)
		print(region_num1,region_num2)
		
		id_vec1, id_vec2 = overlapping_with_regions_sub1(data1,data2,tol=tol1)
		list1.append(id_vec1)
		list2.append(id_vec2)
		label[id_vec1,i-1] = 1

	d1 = np.sum(label,axis=1)
	if num1==2:
		thresh1 = 0
	id_1 = np.where(d1>thresh1)[0]
	d2 = np.sum(label,axis=0)
	print(region_num1,d2,d2/region_num1)

	data1['overlap'] = (d1+1)
	for i in range(1,num1):
		data1[i+1] = label[:,i-1]

	data1 = data1.loc[id_1,:]
	data1.reset_index(drop=True,inplace=True)

	b1 = filename1.find('.txt')
	output_filename1 = filename1[0:b1]+'.%d_%d.txt'%(tol,thresh1+2)
	data1.to_csv(output_filename1,index=False,sep='\t')

	return data1, output_filename1

# compare with RT state
# input: filename1: original RT signal with state estimation
#        filename_list: list of RT estimation file on genome 2 and output filename
def compare_RT_sub1(filename_ori,filename_list,thresh=-10,config={}):
	
	# filename_centromere = config['filename_centromere']
	# file_path = config['file_path']
	# feature_id, interval, sel_id = config['feature_id'], config['interval'], config['sel_id']

	data1 = pd.read_csv(filename_ori,header=None,sep='\t')
	colnames = list(data1)
	# print(colnames)
	# data1 = data1.loc[:,colnames[0:6]+colnames[7:]]
	# data1.to_csv(filename_ori,index=False,header=False,sep='\t',float_format='%.7f')
	# print(data1.shape)

	fields = ['chrom','start','stop','serial','signal','domain','q1','q2','q_1','q_2','local_peak1','local_peak2',
					'state','group','group1']
	if len(colnames)<len(fields):
		print('state does not exist')
		return -1

	data1.columns = fields
	chrom = np.asarray(data1['chrom'])
	id1 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]
	data1 = data1.loc[id1,:]
	data1.reset_index(drop=True,inplace=True)
	signal_ori = np.asarray(data1['signal'])
	print(data1.shape)

	thresh1 = thresh
	if thresh>0:
		thresh1 = np.quantile(signal_ori,thresh)
		print(np.max(signal_ori),np.min(signal_ori),thresh1)

	if thresh1>-10:
		id2 = np.where(signal_ori>thresh1)[0]
		data1 = data1.loc[id2,:]
		data1.reset_index(drop=True,inplace=True)

	print(data1.shape)
	
	chrom, start, stop, serial_ori = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	signal_ori = np.asarray(data1['signal'])
	state, group, group1 = np.asarray(data1['state']), np.asarray(data1['group']), np.asarray(data1['group1'])

	sel_id, filename_centromere = config['sel_id'], config['filename_centromere']
	num1 = len(filename_list)
	list1 = []
	for i in range(num1):
		filename2, annot1 = filename_list[i]
		ref_data = (chrom,start,stop,serial_ori,signal_ori,[state,group,group1])
		data_1 = compare_RT_sub2(ref_data,filename2,sel_id=sel_id,filename_centromere=filename_centromere,annot=annot1)
		list1.append(data_1)

	data_2 = pd.concat(list1, axis=0, join='outer', ignore_index=True, 
			keys=None, levels=None, names=None, verify_integrity=False, copy=True)
	output_filename = config['output_filename']
	data_2.to_csv(output_filename,index=False,sep='\t',float_format='%.7f')

	return data_2

# compare with RT state
# input:chrom,start,stop,serial: positions of mapped RT estimation regions 
#		filename2: RT estimation score
def compare_RT_sub2(ref_data,filename2,sel_id='Q2',thresh=-10,filename_centromere='',annot=''):

	chrom,start,stop,serial,signal,label = ref_data
	state, group_id, group_id1 = label
	# state = state[state>0]
	# group_id = group_id[group_id!='-1']
	# group_id1 = group_id[group_id1!='-2']
	state_vec, group, group1 = np.unique(state), np.unique(group_id), np.unique(group_id1)
	state_num, group_num, group_num1 = len(state_vec), len(group), len(group1)

	data2 = pd.read_csv(filename2,sep='\t')
	if filename_centromere!='':
		data2 = region_filter(data2,filename_centromere)

	chrom2 = np.asarray(data2['chrom'])
	serial2 = np.asarray(data2['serial'])
	chrom_vec, chrom_vec2 = np.unique(chrom), np.unique(chrom2)

	id1 = mapping_Idx(serial,serial2)
	b1 = np.where(id1>=0)[0]
	id1 = id1[b1]
	data2 = data2.loc[b1,:]
	data2.reset_index(drop=True,inplace=True)

	chrom2, start2, stop2, predicted_attention = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop']), np.asarray(data2[sel_id])
	serial2 = np.asarray(data2['serial'])
	state, group_id, group_id1 = state[id1], group_id[id1], group_id1[id1]
	sample_num = len(chrom2)

	vec1 = np.zeros((state_num+1,3))
	for i in range(state_num):
		t_state = state_vec[i]
		b1 = np.where(state==t_state)[0]
		t1 = predicted_attention[b1]
		vec1[i+1] = [len(b1),np.mean(t1),np.std(t1)]

	t1 = vec1[1:(state_num+1),0]
	vec1[1:(state_num+1),0] = t1/sample_num

	vec2, vec3 = np.zeros((group_num,3)), np.zeros((group_num1,3))
	for i in range(group_num):
		b1 = np.where(group_id==group[i])[0]
		t1 = predicted_attention[b1]
		vec2[i] = [len(b1),np.mean(t1),np.std(t1)]
	vec2[:,0] = vec2[:,0]/sample_num

	for i in range(group_num1):
		b1 = np.where(group_id1==group1[i])[0]
		t1 = predicted_attention[b1]
		vec3[i] = [len(b1),np.mean(t1),np.std(t1)]
	vec3[:,0] = vec3[:,0]/sample_num

	# mean_value = np.hstack((state_vec[:,np.newaxis],vec1[1:]))
	mean_value1, std1 = np.mean(predicted_attention), np.std(predicted_attention)
	print(thresh,mean_value1,std1)
	vec1[0] = [sample_num,mean_value1,std1]
	t1 = np.vstack((vec1,vec2,vec3))
	fields = ['run_id','label','percentage','mean_value','std']
	num1 = len(fields)
	data_1 = pd.DataFrame(columns=fields)
	data_1['run_id'] = [annot]*len(t1)
	data_1['label'] = [-2]+list(state_vec)+list(group)+list(group1)
	for i in range(2,num1):
		t_column = fields[i]
		data_1[fields[i]] = t1[:,i-2]

	return data_1

# compare with rmsk
def compare_rmsk_sub1_pre(filename1,filename2,output_filename):

	data1 = pd.read_csv(filename1,sep='\t')
	# bin, swScore, genoName, genoStart, genoEnd, strand, repName, repClass, repFamily
	colnames = list(data1)
	data2 = data1.loc[:,['genoName','genoStart','genoEnd','strand','repName','repClass','repFamily']]
	repName = np.unique(data2['repName'])
	repClass = np.unique(data2['repClass'])
	repFamily = np.unique(data2['repFamily'])
	print(data2.shape)

	b1 = filename1.find('.txt')
	output_filename1 = filename1[0:b1]+'.1.txt'
	data2.to_csv(output_filename1,index=False,header=False,sep='\t')

	output_filename1 = filename1[0:b1]+'.repName.txt'
	np.savetxt(output_filename1,repName,fmt='%s',delimiter='\t')

	output_filename1 = filename1[0:b1]+'.repClass.txt'
	np.savetxt(output_filename1,repClass,fmt='%s',delimiter='\t')

	output_filename1 = filename1[0:b1]+'.repFamily.txt'
	np.savetxt(output_filename1,repFamily,fmt='%s',delimiter='\t')

	return data2

# compare with rmsk
# find overlapping of each rmsk family with each genomic locus
# input: filename1: rmsk file
# 		 filename2: genomic loci file
def compare_rmsk_sub1(filename1,filename2,output_file_path,chrom_num=22):

	fields = ['genoName','genoStart','genoEnd','strand','repName','repClass','repFamily']
	data1 = pd.read_csv(filename1,header=None,sep='\t',names=fields)
	# bin, swScore, genoName, genoStart, genoEnd, strand, repName, repClass, repFamily
	colnames = list(data1)
	# data_1 = data1.loc[:,['genoName','genoStart','genoEnd','strand','repName','repClass','repFamily']]
	repName, repClass, repFamily = np.asarray(data1['repName']), np.asarray(data1['repClass']), np.asarray(data1['repFamily'])
	repName_vec, repClass_vec, repFamily_vec = np.unique(repName), np.unique(repClass), np.unique(repFamily) 
	print(data1.shape)

	chrom_vec1 = np.arange(1,chrom_num+1)
	chrom_vec = ['chr%d'%(i) for i in chrom_vec1]
	chrom_vec = np.asarray(chrom_vec)
	chrom, start, stop = np.asarray(data1[colnames[0]]), np.asarray(data1[colnames[1]]), np.asarray(data1[colnames[2]])

	data2 = pd.read_csv(filename2,header=None,sep='\t')
	ref_chrom, ref_start, ref_stop = np.asarray(data2[0]), np.asarray(data2[1]), np.asarray(data2[2])
	serial = np.asarray(data2[3])
	colnames2 = list(data2)

	region_len1 = stop-start
	region_len2 = ref_stop-ref_start

	chrom_dict1 = dict()
	chrom_dict2 = dict()
	region_num1, region_num2 = 0, 0
	for i in range(chrom_num):
		t_chrom = 'chr%d'%(i+1)
		b1 = np.where(chrom==t_chrom)[0]
		chrom_dict1[t_chrom] = b1
		region_num1 += len(b1)

		b2 = np.where(ref_chrom==t_chrom)[0]
		chrom_dict2[t_chrom] = b2
		region_num2 += len(b2)
		print(t_chrom,len(b1),len(b2))
	
	print(region_num1, region_num2)
	print('repFamily',len(repFamily),repFamily)

	repFamily_dict = dict()
	list1 = []
	for t_repFamily in repFamily_vec:
		b1 = np.where(repFamily==t_repFamily)[0]
		print(t_repFamily,len(b1))
		t_chrom, t_start, t_stop = chrom[b1], start[b1], stop[b1]
		t_repClass = repClass[b1]
		list1.append([t_repFamily,t_repClass[0]])
		list2 = []

		for t_chrom1 in chrom_vec:
			id1 = np.where(t_chrom==t_chrom1)[0]
			if len(id1)==0:
				continue

			id2 = chrom_dict2[t_chrom1]
			print(t_repFamily,t_chrom1,len(id1))

			for t_id1 in id1:
				t_start1, t_stop1 = t_start[t_id1], t_stop[t_id1]
				b2 = np.where((ref_start[id2]<t_stop1)&(ref_stop[id2]>t_start1))[0]
				if len(b2)>0:
					b2 = id2[b2]
					for t_id2 in b2:
						overlap = np.min([t_stop1-ref_start[t_id2],ref_stop[t_id2]-t_start1,t_stop1-t_start1,region_len2[t_id2]])
						list2.append([serial[t_id2],overlap,t_start1,t_stop1])

		if len(list2)==0:
			continue

		list2 = np.asarray(list2)
		b_1 = t_repFamily.find('?')
		if b_1>=0:
			t_repFamily1 = t_repFamily[0:b_1]+'_sub1'
		else:
			t_repFamily1 = t_repFamily

		t_repClass1 = t_repClass[0]
		b_2 = t_repClass1.find('?')
		if b_2>=0:
			t_repClass1 = t_repClass1[0:b_2]+'_sub1'

		id_1 = mapping_Idx(serial,list2[:,0])
		assert np.sum(id1<0)==0

		t_data2 = data2.loc[id_1,:]
		t_overlap = list2[:,1]
		t_data2['overlap'] = t_overlap
		t_data2['start1'], t_data2['stop1'] = list2[:,2], list2[:,3]
		t_data2 = t_data2.sort_values(by=[colnames2[3]])
		print(np.max(t_overlap),np.min(t_overlap),np.median(t_overlap))

		output_filename1 = '%s/%s.%s.overlap.txt'%(output_file_path,t_repFamily1,t_repClass1)
		t_data2.to_csv(output_filename1,index=False,header=False,sep='\t')

	output_filename2 = '%s/repName_repFamily.txt'%(output_file_path)
	data_1 = pd.DataFrame(columns=['repFamily','repClass'],data=list1)
	data_1.to_csv(output_filename2,index=False,sep='\t')
		
	return data_1

def region_filter(data1,filename_centromere):

	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	id1 = select_idx_centromere(chrom,start,stop,filename_centromere)
	print('select_idx_centromere', len(chrom), len(id1), len(id1)/len(chrom))
	data1 = data1.loc[id1,:]
	data1.reset_index(drop=True,inplace=True)

	return data1

# input is probability
class Sample_Concrete1(Layer):
	"""
	Layer for sample Concrete / Gumbel-Softmax variables. 

	"""
	def __init__(self, tau0, k, n_steps, type_id, **kwargs): 
	# def __init__(self, tau0, k, n_steps): 
		self.tau0 = tau0
		self.k = k
		self.n_steps = n_steps
		self.type_id = type_id
		super(Sample_Concrete1, self).__init__(**kwargs)

	def call(self, logits):
		logits_ = K.permute_dimensions(logits, (0,2,1))
		#[batchsize, 1, MAX_SENTS]

		unif_shape = tf.shape(logits_)[0]
		uniform = tf.random.uniform(shape =(unif_shape, self.k, self.n_steps), 
			minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
			maxval = 1.0)

		gumbel = - K.log(-K.log(uniform))
		eps = tf.compat.v1.keras.backend.constant(1e-12)
		# print('eps:', eps)
		# noisy_logits = (gumbel + logits_)/self.tau0
		# logits_ = K.log(logits_) # the input is probability
		if self.type_id==2:
			logits_ = -K.log(-K.log(logits_ + eps))	# the input is probability
		elif self.type_id==3:
			logits_ = K.log(logits_ + eps) # the input is probability
		# elif self.type_id==5:
		# 	logits_ = -logits_
		elif self.type_id==5:
			eps1 = tf.compat.v1.keras.backend.constant(1+1e-12)
			# x = Lambda(lambda x: x * 2)(layer)
			logits_ = K.log(logits_ + eps1)
		else:
			pass
		noisy_logits = (gumbel + logits_)/self.tau0
		samples = K.softmax(noisy_logits)
		samples = K.max(samples, axis = 1)
		samples = K.expand_dims(samples, -1)

		discrete_logits = K.one_hot(K.argmax(logits_,axis=-1), num_classes = self.n_steps)
		discrete_logits = K.permute_dimensions(discrete_logits,(0,2,1))

		# return K.in_train_phase(samples, discrete_logits)
		return samples

	def compute_output_shape(self, input_shape):
		return input_shape

def filter_region_signal_sub2_align(region_data1,region_data2,thresh_vec=[0.5]):

	# species 1
	# chrom1,start1,stop1,serial1 = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	# signal1 = np.asarray(data1['signal'])
	# sample_num1 = len(chrom1)
	colnames = list(region_data1)
	region_chrom1, region_start1, region_stop1 = np.asarray(region_data1[colnames[0]]), np.asarray(region_data1[colnames[1]]), np.asarray(region_data1[colnames[2]])
	region_serial1 = np.asarray(region_data1[colnames[3]])
	region_label1 = np.asarray(region_data1['label'])
	region_num1 = len(region_chrom1)
	print(region_num1)

	# species 2
	# chrom2,start2,stop2,serial2 = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop']), np.asarray(data2['serial'])
	# signal2 = np.asarray(data2['signal'])
	# sample_num2 = len(chrom2)
	colnames = list(region_data2)
	region_chrom2, region_start2, region_stop2 = np.asarray(region_data2[colnames[0]]), np.asarray(region_data2[colnames[1]]), np.asarray(region_data2[colnames[2]])
	region_serial2 = np.asarray(region_data2[colnames[3]])
	region_label2 = np.asarray(region_data2['label'])
	region_num2 = len(region_chrom2)
	print(region_num2)
	print(region_serial1,region_serial2)

	id_vec1 = np.zeros(region_num2,dtype=bool)
	region_chrom_1 = np.asarray(['chr22']*region_num2)
	region_pos_1 = np.zeros((region_num2,2),dtype=np.int64)
	region_label_1 = np.zeros(region_num2,dtype=np.float32)
	thresh1 = 200
	for i in range(region_num1):
		t_serial = region_serial1[i]
		t_label1 = region_label1[i]
		b1 = np.where(region_serial2==t_serial)[0]
		t_chrom2 = region_chrom2[b1]
		t_label2 = region_label2[b1]
		num1 = len(b1)

		if num1>0:
			# print(t_serial,num1)
			region_chrom_1[b1] = region_chrom1[i]
			region_pos_1[b1] = [region_start1[i],region_stop1[i]]
			region_label_1[b1] = t_label1

			for j in range(num1):
				try:
					t_chrom_id = t_chrom2[j]
					t_chrom_id = int(t_chrom_id[3:])
				except:
					continue
				id1 = b1[j]
				if region_stop2[id1]-region_start2[id1]<thresh1:
					continue

				temp1 = (t_label1>0.5)&(t_label2[j]>0.5)
				temp2 = (t_label1<0.5)&(t_label2[j]<0.5)
				id_vec1[id1] = (temp1|temp2)

	id1 = np.where(id_vec1>0)[0]
	region_data2['chrom1'], region_data2['start1'], region_data2['stop1'] = region_chrom_1,region_pos_1[:,0],region_pos_1[:,1]
	region_data2['label1'] = region_label_1
	region_data2 = region_data2.loc[id1,:]
	region_data2.reset_index(drop=True,inplace=True)
	
	# region_data2.sort_values(by=[colnames[0]])

	return region_data2

def filter_region_signal_sub1(data1,region_data,thresh_vec):

	chrom,start,stop,serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	signal = np.asarray(data1['signal'])
	sample_num = len(chrom)
	colnames = list(region_data)
	region_chrom, region_start, region_stop = np.asarray(region_data[colnames[0]]), np.asarray(region_data[colnames[1]]), np.asarray(region_data[colnames[2]])
	region_num = len(region_chrom)
	print(sample_num,region_num)

	dict1 = dict()
	dict2 = dict()
	thresh_num = len(thresh_vec)
	flag_vec = np.zeros((region_num,thresh_num),dtype=np.int8)
	thresh_vec1 = np.quantile(signal,thresh_vec)
	thresh_vec1 = list(thresh_vec1)+[np.max(signal)+1e-05]
	t_vec1 = -np.ones((region_num,5),dtype=np.float32)
	for i in range(region_num):
		b2 = np.where((chrom==region_chrom[i])&(start<region_stop[i])&(stop>region_start[i]))[0]
		if len(b2)>0:
			t_value = [np.max(signal[b2]),np.min(signal[b2]),np.median(signal[b2]),np.mean(signal[b2])]
			ranking = np.sum(signal<t_value[2])/sample_num
			t_vec1[i] = t_value+[ranking]
			for l in range(thresh_num):
				thresh_1, thresh_2 = thresh_vec1[l], thresh_vec1[l+1]
				if (t_value[2]>=thresh_1) and (t_value[2]<thresh_2):
					flag_vec[i,l] = 1

	for l in range(thresh_num):
		thresh_1, thresh_2 = thresh_vec1[l], thresh_vec1[l+1]
		id1 = np.where(flag_vec[:,l]==1)[0]
		dict1[thresh_vec[l]] = id1

		id2 = np.where((signal>=thresh_1)&(signal<thresh_2))[0]
		print(id2)
		dict2[thresh_vec[l]] = id2
		print(thresh_1,thresh_2,len(id1),len(id1)/region_num)
		print(thresh_1,thresh_2,len(id2),len(id2)/sample_num)

	fields = ['max','min','median','mean','label']
	num1 = len(fields)
	region_data['serial'] = np.arange(1,region_num+1)
	for i in range(num1):
		t_id = fields[i]
		region_data[t_id] = t_vec1[:,i]

	# label1 = -np.ones(region_num,dtype=np.float32)
	# for i in range(thresh_num):
	# 	id1 = dict1[thresh_vec[i]]
	# 	label1[id1] = thresh_vec[i]
	# region_data['label'] = label1

	return dict1, dict2, region_data

def filter_region_signal(filename1,region_filename,output_filename,thresh_vec):

	data1 = pd.read_csv(filename1,sep='\t')
	region_data = pd.read_csv(region_filename,header=None,sep='\t')

	# thresh_vec = [0,0.25,0.5,0.75]
	dict1, dict2, region_data = filter_region_signal_sub1(data1,region_data,thresh_vec)
	
	region_data.to_csv(output_filename,index=False,sep='\t')

	return True

def filter_region_signal_align(region_filename1,region_filename2,output_filename):

	region_data1 = pd.read_csv(region_filename1,sep='\t')
	region_data2_ori = pd.read_csv(region_filename2,sep='\t')
	region_data2 = filter_region_signal_sub2_align(region_data1,region_data2_ori)
	print(len(region_data2_ori),len(region_data2))

	region_data2.to_csv(output_filename,index=False,header=False,sep='\t')

def find_serial(chrom,chrom_num,chrom_vec=[],type_id=0):

	if len(chrom_vec)==0:
		for i in range(1,chrom_num+1):
			chrom_vec.append('chr%d'%(i))

		if type_id==1:
			chrom_vec += ['chrX']
			chrom_vec += ['chrY']

	serial_vec = []
	for t_chrom in chrom_vec:
		b1 = np.where(chrom==t_chrom)[0]
		if len(b1)>0:
			serial_vec.extend(b1)

	print(len(chrom),chrom_vec,len(serial_vec))

	return np.asarray(serial_vec)

# training and test chromosome list
def find_list(train_vec,chrom_vec,test_vec=[]):
	vec3_pre = train_vec
	vec3, vec3_1 = [], []
	if len(test_vec)==0:
		for t_value1 in vec3_pre:
			t_list1_1, t_list2_1 = [],[]
			for t_list1 in t_value1:
				str1 = [str(t1) for t1 in t_list1]
				t_list1_1.append(','.join(str1))
				t_list2 = np.sort(list(set(chrom_vec)-set(t_list1)))
				str2 = [str(t2) for t2 in t_list2]
				t_list2_1.append(','.join(str2))
			vec3.append(t_list1_1)
			vec3_1.append(t_list2_1)
	else:
		vec3_pre_1 = test_vec
		vec3, vec3_1 = [], []	
		for (t_value1,t_value2) in zip(vec3_pre,vec3_pre_1):
			t_list1_1 = []
			for t_list1 in t_value1:
				str1 = [str(t1) for t1 in t_list1]
				t_list1_1.append(','.join(str1))
			t_list2_1 = []
			for t_list2 in t_value2:
				str2 = [str(t2) for t2 in t_list2]
				t_list2_1.append(','.join(str2))
			vec3.append(t_list1_1)
			vec3_1.append(t_list2_1)

	return vec3, vec3_1

def load_data_sub1(file_path,label_ID,sel_id1,id1):

	label_id,label_serial,t_filename,local_id = label_ID[sel_id1]
	# id1, t_filename = ID
	t_filename1 = '%s/%s'%(file_path,t_filename)
	start1 = time.time()
	with h5py.File(t_filename1,'r') as fid:
		# serial2 = fid["serial"][:]
		x_train = fid["vec"][:]
		#print(x_train.shape)
	stop1 = time.time()
	print(id1,sel_id1)

	return (x_train,sel_id1)

# load training data
def load_data_1(file_path,label_ID,id_vec):

	queue1 = mp.Queue()
	num1 = len(id_vec)

	print("processes")
	start = time.time()
	processes = [mp.Process(target=load_data_sub1, 
				args=(file_path,label_ID,id_vec[id1],id1)) for id1 in range(num1)]

	# Run processes
	for p in processes:
		p.start()

	results = [queue1.get() for p in processes]
	print(len(results))		

	# Exit the completed processes
	print("join")
	for p in processes:
		p.join()

	end = time.time()
	print("use time load vectors: %s %s %s"%(start, end, end-start))

	list1 = results

	return list1

# load kmer frequency feature
def prep_data_sequence_kmer(filename1,kmer_size,output_filename=''):

	kmer_dict1 = kmer_dict(kmer_size)
	f_list, f_mtx, serial = load_seq_2(filename1, kmer_size, kmer_dict1, sel_idx=[])

	return f_list, serial

# load kmer frequency feature
def prep_data_sequence_kmer_chrom(filename1,filename2,kmer_size,chrom_vec=[],filename_prefix='',save_mode=1,region_size=1):

	kmer_dict1 = kmer_dict(kmer_size)
	file1 = pd.read_csv(filename1,sep='\t')
	seq1 = np.asarray(file1['seq'])
	serial1 = np.asarray(file1['serial'])

	file2 = pd.read_csv(filename2,header=None,sep='\t')
	chrom, start, stop, ref_serial = np.asarray(file2[0]), np.asarray(file2[1]), np.asarray(file2[2]), np.asarray(file2[3])

	n1, n2 = len(serial1), len(ref_serial)
	if n1!=n2:
		print('error!',n1,n2)
		return

	b1 = (serial1!=ref_serial)
	count1 = np.sum(b1)
	if count1 > 0:
		print('error!',count1,n1,n2)
		return

	list1 = []
	chrom_num = len(chrom_vec)
	for chrom_id in chrom_vec:
		sel_idx = np.where(chrom=='chr%d'%(chrom_id))[0]

		if len(sel_idx)==0:
			continue

		t_serial = ref_serial[sel_idx]
		id1 = mapping_Idx(serial1,t_serial)
		b1 = (id1>=0)
		n1,n2 = np.sum(b1), len(t_serial)
		if n1!=n2:
			print('error!',chrom_id,n1,n2)
		sel_idx = id1[b1]
		# f_list, chrom_id = load_seq_2_kmer1(seq1,serial1,kmer_size,kmer_dict1,chrom_id=chrom_id,sel_idx=sel_idx)
		list1.append(sel_idx)

	feature_dim = len(kmer_dict1)
	num_region = len(serial1)
	if region_size<=1:
		f_list = np.zeros((num_region,feature_dim))
	else:
		num_subregion = int(np.ceil(len(seq1[0])/region_size))
		f_list = np.zeros((num_region,num_subregion,feature_dim))
		f_list = [None]*num_region

	queue1 = mp.Queue()
	# chrom_vec = range(20,22)
	# chrom_vec = [1,9,10]
	# chrom_vec = range(1,6)

	print("processes")
	start = time.time()
	# processes = [mp.Process(target=self._compute_posteriors_graph_test, args=(len_vec, X, region_id,self.posteriors_test,self.posteriors_test1,self.queue)) for region_id in range(0,num_region)]
	if region_size<2:
		processes = [mp.Process(target=load_seq_2_kmer1, 
						args=(seq1, serial1, kmer_size, kmer_dict1, chrom_vec[i], list1[i], queue1, 
								filename_prefix, save_mode)) for i in range(chrom_num)]
	else:
		processes = [mp.Process(target=load_seq_2_kmer1_subregion, 
						args=(seq1, serial1, kmer_size, kmer_dict1, chrom_vec[i], list1[i], reigon_size, 
								queue1, filename_prefix, save_mode)) for i in range(chrom_num)]

	# Run processes
	for p in processes:
		p.start()

	results = [queue1.get() for p in processes]
	print(len(results))		

	# Exit the completed processes
	print("join")
	for p in processes:
		p.join()

	end = time.time()
	print("use time load chromosomes: %s %s %s"%(start, end, end-start))

	chrom_num = len(chrom_vec)
	chrom_vec1 = np.zeros(chrom_num)
	output_filename_list = []

	if save_mode==1:
		for i in range(0,chrom_num):
			vec1 = results[i]
			chrom_id, sel_idx = vec1[0], vec1[1]
			chrom_vec1[i] = chrom_id

			if save_mode==1:
				output_filename1 = vec1[-1]
				output_filename_list.append(output_filename1)

				# sorted_idx = np.argsort(chrom_vec1)
				# output_filename_list = np.asarray(output_filename_list)
				# output_filename_list = output_filename_list[sorted_idx]

				with h5py.File(output_filename1,'r') as fid:
					t_serial = fid["serial"][:]
					fmtx = fid["vec"][:]
			else:
				t_serial, fmtx = vec1[2], vec1[3]

			id1 = mapping_Idx(serial1,t_serial)
			b1 = (id1!=sel_idx)
			count1 = np.sum(b1)
			if count1>0:
				print('error!',chrom_id)
			print(chrom_id,count1,len(sel_idx))
			f_list[sel_idx] = fmtx

	f_list = np.asarray(f_list)
	print('kmer feature',f_list.shape)
	# output_filename = '%s_kmer%d.h5'(filename_prefix,kmer_size)
	# with h5py.File(output_filename,'w') as fid:
	# 	fid.create_dataset("serial", data=ref_serial, compression="gzip")
	# 	fid.create_dataset("vec", data=f_list, compression="gzip")

	return f_list, serial1

# merge files
def test_1(filename_list,output_filename_list):

	num1 = len(filename_list)
	chrom_numList = []
	for i in range(num1):
		filename_list1 = filename_list[i]
		output_filename1 = output_filename_list[i]

		data2, t_chrom_numList = select_region1_merge(filename_list1,output_filename1,type_id1=0,type_id2=1)
		chrom_numList.append(t_chrom_numList)

	return chrom_numList

def train_test_index_chromosome(x,y,group_label,chrom_idvec,train_chromvec,test_chromvec,ratio=0.1):
	
	id_train_1, id_test = [], []

	for chrom_id in train_chromvec:
		id1 = np.where(chrom_idvec==chrom_id)[0]
		id_train_1.extend(id1)

	for chrom_id in test_chromvec:
		id1 = np.where(chrom_idvec==chrom_id)[0]
		id_test.extend(id1)

	id_train_1, id_test = np.asarray(id_train_1), np.asarray(id_test)
	id_train, id_valid, y_train, y_valid, id_train1, id_valid1 = train_test_split_group(id_train_1,y[id_train_1],group_label[id_train_1],ratio=ratio)

	return id_train, id_valid, id_test

def train_test_split_group(x,y,group_label,ratio=0.2):
	
	group_label_vec = np.unique(group_label)
	num1 = len(group_label_vec)
	id1 = np.arange(num1)

	id_1, id_2, y_train_group, y_valid_group = train_test_split(id1, group_label_vec, test_size=ratio, shuffle=True, random_state=42)

	# num2 = x.shape[0]
	# id_train_1 = np.zeros(num2,dtype=bool)
	# for t_id in id_1:
	# 	id_train_1 = id_train_1|(group_label==group_label_vec[t_id])
	# id_train = np.where(id_train_1>0)[0]

	id_train, id_valid = [], []
	for t_id in id_1:
		id_train.extend(np.where(group_label==group_label_vec[t_id])[0])

	for t_id in id_2:
		id_valid.extend(np.where(group_label==group_label_vec[t_id])[0])

	id_train, id_valid = np.asarray(id_train), np.asarray(id_valid)
	x_train, x_valid = x[id_train], x[id_valid]
	y_train, y_valid = y[id_train], y[id_valid]

	return x_train, x_valid, y_train, y_valid, id_train, id_valid
	
# merge files
def run_1_merge(run_idlist,config):

	if 'file_path' in config:
		file_path = config['file_path']
	else:
		file_path = './'

	type_id, type_id1 = config['type_id_1'], config['type_id1_1']
	feature_id1 = config['feature_id1']

	filename_list, output_filename_list = [], []
	num1 = len(run_idlist)
	vec1 = np.zeros(num1,dtype=np.int32)
	chrom_numList = []

	for pair1 in run_idlist:
		run_id1, run_id2, method1 = pair1
		if 'filename_list1' in config:
			print('filename_list1',config['filename_list1'][run_id1])
			filename_list_1 = config['filename_list1'][run_id1]
			filename_list1, output_filename = filename_list_1[0], filename_list_1[1]
		else:
			filename1 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id1,method1)
			filename2 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id2,method1)
			output_filename = '%s/test_vec2_%d_%d_[%d].%d_%d.1.txt'%(file_path,run_id1,method1,feature_id1,type_id,type_id1)
			filename_list1 = [filename1,filename2]

		filename_list.append(filename_list1)
		output_filename_list.append(output_filename)

		if os.path.exists(output_filename)==False:
			data2, t_chrom_numList = select_region1_merge(filename_list1,output_filename,type_id1=0,type_id2=1)
		else:
			num1 = len(filename_list1)
			t_chrom_numList = []
			if type_id1==1:
				vec1 = list(range(num1-1,-1,-1))
			else:
				vec1 = list(range(num1))
			for i in vec1:
				data_2 = pd.read_csv(filename_list1[i],sep='\t')
				t_chrom_numList.append(np.unique(data_2['chrom']))

		print(pair1,output_filename,t_chrom_numList)
		chrom_numList.append(t_chrom_numList)

	return filename_list, output_filename_list, chrom_numList

# merge estimation files
def test_merge_1(run_idlist,output_filename,config,mode=1):

	if 'file_path' in config:
		file_path = config['file_path']
	else:
		file_path = './'

	# config.update({'type_id_1':1, 'type_id1_1':0, 'feature_id1':feature_id1})
	# run_idlist = list(np.vstack((t_list1,t_list2,t_list3)).T)

	cell_idtype, method1 = config['cell_type1'], config['method1']
	filename_list, output_filename_list, chrom_numList = run_1_merge(run_idlist,config)

	run_idlist = np.asarray(run_idlist)
	run_idlist1 = run_idlist[:,0]
	list1 = []

	print(run_idlist1,output_filename_list,chrom_numList)
	for (run_id,filename1,t_chrom_numList) in zip(run_idlist1,output_filename_list,chrom_numList):
		config.update({'chrom_vec1_pre':t_chrom_numList[0]})
		config.update({'cell_type1':cell_idtype,'method1':method1})
		data1 = compute_mean_std(run_id, filename1, config)
		print(run_id,t_chrom_numList,data1.shape)
		list1.append(data1)

	data_1 = pd.concat(list1, axis=0, join='outer', ignore_index=True, 
				keys=None, levels=None, names=None, verify_integrity=False, copy=True)

	# if mode==0:
	# 	data_1.to_csv(output_filename,index=False,sep='\t')
	# else:
	# 	data_pre = pd.read_csv(output_filename,sep='\t')
	# 	data_2 = pd.concat([data_pre,data_1], axis=0, join='outer', ignore_index=True, 
	# 		keys=None, levels=None, names=None, verify_integrity=False, copy=True)
	# 	data_2.to_csv(output_filename,index=False,sep='\t')

	if (os.path.exists(output_filename)==True) and (mode==1):
		data_pre = pd.read_csv(output_filename,sep='\t')
		data_2 = pd.concat([data_pre,data_1], axis=0, join='outer', ignore_index=True, 
				keys=None, levels=None, names=None, verify_integrity=False, copy=True)
		data_2.to_csv(output_filename,index=False,sep='\t',float_format='%.6f')
	else:
		data_1.to_csv(output_filename,index=False,sep='\t',float_format='%.6f')

	return True

def table_format(filename1,pre_colnames=[],type_id1=0):

	data1 = pd.read_csv(filename1,sep='\t')
	colnames = list(data1)

	if type_id1==0:
		if 'method' in colnames:
			local_colnames = colnames[4:]
		else:
			local_colnames = colnames[2:]

		if 'train_chrNum' in colnames:
			local_colnames = local_colnames[0:-1]

		if len(pre_colnames)==0:
			pre_colnames = list(np.setdiff1d(colnames,local_colnames,assume_unique=True))
	else:
		num1 = len(pre_colnames)
		local_colnames = colnames[num1:]

	data_1 = data1.loc[:,local_colnames]
	data_1 = np.asarray(data_1)
	data_2 = data1.loc[:,pre_colnames]
	data_2 = np.asarray(data_2)

	num_sample, sel_num = data_1.shape
	vec1 = np.ravel(data_1)
	vec2 = np.tile(local_colnames,num_sample)

	print(data_1.shape,data_2.shape)
	data_2 = np.repeat(data_2,sel_num,axis=0)
	print(data_2.shape)

	data3 = pd.DataFrame(columns=pre_colnames,data=data_2)
	data3['value'] = vec1
	data3['metrics'] = vec2

	id1 = filename1.find('txt')
	filename2 = filename1[0:id1]+'copy1.txt'

	data3.to_csv(filename2,index=False,sep='\t',float_format='%.7f')

	return True

def select_idx_centromere(chrom,start,stop,filename_centromere=''):

	centromere = pd.read_csv(filename_centromere,header=None,sep='\t')
	chrom1, start1, stop1 = np.asarray(centromere[0]), np.asarray(centromere[1]), np.asarray(centromere[2])

	num1 = len(chrom1)
	list1 = []
	for i in range(num1):
		t_chrom1, t_start1, t_stop1 = chrom1[i], start1[i], stop1[i]
		b1 = np.where((chrom==t_chrom1)&(start<t_stop1)&(stop>t_start1))[0]
		list1.extend(b1)

	list1 = np.asarray(list1)
	id1 = np.arange(len(chrom))
	id1 = np.setdiff1d(id1,list1)
	print(len(id1),len(list1),len(chrom1))

	return id1

# input: estimated attention, type_id: training, validation, or test data
# output: ranking of attention
def select_region1_sub(filename,type_id,data1=[],filename_centromere=''):

	if len(data1)==0:
		data1 = pd.read_csv(filename,sep='\t')
	colnames = list(data1)

	if filename_centromere!='':
		chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
		id1 = select_idx_centromere(chrom,start,stop,filename_centromere)
		print('select_idx_centromere', len(chrom), len(id1), len(id1)/len(chrom))
		data1 = data1.loc[id1,:]
		# data1.reset_index(drop=True,inplace=True)
		data1.reset_index(drop=True,inplace=True)

	# chrom	start	stop	serial	signal	predicted_signal	predicted_attention
	chrom, start, serial = data1['chrom'], data1['start'], data1['serial']
	chrom, start, serial = np.asarray(chrom), np.asarray(start), np.asarray(serial)

	if 'predicted_attention' in data1:
		predicted_attention = data1['predicted_attention']
	else:
		predicted_attention = np.zeros(len(chrom),dtype=np.float32)
	predicted_attention = np.asarray(predicted_attention)

	ranking = stats.rankdata(predicted_attention,'average')/len(predicted_attention)
	rank1 = np.zeros((len(predicted_attention),2))
	rank1[:,0] = ranking

	flag1 = 0
	if 'predicted_attention1' in colnames:
		flag1 = 1
		predicted_attention1 = np.asarray(data1['predicted_attention1'])

		ranking1 = stats.rankdata(predicted_attention1,'average')/len(predicted_attention1)
		rank_1 = np.zeros((len(predicted_attention1),2))
		rank_1[:,0] = ranking1

	chrom_vec = np.unique(chrom)
	for t_chrom in chrom_vec:
		b1 = np.where(chrom==t_chrom)[0]
		t_attention = predicted_attention[b1]
		t_ranking = stats.rankdata(t_attention,'average')/len(t_attention)
		rank1[b1,1] = t_ranking

		if flag1==1:
			t_attention1 = predicted_attention1[b1]
			t_ranking1 = stats.rankdata(t_attention1,'average')/len(t_attention1)
			rank_1[b1,1] = t_ranking1

	data1['Q1'] = rank1[:,0]	# rank across all the included chromosomes
	data1['Q2'] = rank1[:,1]	# rank by each chromosome
	data1['typeId'] = np.int8(type_id*np.ones(len(rank1)))

	if flag1==1:
		data1['Q1_1'] = rank_1[:,0]	# rank across all the included chromosomes
		data1['Q2_1'] = rank_1[:,1]	# rank by each chromosome
		t1 = np.hstack((rank1,rank_1))
		data1['Q1(2)'] = np.max(t1[:,[0,2]],axis=1)
		data1['Q2(2)'] = np.max(t1[:,[1,3]],axis=1)

	return data1,chrom_vec

# merge estimated attention from different training/test splits
# type_id1: chromosome order; type_id2: training: 0, test: 1, valid: 2
# filename_centromere: centromere file
def select_region1_merge(filename_list,output_filename,type_id1=0,type_id2=1,filename_centromere=''):

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
		data1, chrom_vec = select_region1_sub(filename1,type_id2,filename_centromere=filename_centromere)
		print(filename1,len(data1))
		# list1.append(data1)
		# if i==0:
		# 	serial1 = np.asarray(data1['serial'])
		print(serial1)
		t_serial = np.asarray(data1['serial'],dtype=np.int64)
		t_serial2 = np.setdiff1d(t_serial,serial1)
		serial1 = np.union1d(serial1,t_serial)
		print(len(t_serial),len(t_serial2),len(serial1))
		id1 = mapping_Idx(t_serial,t_serial2)
		colnames = list(data1)
		data1 = data1.loc[id1,colnames]
		list1.append(data1)
		chrom_numList.append(chrom_vec)

	data2 = pd.concat(list1, axis=0, join='outer', ignore_index=True, 
				keys=None, levels=None, names=None, verify_integrity=False, copy=True)
	print('sort')
	data2 = data2.sort_values(by=['serial'])
	data2.to_csv(output_filename,index=False,sep='\t',float_format='%.6f')

	return data2, chrom_numList

# sample region without replacement
def sample_region_sub1(sample_num,sel_num,sample_weight,sel_ratio=1):
	
	if sel_ratio!=1:
		sel_num = int(sample_num*sel_ratio)

	select_serial = -np.ones(sel_num,dtype=np.int32)
	# prob1 = np.random.rand(sample_num)
	# select_serial = np.where(prob1<=sample_weight)[0]
	vec1 = np.asarray(range(0,sample_num))
	limit1 = sel_num*100
	i, cnt1 = 0, 0
	while i < sel_num:
		i1 = vec1[np.random.randint(0,sel_num-i)]
		prob1 = np.random.rand()
		cnt1 = cnt1+1
		if prob1<=sample_weight[i1]:
			select_serial[i] = i1
			i = i+1
			vec1 = np.setdiff1d(vec1,select_serial[0:i],assume_unique=True)
		if cnt1 > limit1:
			sorted_idx = np.argsort(-sample_weight[vec1])
			select_serial[i:sel_num] = vec1[sorted_idx[i:sel_num]]

	# while i < sel_num:
	# 	i1 = vec1[np.random.randint(0,sel_num-i)]
	# 	prob1 = np.random.rand()
	# 	cnt1 = cnt1+1
	# 	if prob1>=sample_weight[i1]:
	# 		select_serial[i] = i1
	# 		i = i+1
	# 		vec1 = np.setdiff1d(vec1,select_serial[0:i],assume_unique=True)
	# 	if cnt1 > limit1:
	# 		sorted_idx = np.argsort(sample_weight[vec1])
	# 		select_serial[i:sel_num] = vec1[sorted_idx[i:sel_num]]

	return select_serial

# sample regions
# type_id: 0: sample using sel_num; 1: sample by each region
def sample_region(sample_weight,sel_num,thresh=0.6,sel_ratio=1,type_id=1,epsilon=0.15,thresh_1=0.9):

	sample_num = len(sample_weight)
	if sel_ratio!=1:
		sel_num = int(sample_num*sel_ratio)

	# random.seed(seed1)
	# tf.compat.v1.set_random_seed(seed1)
	# seed1 = 0
	# np.random.seed(seed1)
	if type_id==0:
		select_serial = sample_region_sub1(sample_num,sel_num,sample_weight)
	elif type_id==1:
		prob1 = np.random.rand(sample_num)
		print(np.max(sample_weight),np.min(sample_weight),np.mean(sample_weight),np.median(sample_weight))
		select_serial = np.where(prob1<=sample_weight)[0]
		# select_serial = np.where((prob1<=sample_weight)&(sample_weight>thresh))[0]
		print(sample_num,len(select_serial))
		b1 = np.where(sample_weight[select_serial]<thresh)[0]
		num1 = len(b1)

		prob2 = np.random.rand(num1)
		b2 = np.where(prob2>epsilon)[0]
		id2 = b1[b2]
		serial2 = select_serial[id2]
		select_serial = np.setdiff1d(select_serial,serial2)

		thresh1 = thresh_1
		b1 = np.where(sample_weight>thresh1)[0]
		select_serial = np.union1d(select_serial,b1)

		print(num1,len(serial2),len(b1),len(select_serial))

	else:
		prob1 = np.random.rand(sample_num)
		select_serial = np.where(prob1<=sample_weight)[0]
		t_sel_num = len(select_serial)
		if t_sel_num<sel_num:
			sample_num1 = sample_num-t_sel_num
			sel_num1 = sel_num-t_sel_num
			vec1 = np.setdiff1d(range(0,sample_num),select_serial,assume_unique=True)
			sample_weight1 = sample_weight[vec1]
			select_serial1 = sample_region_sub1(sample_num1,sel_num1,sample_weight1)
			select_serial = np.union1d(select_serial,select_serial1)
		else:
			sample_weight1 = sample_weight[select_serial]
			sorted_idx = np.argsort(-sample_weight1)
			select_serial = select_serial[sorted_idx[0:sel_num]]

	print('select_serial',len(select_serial),len(select_serial)/sample_num)

	return select_serial

## evaluation 
# compare estimated score with existing elements
# filename1: estimated attention
# filename2: ERCE
def compare_with_regions(filename1, filename2, output_filename, output_filename1, tol=2, filename1a=''):

	data1 = pd.read_csv(filename1,sep='\t')
	# data1a = pd.read_csv(filename1a,sep='\t')
	colnames1 = list(data1)
	chrom1, serial1 = np.asarray(data1['chrom']), np.asarray(data1['serial'])
	# b1 = np.where(chrom1!='chr16')[0]
	# data_1 = data1.loc[b1,colnames1]

	# data3 = pd.concat([data_1,data1a], axis=0, join='outer', ignore_index=True, 
	# 			keys=None, levels=None, names=None, verify_integrity=False, copy=True)
		
	data3 = data1
	print(list(data3))
	# data3.sort_values(by=['serial'])
	num1 = data3.shape[0]
	print(num1)
	label1 = np.zeros(num1)
	chrom1, start1, stop1 = data3['chrom'], data3['start'], data3['stop']
	attention1 = data3[colnames1[-1]]

	# load ERCE files
	data2 = pd.read_csv(filename2,header=None,sep='\t')
	colnames2 = list(data2)
	col1, col2, col3 = colnames2[0], colnames2[1], colnames2[2]
	chrom2, start2, stop2 = data2[col1], data2[col2], data2[col3]

	num2 = len(chrom2)
	score1 = -np.ones(num2)
	for i in range(0,num2):
		t_chrom, t_start, t_stop = chrom2[i], start2[i], stop2[i]
		b1_ori = np.where((chrom1==t_chrom)&(start1<t_stop)&(stop1>t_start))[0]
		if len(b1_ori)==0:
			continue
			# tolerance of the region
		s1 = max(0,b1_ori[0]-tol)
		s2 = min(len(chrom1),b1_ori[0]+tol+1)
		b1 = list(range(s1,s2))
		# b1 = np.where((chrom1==t_chrom)&(start1>=t_start)&(stop1<=t_stop))[0]
		label1[b1_ori] = 1+i
		# select the maximum score in a region
		t_score = np.max(attention1[b1])
		score1[i] = t_score
		if i%100==0:
			print(i,t_score)

	data3['label'] = label1
	data3.to_csv(output_filename,index=False,sep='\t')

	data2['score'] = score1
	# b1 = np.where(score1>0)[0]
	# data2 = data2.loc[b1,list(data2)]
	data2.to_csv(output_filename1,index=False,sep='\t')

	return data2, data3

# sample regions randomly to compare with elements
def compare_with_regions_sub1(chrom1,start1,stop1,attention1,
							sample_num,region_len,chrom_size,bin_size,tol):

	start_pos = np.random.permutation(chrom_size-int(region_len/bin_size)-1)
	start_pos = start1[start_pos+tol]
	sel_num = attention1.shape[1]
	vec2 = -np.ones((sample_num,2+2*sel_num))

	# start_pos1 = start_pos[0:sample_num]
	start_pos1 = start_pos
	pos1 = np.vstack((start_pos1,start_pos1+region_len)).T
	# print(chrom_size,region_len,region_len/bin_size)
	# print(len(start_pos),len(pos1),pos1[0:2])
	num1 = len(pos1)
	cnt1 = 0
	# attention_1 = attention1[:,0]
	# attention_2 = attention1[:,1]
	for i in range(0,num1):
		t_pos = pos1[i]
		len1 = (t_pos[1]-t_pos[0])/bin_size
		# t_start2 = max(0,t_pos[0]-tol*bin_size)
		# t_stop2 = min(stop1[-1],t_pos[1]+tol*bin_size)
		t_start2, t_stop2 = t_pos[0], t_pos[1]
		b1_ori = np.where((start1<t_stop2)&(stop1>t_start2))[0]
		# print(t_pos,t_start2,t_stop2,(t_stop2-t_start2)/bin_size,len(b1_ori))
		if len(b1_ori)<len1*0.5:
			continue
			# s1 = max(0,b1_ori[0]-tol)
			# s2 = min(t_chrom_size,b1_ori[0]+tol+1)
			# b1 = b2[s1:s2]
		# vec2[cnt1] = np.max(attention_1[b1_ori])
		t_vec2 = []
		for l in range(sel_num):
			t_vec2.extend([np.max(attention1[b1_ori,l]),np.mean(attention1[b1_ori,l])])
		vec2[cnt1] = [t_start2,t_stop2]+t_vec2

		cnt1 += 1
		if cnt1>=sample_num:
			break

	vec2 = vec2[vec2[:,0]>=0]

	return vec2

# input: data1: data that query regions from
#        position: the positions to query
def query_region(data1,position,sel_column=[]):

	colnames = list(data1)
	chrom1, start1, stop1 = data1[colnames[0]], data1[colnames[1]], data1[colnames[2]]
	chrom1, start1, stop1 = np.asarray(chrom1), np.asarray(start1), np.asarray(stop1)

	vec1, vec2 = [], []
	if len(sel_column)>0:
		for t_sel_column in sel_column:
			if not(t_sel_column in colnames):
				print('column not found', t_sel_column)
				return
	else:
		sel_column = colnames[0:2]

	value1 = np.asarray(data1[sel_column])
	vec2 = []
	for t_position in position:
		t_chrom, t_start, t_stop = t_position
		b1 = np.where((chrom1==t_chrom)&(t_start<stop1)&(t_stop>start1))[0]
		vec1.append(value1[b1])
		vec2.append(b1)
	
	return vec1, vec2

# sample regions randomly to compare with elements
def compare_with_regions_random1(filename1,filename2,output_filename,output_filename1,
									output_filename2, tol=1, sample_num=200, type_id=1):

	data1 = pd.read_csv(filename1,sep='\t')
	# data1a = pd.read_csv(filename1a,sep='\t')
	# colnames1 = list(data1)
	# chrom1, serial1 = np.asarray(data1['chrom']), np.asarray(data1['serial'])
	# b1 = np.where(chrom1!='chr16')[0]
	# data_1 = data1.loc[b1,colnames1]

	# data3 = pd.concat([data_1,data1a], axis=0, join='outer', ignore_index=True, 
	# 			keys=None, levels=None, names=None, verify_integrity=False, copy=True)

	# load ERCE files
	data2 = pd.read_csv(filename2,header=None,sep='\t')
	colnames2 = list(data2)
	col1, col2, col3 = colnames2[0], colnames2[1], colnames2[2]
	chrom2, start2, stop2 = data2[col1], data2[col2], data2[col3]
		
	data3 = data1
	colnames1 = list(data3)
	# print(colnames1)
	# data3 = data3.sort_values(by=['serial'])
	num1 = data3.shape[0]
	print(num1)
	label1 = np.zeros(num1,dtype=np.int32)
	chrom1, start1, stop1 = np.asarray(data3['chrom']), np.asarray(data3['start']), np.asarray(data3['stop'])
	quantile_vec1 = ['Q1','Q2']
	# attention1 = np.asarray(data3['predicted_attention'])
	# attention2 = np.asarray(data3[quantile_vec1[type_id]])	# ranking
	sel_column = ['predicted_attention',quantile_vec1[type_id]]
	attention1 = np.asarray(data3.loc[:,sel_column])

	chrom_vec = np.unique(chrom2)
	chrom_num = len(chrom_vec)
	chrom_size1 = len(chrom1)
	bin_size = stop1[1]-start1[1]

	num2 = len(chrom2)
	sel_num = 2
	sel_num1 = 2*sel_num
	score1 = -np.ones((num2,sel_num*6),dtype=np.float32)
	vec1 = []

	for t_chrom in chrom_vec:
		# t_chrom = chrom_vec[i]
		b1 = np.where(chrom2==t_chrom)[0]
		num2 = len(b1)
		b2 = np.where(chrom1==t_chrom)[0]
		print(num2,len(b1),len(b2))

		if len(b1)==0 or len(b2)==0:
			print('chromosome not found', t_chrom)
			continue
			
		t_chrom_size = len(b2)
		print('sample regions %d'%(sample_num),t_chrom_size)

		for l in range(0,num2):
			i1 = b1[l]
			t_chrom, t_start, t_stop = chrom2[i1], start2[i1], stop2[i1]
			t_chrom1, t_start1, t_stop1 = chrom1[b2], start1[b2], stop1[b2]
			t_attention1 = attention1[b2]

			print(t_stop,t_start)
			region_len_ori = t_stop-t_start
			t_start = max(0,t_start-tol*bin_size)
			t_stop = min(t_stop1[-1],t_stop+tol*bin_size)

			b1_ori = np.where((t_start1<t_stop)&(t_stop1>t_start))[0]
			if len(b1_ori)==0:
				continue

			b1_ori = b2[b1_ori]
			# s1 = max(0,b1_ori[0]-tol)
			# s2 = min(chrom_size1,b1_ori[0]+tol+1)
			# b1 = list(range(s1,s2))
			# b1 = np.where((chrom1==t_chrom)&(start1>=t_start)&(stop1<=t_stop))[0]
			label1[b1_ori] = 1+i1
			# select the maximum score in a region
			# t_score1 = np.max(attention1[b1_ori])
			# t_score2 = np.mean(attention2[b1_ori])
			# t_score1_1 = np.max(attention2[b1_ori])
			# t_score2_2 = np.mean(attention2[b1_ori])
			for l1 in range(sel_num):
				id2 = 2*l1
				score1[i1,id2:(id2+2)] = [np.max(attention1[b1_ori,l1]),np.mean(attention1[b1_ori,l1])]

			# randomly sample regions
			region_len = t_stop-t_start
			t_chrom_size = len(b2)
			# sample_num = 200
			vec2 = compare_with_regions_sub1(t_chrom1,t_start1,t_stop1,t_attention1,
										sample_num,region_len,t_chrom_size,bin_size,tol)

			# vec2 = compare_with_regions_sub2(t_chrom1,t_start1,t_stop1,t_attention1,
			# 							sample_num,region_len,t_chrom_size,bin_size,tol)
				
			vec3 = []
			print(vec2.shape)
			num3 = vec2.shape[1]
			# if num3!=sel_num1:
			# 	print('error!',num3,sel_num1)
			# 	return
			assert num3==(sel_num1+2)

			vec2_1 = vec2[:,2:]

			for l1 in range(sel_num1):
				t_score_mean1, t_score_std1 = np.mean(vec2_1[:,l1]), np.std(vec2_1[:,l1])
				vec3.extend([t_score_mean1,t_score_std1])

			sample_num1 = len(vec2)
			score1[i1,sel_num1:] = vec3
			t1 = np.asarray([1+i1]*sample_num1)
			vec1.extend(np.hstack((t1[:,np.newaxis],vec2)))

			# if i%100==0:
			# 	print(i,score1[i],len(vec2))
			print(i1,score1[i1],len(vec2),vec2.shape)
			# if l>10:
			# 	break

	data3['label'] = label1
	data3.to_csv(output_filename,index=False,sep='\t')

	for l in range(sel_num1):
		data2['score%d'%(l+1)] = score1[:,l]
		id1 = sel_num1+2*l
		data2['score_comp_mean%d'%(l+1)], data2['score_comp_std%d'%(l+1)] = score1[:,id1],score1[:,id1+1]
	
	# data2['score'] = score1[:,0]
	# data2['score_quantile'] = score1[:,1]
	# data2['score_comp_mean'], data2['score_comp_std'] = score1[:,2],score1[:,3]
	# data2['score_comp_mean1'], data2['score_comp_std1'] = score1[:,4],score1[:,5]
	# b1 = np.where(score1>0)[0]
	# data2 = data2.loc[b1,list(data2)]
	data2.to_csv(output_filename1,index=False,sep='\t')

	vec1 = np.asarray(vec1)
	num1 = vec1.shape[1]
	print(vec1.shape)
	fields = ['region_id','start','stop']
	data_1 = pd.DataFrame(columns=fields)
	for i in range(3):
		data_1[fields[i]] = np.int64(vec1[:,i])
	for i in range(3,num1):
		data_1['sel%d'%(i-2)] = vec1[:,i]
	data_1.to_csv(output_filename2,index=False,sep='\t')

	return True

# find overlapping regions
# input: data1: position file 1
#        data2: position file 2
#        mode: 0, for each position in file 1, find all positions in file 2 overlapping with this position
#		 mode: 1, for each posiiton in file 1, find position in flie 2 that has the longest overlap with this position
def overlapping_with_regions(data1,data2,mode=0):

	colnames1 = list(data1)
	chrom1, start1, stop1 = np.asarray(data1[colnames1[0]]), np.asarray(data1[colnames1[1]]), np.asarray(data1[colnames1[2]])
	num1 = len(chrom1)

	colnames2 = list(data2)
	chrom2, start2, stop2 = np.asarray(data2[colnames2[0]]), np.asarray(data2[colnames2[1]]), np.asarray(data2[colnames2[2]])
	num1 = len(chrom1)

	id_vec1, id_vec2 = [], []
	for i in range(num1):
		t_chrom1, t_start1, t_stop1 = chrom1[i], start1[i], stop1[i]
		b1 = np.where((chrom2==t_chrom1)&(start2<t_stop1)&(stop2>t_start1))[0]
		if len(b1)>0:
			id_vec1.append(i)
			id_vec2.append(b1)
		if i%1000==0:
			print(t_chrom1,t_start1,t_stop1,t_stop1-t_start1)

	id_vec1 = np.asarray(id_vec1)
	id_vec2 = np.asarray(id_vec2)

	return id_vec1, id_vec2

# sample regions randomly to compare with elements
# input: filename1: estimation file
# 		 filename2: ERCE file
#		 output_filename: save ERCE for the cell type
#		 tol: tolerance for extension of the ERCE region
#		 label_name: label of genomic loci overlapping with ERCE
# return: data3: estimation file, data2: ERCE
def compare_with_regions_pre(filename1,filename2,output_filename='',tol=2,
								label_name='label',save_mode=0,region_data=[],
								select_id=0,config={}):
	
	# load genomic loci file
	data1 = pd.read_csv(filename1,sep='\t')
	# label_name = 'label1'

	# load ERCE files
	if len(region_data)==0:
		# data2 = pd.read_csv(filename2,header=None,sep='\t')
		data2 = pd.read_csv(filename2,sep='\t')
		colnames = list(data2)
		if colnames[0]!='chrom':
			data2 = pd.read_csv(filename2,header=None,sep='\t')
	else:
		data2 = region_data

	colnames2 = list(data2)
	col1, col2, col3 = colnames2[0], colnames2[1], colnames2[2]
	chrom2, start2, stop2 = np.asarray(data2[col1]), np.asarray(data2[col2]), np.asarray(data2[col3])

	data3 = data1
	colnames1 = list(data3)
	num1 = data3.shape[0]
	print(num1,colnames1)
	label1 = np.zeros(num1,dtype=np.int64)
	label2 = np.zeros(num1,dtype=np.int64)
	chrom1, start1, stop1 = np.asarray(data3['chrom']), np.asarray(data3['start']), np.asarray(data3['stop'])

	if os.path.exists(output_filename):
		data3_1 = pd.read_csv(output_filename,sep='\t')
		t_chrom, t_start = data3_1['chrom'], data3_1['start']
		print(output_filename)
		print(len(chrom1),len(t_chrom))
		b1 = (chrom1!=t_chrom)
		b2 = (start1!=t_start)
		if np.sum(b1)>0 or np.sum(b2)>0:
			print('error!')
			return

		data3[label_name] = data3_1[label_name].copy()
		label_name1 = label_name+'_tol%d'%(tol)
		data3[label_name1] = data3_1[label_name1].copy()
		return data3, data2

	chrom_vec = np.unique(chrom2)
	chrom_num = len(chrom_vec)
	chrom_size1 = len(chrom1)
	bin_size = stop1[1]-start1[1]

	num2 = len(chrom2)
	sel_num = 2
	sel_num1 = 2*sel_num
	score1 = -np.ones((num2,sel_num*6),dtype=np.float32)
	vec1 = []

	for t_chrom in chrom_vec:
		# t_chrom = chrom_vec[i]
		b1 = np.where(chrom2==t_chrom)[0]
		num2 = len(b1)
		b2 = np.where(chrom1==t_chrom)[0]
		# print(num2,len(b1),len(b2))

		if len(b1)==0 or len(b2)==0:
			print('chromosome not found', t_chrom)
			continue
			
		t_chrom_size = len(b2)
		print(t_chrom, t_chrom_size)

		for l in range(0,num2):
			i1 = b1[l]
			t_chrom, t_start, t_stop = chrom2[i1], start2[i1], stop2[i1]
			t_chrom1, t_start1, t_stop1 = chrom1[b2], start1[b2], stop1[b2]

			# print(t_stop,t_start)
			region_len_ori = t_stop-t_start

			start_1 = max(0,t_start-tol*bin_size)
			stop_1 = min(t_stop1[-1],t_stop+tol*bin_size)

			b1_ori = np.where((t_start1<stop_1)&(t_stop1>start_1))[0]
			if len(b1_ori)==0:
				continue

			b1_ori = b2[b1_ori]
			label1[b1_ori] = 1+i1
			# print(i1)

			start_1 = max(0,t_start)
			stop_1 = min(t_stop1[-1],t_stop)
			b1_ori = np.where((t_start1<stop_1)&(t_stop1>start_1))[0]
			if len(b1_ori)==0:
				continue

			b1_ori = b2[b1_ori]	
			label2[b1_ori] = 1+i1
			# if l>10:
			# 	break

	# data3['label'] = label1
	data3[label_name] = label2
	label_name1 = '%s_tol%d'%(label_name,tol)
	data3[label_name1] = label1

	print('region',data2.shape)
	print('estimation', data3.shape)
	if select_id==1:
		signal = np.asarray(data3['signal'])
		median1 = np.median(signal)
		thresh1 = np.quantile(signal,config['thresh1'])-1e-12
		print(median1,thresh1)
		# return -1

		id1 = np.where(signal>thresh1)[0]
		data3 = data3.loc[id1,:]
		data3.reset_index(drop=True,inplace=True)
		print(data3.shape,len(id1),np.median(data3['signal']))

		id2, id2_1 = overlapping_with_regions(data2,data3)
		data2 = data2.loc[id2,:]
		data2.reset_index(drop=True,inplace=True)
		print(data2.shape,len(id2))

	elif select_id==2:
		region_data2 = config['region_data2']
		id1, id1_1 = overlapping_with_regions(data3,region_data2)
		data3 = data3.loc[id1,:]
		data3.reset_index(drop=True,inplace=True)
		print(data3.shape,len(id1))

		id2, id2_1 = overlapping_with_regions(data2,region_data2)
		data2 = data2.loc[id2,:]
		data2.reset_index(drop=True,inplace=True)
		print(data2.shape,len(id2))

	elif select_id==3:
		x = 1

	else:
		pass

	data3_1 = data3.loc[:,['chrom','start','stop','serial','signal',label_name,label_name1]]
	if save_mode==1:
		if output_filename=='':
			output_filename = filename1+'.label.txt'
		data3_1.to_csv(output_filename,index=False,sep='\t')
	
	return data3, data2

# location of elements
# input: filename1: annoation file
# 		 filename2: genomic loci file
#		 output_filename: save serial of elements
# return: save serial of elements
def compare_with_regions_pre1(filename1,filename2,output_filename):

	data1 = pd.read_csv(filename1,header=None,sep='\t')
	data2 = pd.read_csv(filename2,sep='\t')

	sample_num1 = data1.shape[0]
	sample_num2 = data2.shape[0]
	print(sample_num1,sample_num2)

	colnames = list(data2)
	chrom1, start1, stop1 = np.asarray(data1[0]), np.asarray(data1[1]), np.asarray(data1[2])
	chrom2, start2, stop2, serial = np.asarray(data2[colnames[0]]), np.asarray(data2[colnames[1]]), np.asarray(data2[colnames[2]]), np.asarray(data2[colnames[3]])
	signal = np.asarray(data2[colnames[4]])
	
	chrom_num = 22
	chrom_vec = ['chr%d'%(i) for i in range(1,chrom_num+1)]
	id1, id2 = dict(), dict()

	serial1, num_vec = -np.ones(sample_num1,dtype=np.int64), np.zeros(sample_num1,dtype=np.int64)
	signal1 = np.zeros(sample_num1,dtype=np.float32)
	for i in range(1,chrom_num+1):
		t_chrom = 'chr%d'%(i)
		b1 = np.where(chrom1==t_chrom)[0]
		id1[t_chrom] = b1
		b2 = np.where(chrom2==t_chrom)[0]
		id2[t_chrom] = b2
		print(t_chrom,len(b1),len(b2))

		for id_1 in b1:
			t_chrom1, t_start1, t_stop1 = chrom1[id_1], start1[id_1], stop1[id_1]
			idx = np.where((start2[b2]<t_stop1)&(stop2[b2]>t_start1))[0]
			if len(idx)>0:
				serial1[id_1] = serial[b2[idx[0]]]
				num_vec[id_1] = len(idx)
				signal1[id_1] = signal[b2[idx[0]]]

	colnames = list(data1)
	data1['serial'] = serial1
	data1['num'] = num_vec
	data1['signal'] = signal1
	data1 = data1.loc[:,colnames[0:3]+[(colnames[-2])]+['serial','num','signal']]

	data1.to_csv(output_filename,index=False,header=False,sep='\t')

	return True

# location of elements
# input: filename1: annotation file 1
# 		 filename2: annotation file 2
#		 output_filename: save serial of elements
# return: save label of elements
def compare_with_regions_pre2(filename1,filename2,output_filename,type_id=0,chrom_num=22):

	if type_id==0:
		data1 = pd.read_csv(filename1,header=None,sep='\t')
	else:
		data1 = pd.read_csv(filename1,sep='\t')

	data2 = pd.read_csv(filename2,header=None,sep='\t')

	sample_num1 = data1.shape[0]
	sample_num2 = data2.shape[0]
	print(sample_num1,sample_num2)

	colnames = list(data1)
	chrom1, start1, stop1 = np.asarray(data1[colnames[0]]), np.asarray(data1[colnames[1]]), np.asarray(data1[colnames[2]])
	colnames = list(data2)
	chrom2, start2, stop2 = np.asarray(data2[colnames[0]]), np.asarray(data2[colnames[1]]), np.asarray(data2[colnames[2]])
	
	chrom_vec = ['chr%d'%(i) for i in range(1,chrom_num+1)]
	id1, id2 = dict(), dict()

	label = np.zeros(sample_num1,dtype=np.int64)
	for i in range(1,chrom_num+1):
		t_chrom = 'chr%d'%(i)
		b1 = np.where(chrom1==t_chrom)[0]
		id1[t_chrom] = b1
		b2 = np.where(chrom2==t_chrom)[0]
		id2[t_chrom] = b2
		print(t_chrom,len(b1),len(b2))

		for id_1 in b1:
			t_chrom1, t_start1, t_stop1 = chrom1[id_1], start1[id_1], stop1[id_1]
			idx = np.where((start2[b2]<t_stop1)&(stop2[b2]>t_start1))[0]
			if len(idx)>0:
				id_2 = b2[idx]
				overlapping = 0
				for t_id in id_2:
					t_start2, t_stop2 = start2[t_id], stop2[t_id]
					overlapping += np.min([t_stop1-t_start2,t_stop2-t_start2,t_stop2-t_start1,t_stop1-t_start1])
				label[id_1] = overlapping

	colnames = list(data1)
	data1['label'] = label

	data1.to_csv(output_filename,index=False,header=False,sep='\t')

	return True

# local peak search: attention score
def compare_with_regions_peak_search(chrom,start,stop,serial,value,seq_list,config={}):

	thresh_vec = config['thresh_vec']
	peak_type = config['peak_type']
	if 'distance_peak_thresh' in config:
		distance_thresh = config['distance_peak_thresh']
	else:
		distance_thresh = 5

	sample_num = len(chrom)
	thresh_num = len(thresh_vec)
	dict1 = dict()
	for thresh in thresh_vec:
		dict1[thresh] = []

	width_thresh = 5
	for t_seq in seq_list:
		pos1, pos2 = t_seq[0], t_seq[1]+1
		b1 = np.asarray(range(pos1,pos2))
		x = value[b1]

		chrom_id = chrom[pos1]
		t_serial = serial[b1]
		# s1, s2 = np.max(x), np.min(x)
		# print(chrom_id,len(x),np.max(x,axis=0),np.min(x,axis=0),np.median(x,axis=0))
		x1, x2 = x[:,0], x[:,1] # x1: find peak,  x2: prominence
		if peak_type==0:
			threshold = config['threshold']
			if threshold>0:
				peaks, c1 = find_peaks(x1,distance=distance_thresh,threshold=threshold,width=(1,10),plateau_size=(1,10))
			else:
				peaks, c1 = find_peaks(x1,distance=distance_thresh,width=(1,10),plateau_size=(1,10))

		else:
			if 'width' in config:
				width = config['width']
			else:
				width = 10
			width1 = np.arange(1,width+1)
			peaks = find_peaks_cwt(x1, width1)

		if len(peaks)>0:
			# print(peak_type,x1[peaks],t_serial[peaks])
			dict1 = peak_region_search_1(x1,x2,peaks,b1,width_thresh,thresh_vec,dict1)

	# label = np.zeros((sample_num,thresh_num,2),dtype=np.int64)
	print(dict1.keys())

	dict2 = dict()
	for l in range(thresh_num):
		thresh = thresh_vec[l]
		list1 = dict1[thresh]
		list1 = np.asarray(list1)
		# print(list1.shape)

		id1 = list1[:,0]
		serial1 = serial[id1]
		id1 = mapping_Idx(serial,serial1)
		id_1 = np.argsort(serial1)
		id1 = id1[id_1]
		annot1 = [thresh,0]
		dict2[thresh] = [chrom[id1],start[id1],stop[id1],serial[id1],annot1]

	return dict2

# local peak search: attention score
def compare_with_regions_peak_search1(chrom,start,stop,serial,value,seq_list,thresh_vec=[0.9]):

	sample_num = len(chrom)
	thresh_num = len(thresh_vec)
	dict1 = dict()
	for thresh in thresh_vec:
		list1, list2 = [], []
		dict1[thresh] = [list1,list2]

	width_thresh = 5
	for t_seq in seq_list:
		pos1, pos2 = t_seq[0], t_seq[1]+1
		b1 = np.asarray(range(pos1,pos2))
		x = value[b1]

		chrom_id = chrom[pos1]
		t_serial = serial[b1]
		# s1, s2 = np.max(x), np.min(x)
		print(chrom_id,len(x),np.max(x,axis=0),np.min(x,axis=0),np.median(x,axis=0))
		x1, x2 = x[:,0], x[:,1]
		peaks, c1 = find_peaks(x1,distance=10,width=(1,10),plateau_size=(1,10))
		width1 = np.arange(1,11)
		peaks_cwt = find_peaks_cwt(x1, width1)

		if len(peaks)>0:
			dict1 = peak_region_search(x1,x2,peaks,b1,width_thresh,thresh_vec,dict1,type_id2=0)
		if len(peaks_cwt)>0:
			dict1 = peak_region_search(x1,x2,peaks_cwt,b1,width_thresh,thresh_vec,dict1,type_id2=1)

	label = np.zeros((sample_num,thresh_num,2),dtype=np.int64)
	print(dict1.keys())

	dict2 = dict()
	for l in range(thresh_num):
		thresh = thresh_vec[l]
		list1, list2 = dict1[thresh]

		list1 = np.asarray(list1)
		list2 = np.asarray(list2)
		print(len(list1),len(list2))

		id1, id2 = list1[:,0], list2[:,0]
		serial1, serial2 = serial[id1], serial[id2]
		id_1, id_2 = np.argsort(serial1), np.argsort(serial2)
		id1, id2 = id1[id_1], id2[id_2]
		annot1, annot2 = [thresh,0], [thresh,1]
		dict2[thresh] = [[chrom[id1],start[id1],stop[id1],serial[id1],annot1],
							[chrom[id2],start[id2],stop[id2],serial[id2],annot2]]

	return dict2

# local peak search: signal
def compare_with_regions_peak_search2(chrom,start,stop,serial,value,seq_list,thresh_vec=[0],config={}):

	sample_num = len(chrom)
	thresh_num = len(thresh_vec)
	dict1 = dict()
	for thresh in thresh_vec:
		list1, list2 = [], []
		dict1[thresh] = [list1,list2]

	# width_thresh = 5
	prominence_thresh = 0
	distance_thresh, width_thresh = 20, 20
	if len(config)>0:
		prominence_thresh, distance_thresh, width_thresh = config['prominence_thresh'], config['distance_thresh'], config['width_thresh']
		
	for t_seq in seq_list:
		pos1, pos2 = t_seq[0], t_seq[1]+1
		b1 = np.asarray(range(pos1,pos2))
		x = value[b1]

		chrom_id = chrom[pos1]
		t_serial = serial[b1]
		# s1, s2 = np.max(x), np.min(x)
		print(chrom_id,len(x),np.max(x,axis=0),np.min(x,axis=0),np.median(x,axis=0))
		
		peaks, c1 = find_peaks(x,distance=distance_thresh,width=(1,width_thresh),plateau_size=(1,10))
		# if prominence_thresh>0:
		# 	peaks, c1 = find_peaks(x,distance=distance_thresh,width=(1,width_thresh),prominence=prominence_thresh,plateau_size=(1,10))
		# else:
		# 	peaks, c1 = find_peaks(x,distance=distance_thresh,width=(1,width_thresh),plateau_size=(1,10))
		width1 = np.arange(1,width_thresh+1)
		peaks_cwt = find_peaks_cwt(x, width1)

		if len(peaks)>0:
			dict1 = peak_region_search(x,x,peaks,b1,width_thresh,thresh_vec,dict1,type_id2=0)
		if len(peaks_cwt)>0:
			dict1 = peak_region_search(x,x,peaks_cwt,b1,width_thresh,thresh_vec,dict1,type_id2=1)

	label = np.zeros((sample_num,thresh_num,2),dtype=np.int64)
	print(dict1.keys())

	dict2 = dict()
	for l in range(thresh_num):
		thresh = thresh_vec[l]
		list1, list2 = dict1[thresh]

		list1 = np.asarray(list1)
		list2 = np.asarray(list2)
		print(len(list1),len(list2))

		id1, id2 = list1[:,0], list2[:,0]
		serial1, serial2 = serial[id1], serial[id2]
		id_1, id_2 = np.argsort(serial1), np.argsort(serial2)
		id1, id2 = id1[id_1], id2[id_2]
		annot1, annot2 = [thresh,0], [thresh,1]
		dict2[thresh] = [[chrom[id1],start[id1],stop[id1],serial[id1],annot1],
							[chrom[id2],start[id2],stop[id2],serial[id2],annot2]]

	return dict2

def compare_with_regions_init_search(chrom,start,stop,serial,init_zone,attention1,thresh,flanking=30,bin_size=5000):

	chrom_1, start_1, stop_1 = init_zone	# init zone
	chrom_vec = np.unique(chrom)
	id1 = np.where(attention1>thresh)[0]
	num1 = len(attention1)
	num1_thresh = len(id1)
	tol = flanking*bin_size
	list1 = []
	print(chrom_1[0:10],start_1[0:10],stop_1[0:10])
	print(flanking,tol)

	for chrom_id in chrom_vec:
		b1 = np.where(chrom==chrom_id)[0]
		b2 = np.where(chrom_1==chrom_id)[0]
		num2 = len(b2)
		for i in range(num2):
			t_chrom1, t_start1, t_stop1 = chrom_1[i], start_1[i], stop_1[i]
			t_start_1, t_stop_1 = t_start1-tol, t_stop1+tol
			t_id2 = np.where((start[b1]<t_stop_1)&(stop[b1]>t_start_1))[0]
			t_id2 = b1[t_id2]
			list1.extend(t_id2)

	list1 = np.asarray(list1)
	id2 = np.intersect1d(list1,id1)

	chrom1, start1, stop1, serial1 = chrom[id2], start[id2], stop[id2], serial[id2]
	annot1 = [thresh,flanking]
	print('compare with regions init search',len(id2),num1_thresh,num1,len(id2)/num1_thresh,num1_thresh/num1)

	return (chrom1, start1, stop1, serial1, annot1)

def compare_with_regions_signal_search(chrom,start,stop,serial,signal_vec,attention1,thresh=0.95,flanking=30):

	signal, predicted_signal = signal_vec[:,0], singal_vec[:,1]
	mse = np.abs(signal-predicted_signal)
	thresh1, thresh2 = np.quantile(mse,[0.5,0.25])
	thresh_1, thresh_2 = np.quantile(signal,[0.55,0.45])
	id1 = np.where(attention1>thresh)[0]
	num1 = len(attention1)
	num1_thresh = len(id1)
	t_vec1 = np.zeros(num1,dtype=bool)
	vec1 = score_2a(signal,predicted_signal)
	vec2 = score_2a(signal[id1],predicted_signal[id1])

	# for t_seq in seq_list:
	# 	pos1, pos2 = t_seq[0], t_seq[1]+1
	#	b1 = np.where((id1<pos2)&(id1>=pos1))[0]
	chrom_vec = np.unique(chrom)
	for chrom_id in chrom_vec:
		b1 = np.where(chrom==chrom_id)[0]
		b2 = np.intersect1d(id1,b1)
		num2 = len(b2)
		t_serial = serial[b2]
		for i in range(num2):
			t_id1 = t_serial[i]
			id2 = np.where((serial[b1]>=t_id1-flanking)&(serial[b1]<=t_id1+flanking))[0]
			id2 = b1[id2]
			t_signal, t_pred = signal[id2], predicted_signal[id2]
			error = np.abs(t_signal-t_pred)
			flag1 = np.asarray([np.mean(t_signal)>thresh_1,np.mean(t_signal)<thresh_2])
			flag2 = np.asarray([np.mean(t_pred)>thresh_1,np.mean(t_pred)<thresh_2])
			temp1 = flag1^flag2
			t_vec1[b2[i]] = (np.median(error)<thresh2)&(np.sum(temp1)==0)
	
	id2 = (t_vec1>0)
	chrom1, start1, stop1, serial1 = chrom[id2], start[id2], stop[id2], serial[id2]
	annot1 = [thresh,flanking,thresh2,thresh_1,thresh_2]

	print('compare with regions signal search',len(id2),num1_thresh,num1,len(id2)/num1_thresh,num1_thresh/num1)
	print(vec1)
	print(vec2)

	return (chrom1, start1, stop1, serial1, annot1)

def compare_with_regions_single(chrom,start,stop,serial,label,value,attention1,thresh_vec=[0.05,0.1],value1=[],config={}):

	thresh, thresh_fdr = thresh_vec
	b1 = np.where(label>0)[0]
	b2 = np.where(label==0)[0]
	num1, num2 = len(b1), len(b2)
	region_num = len(chrom)
	print(num1, num2, num1/region_num)

	n_dim = value.shape[1]
	value2 = value[b2]
	mean_value2 = np.mean(value2,axis=0)+1e-12
	thresh_1 = np.zeros(n_dim)
	thresh_2 = np.zeros(n_dim)
	for i in range(n_dim):
		thresh_1[i] = np.quantile(value2[:,i],0.95)
		thresh_2[i] = np.quantile(value2[:,i],0.05)

	value1 = value[b1]
	fold_change = value1/(np.outer(np.ones(num1),mean_value2))
	fold_change = np.asarray(fold_change,dtype=np.float32)

	print(value1.shape,value2.shape)
	mtx1 = np.zeros((num1,n_dim),dtype=np.float32)
	for i in range(num1):
		if i%100==0:
			print(i)
		id_1 = (value2>=value1[i])
		cnt1 = np.sum(id_1,axis=0)
		mtx1[i] = cnt1/num2
		# if i>100:
		# 	break
	# mtx1 = np.int8(value1>thresh_1)
	# mtx1[value1<thresh_2] = -1

	mean_fold_change = np.mean(fold_change,axis=0)
	id1 = np.argsort(-mean_fold_change)

	print(np.max(fold_change),np.min(fold_change))
	print(np.max(mtx1),np.min(mtx1))
	print('compare with regions single')

	if 'feature_name_list' in config:
		feature_name_list = config['feature_name_list']
		feature_name_list = feature_name_list[id1]
	else:
		feature_name_list = id1

	fold_change = fold_change[:,id1]
	mtx1 = mtx1[:,id1]

	fields = ['chrom','start','stop','serial','label','predicted_attention']+list(feature_name_list)
	data2 = pd.DataFrame(columns=fields)
	data2['chrom'], data2['start'], data2['stop'] = chrom[b1], start[b1], stop[b1]
	data2['serial'], data2['label'] = serial[b1], label[b1]
	data2['predicted_attention'] = attention1[b1]

	data_2 = data2.copy()
	data2.loc[:,feature_name_list] = fold_change
	data_2.loc[:,feature_name_list] = mtx1
	print(data2.shape, data_2.shape)

	return data2, data_2, feature_name_list

def compare_with_regions_distribute(chrom,start,stop,serial,label,value,thresh_vec=[0.05,0.1],value1=[],config={}):

	thresh, thresh_fdr = thresh_vec
	b1 = np.where(label>0)[0]
	b2 = np.where(label==0)[0]
	num1, num2 = len(b1), len(b2)
	region_num = len(chrom)
	# print(num1, num2, num1/region_num)

	alternative, feature_name, plot_id = 'two-sided', 'Element', 0
	if 'alternative' in config:
		alternative = config['alternative']
	if 'feature_name' in config:
		feature_name = config['feature_name']
	if 'plot_id' in config:
		plot_id = config['plot_id']

	data1, data2 = value[b1], value[b2]
	value1, value2 = score_2a_1(data1, data2, alternative=alternative) # value1: p-value, value2: statistics
	mannwhitneyu_pvalue,ks_pvalue = value1[0], value1[1]
	mean_fold_change = np.mean(data1)/(np.mean(data2)+1e-12)

	t1, t2 = np.median(data1), np.median(data2)
	thresh1 = 1e-05
	thresh2 = 0.5
	flag1 = 0
	median_ratio = (t1-t2)/(t2+1e-12)
	if (mannwhitneyu_pvalue<thresh) and (ks_pvalue<thresh):
		flag1 = 1
		# print(feature_name, mannwhitneyu_pvalue, ks_pvalue,t1,t2,median_ratio)
		if median_ratio>thresh2:
			flag1 = 2
		if median_ratio<-thresh2:
			flag1 = 3

	if (mannwhitneyu_pvalue<thresh1) and (ks_pvalue<thresh1):
		flag1 = 4
		if median_ratio>thresh2:
			flag1 = 5
		if median_ratio<-thresh2:
			flag1 = 6
		if flag1>=5:
			print(feature_name, mannwhitneyu_pvalue, ks_pvalue,t1,t2,median_ratio)

	output_filename = config['output_filename']
	if flag1>1 and plot_id==1:
		celltype_id = config['celltype_id']
		label_1 = label[label>=0]
		value_1 = value[label>=0]
		output_filename1 = '%s_%s'%(output_filename,feature_name)
		annotation_vec = ['Estimated region','Background',feature_name,celltype_id,flag1]
		plot_sub1(label_1,value_1,output_filename1,annotation_vec)

	return value1, value2, (t1,t2,median_ratio,mean_fold_change), flag1

def plot_sub1(label,value,output_filename1,annotation_vec):
	
	params = {
		 'axes.labelsize': 12,
		 'axes.titlesize': 16,
		 'xtick.labelsize':12,
		 'ytick.labelsize':12}
	pylab.rcParams.update(params)

	id1 = np.where(label>0)[0]
	id2 = np.where(label==0)[0]
	num1, num2 = len(id1), len(id2)
	label1 = [annotation_vec[0]]*num1 + [annotation_vec[1]]*num2
	label1 = np.asarray(label1)

	fields = ['label','mean']
	data1 = pd.DataFrame(columns=fields)
	data1['label'] = label1
	data1['mean'] = value
	
	# output_filename1 = '%s.h5'%(output_filename)
	# with h5py.File(output_filename1,'a') as fid:
	# 	fid.create_dataset("vec", data=data_3, compression="gzip")
	# vec1 = ['ERCE','Background']
	vec1 = annotation_vec
	fig = plt.figure(figsize=(12,11))

	cnt1, cnt2, vec2 = 1, 0, ['mean']
	sel_idList = ['mean']
	num2 = len(sel_idList)
	feature_name = annotation_vec[2]

	for sel_id in sel_idList:
		# print(sel_id)
		plt.subplot(num2,2,cnt1)
		plt.title('%s'%(feature_name))
		# sns.violinplot(x='label', y=sel_id, data=data_3)
		sns.boxplot(x='label', y=sel_id, data=data1, showfliers=False)

		# ax.get_xaxis().set_ticks([])
		ax = plt.gca()
		# ax.get_xaxis().set_visible(False)
		ax.xaxis.label.set_visible(False)
		cnt1 += 1

		# output_filename1 = '%s_%s_1.png'%(output_filename,sel_id)
		# plt.savefig(output_filename1,dpi=300)

		plt.subplot(num2,2,cnt1)
		plt.title('%s'%(feature_name))

		ax = plt.gca()
		ax.xaxis.label.set_visible(False)
		cnt1 += 1
		cnt2 += 1

		for t_label in vec1:
			b1 = np.where(label1==t_label)[0]
			# t_mtx = data_3[data_3['label']==t_label]
			t_mtx = data1.loc[b1,fields]
			sns.distplot(t_mtx[sel_id], hist = True, kde = True,
						 kde_kws = {'shade':True, 'linewidth': 3},
						 label = t_label)

		# output_filename2 = '%s_%s_2.png'%(output_filename,sel_id)
		# plt.savefig(output_filename2,dpi=300)

	file_path = './'
	cell_id = annotation_vec[3]
	flag = annotation_vec[-1]
	output_filename1 = '%s/%s_%s_%d.png'%(file_path,output_filename1,cell_id,flag)
	plt.savefig(output_filename1,dpi=300)

	return True

def compare_with_regions_distribute_test(motif_data,motif_name,est_data,label,thresh_vec=[0.05,0.1],config={}):

	chrom, start, stop, serial = np.asarray(est_data['chrom']), np.asarray(est_data['start']), np.asarray(est_data['stop']), np.asarray(est_data['serial'])
	# attention_1 = np.asarray(est_data['predicted_attention'])
	# attention1 = np.asarray(est_data[sel_column])

	motif_num = len(motif_name)
	motif_data1 = np.asarray(motif_data.loc[:,motif_name])

	vec1 = np.zeros((motif_num,8),dtype=np.float32)
	vec2 = np.zeros(motif_num,dtype=np.int8)
	plot_id = 1
	config.update({'plot_id':plot_id})

	b1 = np.where(label>0)[0]
	b2 = np.where(label==0)[0]
	num1, num2 = len(b1), len(b2)
	region_num = len(chrom)
	print(num1, num2, num1/region_num)

	for i in range(motif_num):

		value = motif_data1[:,i]
		config.update({'feature_name':motif_name[i]})
		value1, value2, t_vec1, flag1 = compare_with_regions_distribute(chrom,start,stop,serial,label,value,thresh_vec=[0.05,0.1],config=config)

		t1, t2, median_ratio, mean_fold_change = t_vec1
		vec1[i] = [i+1,flag1]+list(value1)+[t1, t2, median_ratio, mean_fold_change]
		vec2[i] = flag1

	print(vec2,np.max(vec2),np.min(vec2))
	# p-value correction with Benjamini-Hochberg correction procedure
	thresh, thresh_fdr = thresh_vec
	list1, list2 = [], []
	list1_1, list2_2 = [], []
	vec1_fdr = np.zeros((motif_num,2))
	for i in range(2):
		vec2 = multipletests(vec1[:,i+2],alpha=thresh_fdr,method='fdr_bh')
		vec1_fdr[:,i] = vec2[1]
		b1 = np.where(vec1[:,i+2]<thresh)[0]
		b2 = np.where(vec1_fdr[:,i]<thresh_fdr)[0]
		if i==0:
			id1, id2 = b1, b2
		else:
			id1, id2 = np.intersect1d(id1,b1), np.intersect1d(id2,b2)
		print(len(b1),len(b2),len(id1),len(id2))
		# print(motif_name[id2])
		list1.append(b1)
		list2.append(b2)

	print('compare_with_regions_distribute_test')
	print('mannwhitneyu pvalue')
	print(motif_name[list2[0]])
	print('ks pvalue')
	print(motif_name[list2[1]])
	print(motif_name[id2])

	vec1 = np.asarray(vec1)
	celltype_id = config['celltype_id']
	output_filename1 = config['output_filename']
	output_filename = '%s_%d.txt'%(output_filename1,celltype_id)
	fields = ['motif_id','number','flag','mannwhitneyu_pvalue','ks_pvalue','median_1','median_2','median_ratio','mean_fold_change']
	data1 = pd.DataFrame(columns=fields)
	data1['motif_id'] = motif_name
	for i in range(1,3):
		data1[fields[i]] = np.int64(vec1[:,i-1])
	data1.loc[:,fields[3:]] = vec1[:,2:]
	# np.savetxt(output_filename,vec1,fmt='%.4f',delimiter='\t')
	data1.to_csv(output_filename,index=False,sep='\t')

	return True

def generate_sequences_chrom(chrom,serial,gap_tol=5,region_list=[]):

	num1 = len(chrom)
	idx_sel_list = np.zeros((num1,2),dtype=np.int64)
	chrom_vec = np.unique(chrom)
	for chrom_id in chrom_vec:
		try:
			chrom_id1 = int(chrom_id[3:])
		except:
			print(chrom_id)
			continue
		b1 = np.where(chrom==chrom_id)[0]
		idx_sel_list[b1,0] = chrom_id1
		idx_sel_list[b1,1] = serial[b1]
	
	b1 = (idx_sel_list[:,0]>0)
	idx_sel_list = idx_sel_list[b1]
	print('idx_sel_list',idx_sel_list.shape)
	seq_list = generate_sequences(idx_sel_list, gap_tol=gap_tol, region_list=region_list)

	return seq_list

def output_generate_sequences(chrom,start,stop,serial,idx_sel_list,seq_list,output_filename,save_mode=1):

	num1 = len(seq_list)
	t_serial1 = idx_sel_list[:,1]
	seq_list = np.asarray(seq_list)
	t_serial = t_serial1[seq_list]
	id1 = mapping_Idx(serial,t_serial[:,0])
	chrom1, start1, stop1 = chrom[id1], start[id1], stop[id1]

	id2 = mapping_Idx(serial,t_serial[:,1])
	chrom2, start2, stop2 = chrom[id2], start[id2], stop[id2]

	fields = ['chrom','start','stop','serial1','serial2']
	data1 = pd.DataFrame(columns=fields)
	data1['chrom'], data1['start'], data1['stop'] = chrom1, start1, stop2
	data1['serial1'], data1['serial2'] = t_serial[:,0], t_serial[:,1]
	data1['region_len'] = t_serial[:,1]-t_serial[:,0]+1

	if save_mode==1:
		data1.to_csv(output_filename,index=False,sep='\t')

	return data1

# query predicted signal and importance score for specific regions
def query_importance_score(region_list,filename1,filename2,thresh=0.75):
	
	data_1 = pd.read_csv(filename1,sep='\t')
	data_2 = pd.read_csv(filename2,sep='\t')

	serial = np.asarray(data_2['serial'])
	signal, predicted_signal = np.asarray(data_2['signal']), np.asarray(data_2['predicted_signal'])
	
	serial1 = np.asarray(data_1['serial'])

	assert list(serial)==list(serial1)
	score1 = np.asarray(data_1['Q2'])

	thresh1, thresh2 = np.quantile(signal,0.525), np.quantile(signal,0.475)
	thresh1_1, thresh2_1 = np.quantile(predicted_signal,0.525), np.quantile(predicted_signal,0.475)
	region_num = len(region_list)
	flag_vec = -10*np.ones((region_num,2),dtype=np.int8)

	# thresh = 0.75
	list1 = -np.ones((region_num,7))
	for i in range(region_num):
		region = region_list[i]
		serial_start, serial_stop = region[0], region[1]
		b1 = np.where((serial<=serial_stop)&(serial>=serial_start))[0]
		if len(b1)==0:
			list1[i,0:2] = [serial_start,serial_stop]
			continue

		t_score = score1[b1]
		t_signal = signal[b1]
		t_predicted_signal = predicted_signal[b1]

		b2 = np.where(t_score>thresh)[0]
		temp1 = [np.max(t_score), np.mean(t_score),len(b2)]
		temp2_1 = [np.max(t_signal), np.min(t_signal), np.median(t_signal)]
		temp2_2 = [np.max(t_predicted_signal), np.min(t_predicted_signal), np.median(t_predicted_signal)]

		b_1 = (temp2_1[-1]>thresh1)
		if temp2_1[-1]<thresh2:
			b_1 = -1

		b_2 = (temp2_2[-1]>thresh1_1)
		if temp2_2[-1]<thresh2_1:
			b_2 = -1

		flag_vec[i] = [b_1,b_2]
		list1[i] = [serial_start,serial_stop]+temp2_1[2:]+temp2_2[2:]+temp1

	return flag_vec, list1

def compare_with_regions_motif1_1(est_data,sel_idvec=[1,1,1,1],sel_column='Q2.adj',thresh1=0.95,config={}):

	chrom, start, stop, serial = np.asarray(est_data['chrom']), np.asarray(est_data['start']), np.asarray(est_data['stop']), np.asarray(est_data['serial'])
	signal = np.asarray(est_data['signal'])
	attention_1 = np.asarray(est_data['predicted_attention'])
	attention1 = np.asarray(est_data[sel_column])
	print(est_data.shape)

	value = np.column_stack((attention_1,attention1))
	seq_list = generate_sequences_chrom(chrom,serial)

	dict1 = dict()
	if sel_idvec[0]>0:
		thresh_vec = [0.9]
		dict2 = compare_with_regions_peak_search1(chrom,start,stop,serial,value,seq_list,thresh_vec)
		dict1[0] = dict2[thresh_vec[0]]

	if sel_idvec[1]>0:
		init_zone = config['init_zone']
		flanking = config['flanking1']
		chrom2,start2,stop2,serial2,annot2 = compare_with_regions_init_search(chrom,start,stop,serial,init_zone,attention1,thresh1,flanking=flanking)
		dict1[1] = [chrom2,start2,stop2,serial2,annot2]

	if sel_idvec[2]>0:
		value = signal
		dict_2 = compare_with_regions_peak_search2(chrom,start,stop,serial,value,seq_list)
		chrom_local,start_local,stop_local,serial_local,annot_local = dict_2[0][1]
		local_peak = [chrom_local,start_local,stop_local]

		flanking = config['flanking1']
		chrom_2,start_2,stop_2,serial_2,annot_2 = compare_with_regions_init_search(chrom,start,stop,serial,local_peak,attention1,thresh1,flanking=flanking)
		dict1[2] = [chrom_2,start_2,stop_2,serial_2,annot_2]

	if sel_idvec[3]>0:
		signal_vec = np.asarray(est_data.loc[:,['signal','predicted_signal']])
		chrom3,start3,stop3,serial3,annot3 = compare_with_regions_signal_search(chrom,start,stop,serial,signal_vec,attention1,thresh1)
		dict1[3] = [chrom3,start3,stop3,serial3,annot3]

	# b1 = np.where(attention1>thresh1)[0]
	# serial1 = serial[b1]
	
	# region_num = len(chrom)
	# label1 = np.zeros(region_num,dtype=np.int32)
	# label1[b1] = -1

	return dict1

def compare_with_regions_motif1_sub1(motif_data,motif_name,est_data,dict1,
										sel_idvec=[2,1,1,1],sel_column='Q2.adj',thresh1=0.95):

	chrom, start, stop, serial = np.asarray(est_data['chrom']), np.asarray(est_data['start']), np.asarray(est_data['stop']), np.asarray(est_data['serial'])
	# attention_1 = np.asarray(est_data['predicted_attention'])
	attention1 = np.asarray(est_data[sel_column])

	b1 = np.where(attention1>thresh1)[0]
	serial1 = serial[b1]
	print(np.max(attention1),np.min(attention1),np.median(attention1),np.mean(attention1))
	
	region_num = len(chrom)
	label1 = np.zeros(region_num,dtype=np.int32)
	num1 = len(sel_idvec)

	if np.sum(sel_idvec)>0:
		label1[b1] = -1
		serial_2 = serial1
		for i in range(num1):
			sel_id = sel_idvec[i]
			if sel_id==0:
				continue
			if i==0:
				t_vec1 = dict1[i][sel_id-1]
			else:
				t_vec1 = dict1[i]

			t_chrom1,t_start1,t_stop1,t_serial1,t_annot1 = t_vec1
			serial_2 = np.intersect1d(serial_2,t_serial1)

		id1 = mapping_Idx(serial,serial_2)
		b1 = np.where(id1>=0)[0]
		if len(b1)!=len(serial_2):
			print('error!',len(b1),len(serial_2))
		id1 = id1[b1]
		label1[id1] = serial_2[b1]+1
	else:
		label1[b1] = np.arange(len(b1))+1

	return label1

# select regions and exclude the regions from the original list of serials
# return: serial_list1: selected regions excluded
#		  serial_list2: selected regions
def select_region(chrom, start, stop, serial, regionlist_filename):

	region_list = pd.read_csv(regionlist_filename,header=None,sep='\t')
	colnames = list(region_list)
	col1, col2, col3 = colnames[0], colnames[1], colnames[2]
	chrom1, start1, stop1 = region_list[col1], region_list[col2], region_list[col3]
	num1 = len(chrom1)
	# serial_list1 = self.serial.copy()
	serial_list2 = []
	for i in range(0,num1):
		b1 = np.where((chrom==chrom1[i])&(start>=start1[i])&(stop<=stop1[i]))[0]
		serial_list2.extend(serial[b1])
		
	print(len(serial),len(serial_list2))
	serial_list1 = np.setdiff1d(serial,serial_list2)

	return serial_list1, serial_list2

# generate sequences
# idx_sel_list: chrom, serial
# seq_list: relative positions
def generate_sequences(idx_sel_list, gap_tol=5, region_list=[]):

	chrom = idx_sel_list[:,0]
	chrom_vec = np.unique(chrom)
	chrom_vec = np.sort(chrom_vec)
	seq_list = []
	# print(len(chrom),chrom_vec)
	for chrom_id in chrom_vec:
		b1 = np.where(chrom==chrom_id)[0]
		t_serial = idx_sel_list[b1,1]
		prev_serial = t_serial[0:-1]
		next_serial = t_serial[1:]
		distance = next_serial-prev_serial
		b2 = np.where(distance>gap_tol)[0]

		if len(region_list)>0:
			b_1 = np.where(region_list[:,0]==chrom_id)[0]
			b2 = np.setdiff1d(b2,region_list[b_1,1])

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

# generate sequences
# idx_sel_list: chrom, serial
# seq_list: relative positions
# consider specific regions
def generate_sequences_1(idx_sel_list, gap_tol=5, region_list=[]):

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

		if len(region_list)>0:
			b_1 = np.where(region_list[:,0]==chrom_id)[0]
			b2 = np.setdiff1d(b2,region_list[b_1,1])
			
		print('gap',len(b2))
		if len(b2)>0:
			t_seq = list(np.vstack((b2[0:-1]+1,b2[1:])).T)
			t_seq.insert(0,np.asarray([0,b2[0]]))
			t_seq.append(np.asarray([b2[-1]+1,len(b1)-1]))
		else:
			t_seq = [np.asarray([0,len(b1)-1])]
		# print(t_seq)
		print(chrom_id,len(t_seq),max(distance))
		seq_list.extend(b1[np.asarray(t_seq)])

	return np.asarray(seq_list)

# importance score
def estimate_regions_1(filename1,thresh=0.975,gap_tol=2):

	data1 = pd.read_csv(filename1,sep='\t')
	colnames1 = list(data1)

	# print(colnames1)
	# data3 = data3.sort_values(by=['serial'])
	num1 = data3.shape[0]
	print(num1)
	# label1 = np.zeros(num1,dtype=np.int32)
	chrom1, start1, stop1 = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop'])
	serial = np.asarray(data1['serial'])
	quantile_vec1 = ['Q1','Q2']
	# attention1 = np.asarray(data3['predicted_attention'])
	# attention2 = np.asarray(data3[quantile_vec1[type_id]])	# ranking
	# sel_column = ['predicted_attention',quantile_vec1[type_id]]
	sel_column = [quantile_vec1[type_id]]
	attention1 = np.asarray(data1.loc[:,sel_column])
	b1 = np.where(attention1>thresh)[0]
	data3 = data1.loc[b1,colnames1]
	n1, n2 = len(b1), len(serial)
	ratio1 = len(b1)/len(serial)
	print('thresh',thresh,n1,n2,ratio1)

	chrom2, start2, stop2 = np.asarray(data3['chrom']), np.asarray(data3['start']), np.asarray(data3['stop'])
	serial2 = np.asarray(data3['serial'])

	return True

# sample regions
# input: filename_list: list of filenames of estimation files
#		 filename2: feature regions
def sample_region_1(filename_list,filename2,output_filename):

	filename1 = filename_list[0]
	compare_with_regions_pre(filename1,filename2,output_filename,tol=0,label_name='label')

def check_width(serial1,serial2,thresh1,type_id=0):

	if type_id==0:
		b1 = np.where(serial2<serial1-thresh1)[0]
		serial2[b1] = serial1[b1]-thresh1
	else:
		b1 = np.where(serial2>serial1+thresh1)[0]
		serial2[b1] = serial1[b1]+thresh1

	return serial2

def peak_region_search_1(x,x1,peaks,serial,width_thresh,thresh_vec,dict1):

	# estimate prominence of peaks
	vec1 =  peak_prominences(x, peaks)
	value1, left1, right1 = vec1[0], vec1[1], vec1[2]

	if len(peaks)==0:
		return dict1

	len1 = right1-left1
	n1 = len(peaks)
	# print(n1,len(serial),serial[0],serial[-1],np.max(len1),np.min(len1))

	left1_ori = check_width(peaks,left1,width_thresh,type_id=0)
	right1_ori = check_width(peaks,right1,width_thresh,type_id=1)

	for thresh in thresh_vec:
		list1 = dict1[thresh]

		b1 = np.where(x1[peaks]>thresh)[0]
		b2 = np.where(x1>thresh)[0]
		n2, n3 = len(b1), len(b2)
		# print(n2,n3,n2/(n1+1e-12),n2/(n3+1e-12))
		peak1, left1, right1 = serial[peaks[b1]], serial[left1_ori[b1]], serial[right1_ori[b1]]
		list1.extend(np.vstack((peak1,left1,right1)).T)
		# print(thresh,len(list1),len(peak1))
		#  print(list1[0:10])

		dict1[thresh] = list1

	return dict1

def peak_region_search(x,x1,peaks,serial,width_thresh,thresh_vec,dict1,type_id2):

	# estimate prominence of peaks
	vec1 =  peak_prominences(x, peaks)
	value1, left1, right1 = vec1[0], vec1[1], vec1[2]

	if len(peaks)==0:
		return dict1

	len1 = right1-left1
	n1 = len(peaks)
	print(n1,len(serial),serial[0],serial[-1],np.max(len1),np.min(len1))

	left1_ori = check_width(peaks,left1,width_thresh,type_id=0)
	right1_ori = check_width(peaks,right1,width_thresh,type_id=1)

	for thresh in thresh_vec:
		temp1 = dict1[thresh]
		list1 = temp1[type_id2]

		b1 = np.where(x1[peaks]>thresh)[0]
		b2 = np.where(x1>thresh)[0]
		n2, n3 = len(b1), len(b2)
		print(n2,n3,n2/(n1+1e-12),n2/(n3+1e-12))
		peak1, left1, right1 = serial[peaks[b1]], serial[left1_ori[b1]], serial[right1_ori[b1]]
		list1.extend(np.vstack((peak1,left1,right1)).T)
		print(thresh,len(list1),type_id2,len(peak1))

		temp1[type_id2] = list1
		dict1[thresh] = temp1

	return dict1

# non-ERCE regions
def query_region2(data1,thresh=10,label_name='label'):

	data3_1 = data1.loc[:,['chrom','start','stop','signal','serial',label_name]]
	serial1, label1 = np.asarray(data3_1['serial']), np.asarray(data3_1[label_name])
	start1, stop1 = np.asarray(data3_1['start']), np.asarray(data3_1['stop'])
	chrom1 = np.asarray(data3_1['chrom'])
	chrom_vec = np.unique(chrom1)

	vec1, vec2 = [], []
	for t_chrom in chrom_vec:
		b1 = np.where(chrom1==t_chrom)[0]
		t_label, t_serial = label1[b1], serial1[b1]
		t_start, t_stop = start1[b1], stop1[b1]

		b2 = np.where(t_label==0)[0]
		t_serial2 = t_serial[b2]
		gap1 = b2[1:]-b2[0:-1]
		gap2 = t_serial2[1:]-t_serial2[0:-1]

		id1 = np.where((gap2>thresh)|(gap1>1))[0]
		# print('gap',len(id1))

		if len(id1)>0:
			t_seq = list(np.vstack((id1[0:-1]+1,id1[1:])).T)
			t_seq.insert(0,np.asarray([0,id1[0]]))
			t_seq.append(np.asarray([id1[-1]+1,len(b2)-1]))
			# vec1.extend(t_seq)
		else:
			t_seq = [np.asarray([0,len(b2)-1])]

		print(b2,len(b2))
		# print(t_seq)
		t_seq = np.asarray(t_seq)
		t_seq = b2[t_seq]
			# vec1.append(t_seq)
		# print(t_seq)
		num1 = len(t_seq)
		print(t_chrom,num1,max(gap1),max(gap2))
		
		for pair1 in t_seq:
			t1, t2 = pair1[0], pair1[1]
			vec2.append([t_start[t1], t_stop[t2], t_serial[t1], t_serial[t2],pair1[0],pair1[1]])

		vec1.extend([t_chrom]*num1)

	fields = ['chrom','start','stop','serial1','serial2','pos1','pos2']
	data1 = pd.DataFrame(columns=fields)
	data1['chrom'] = vec1
	num2 = len(fields)
	data1[fields[1:]] = np.asarray(vec2,dtype=np.int64) 

	return data1

def compare_with_regions_load_1(filename1,run_id,data1=[]):

	if len(data1)==0:
		data1 = pd.read_csv(filename1,sep='\t')

	max_value1 = np.asarray(data1['max'])
	id1_ori = np.where(max_value1!=-1)[0]
	# data1 = data1.loc[id1_ori,:]
	sample_num = len(id1_ori)
	sample_num = data1.shape[0]

	t_vec1 = ['1_0.9','2_0.9','1_0.95','2_0.95']
	colnames = []
	for temp1 in t_vec1:
		colnames.append('%d_%s'%(run_id,temp1))

	colnames.extend(['%d_pvalue'%(run_id),'%d_fdr'%(run_id),'%d_label'%(run_id)])

	data2 = data1.loc[:,colnames]

	max_value =np.asarray(data1['%d_max'%(run_id)])

	chrom, start, stop, region_len = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['length'])

	vec1 = []
	mtx1 = np.asarray(data2)
	id1 = np.where(mtx1[:,4]>0)[0]
	id2 = np.where(mtx1[:,5]>0)[0]
	id1_1 = np.union1d(id1,id2)

	id3_1 = np.where((mtx1[:,2])>0)[0]
	id3_2 = np.where((mtx1[:,3])>0)[0]
	id3_3 = np.union1d(id3_1,id3_2)

	thresh = -5
	id5_1 = np.where((mtx1[:,2]<0)&(mtx1[:,2]>thresh))[0]
	id5_2 = np.where((mtx1[:,3]<0)&(mtx1[:,3]>thresh))[0]
	id5_3 = np.union1d(id5_1,id5_2)

	

	id1_2 = np.union1d(id1_1,id3_3)
	id1_3 = np.union1d(id1_2,id5_3)

	id1_2_1 = np.union1d(id1,id3_3)
	id1_3_1 = np.union1d(id1_2_1,id5_3)

	id1_2_2 = np.union1d(id2,id3_3)
	id1_3_2 = np.union1d(id1_2_2,id5_3)

	thresh1 = 0.975
	id6 = np.where(max_value>thresh1)[0]

	vec1.append([len(id1),len(id2),len(id6),len(id3_1),len(id3_2),len(id5_3),len(id1_2),len(id1_3),len(id1_3)])
	vec1.append([len(id1),len(id2),len(id6),len(id3_1),len(id3_2),len(id5_3),len(id1_2_1),len(id1_2_2),len(id1_3)])
	vec1.append([len(id1),len(id2),len(id6),len(id3_1),len(id3_2),len(id5_3),len(id1_2_2),len(id1_3_2),len(id1_3_2)])

	ratio_vec1 = np.asarray(vec1)*1.0/sample_num
	print(ratio_vec1)

	id_vec = (id1,id2,id6,id3_1,id3_2,id5_3,id1_2_1,id1_2_2,id1_3_1,id1_3_2,id1_3)

	return ratio_vec1, data1, id_vec

# sample regions randomly to compare with elements
# compare with regions random: sample by region length
def compare_with_regions_random3(filename1,filename2,type_id1,region_filename='',output_filename='',tol=2,
									sample_num=2000,type_id=1,label_name='label',
									thresh_vec = [0.9,0.95,0.975,0.99,0.995],
									thresh_fdr = 0.05, region_data=[],
									quantile_vec1 = ['Q1','Q2']):
	
	if region_filename=='':
		region_filename = 'region1.%d.tol%d.1.txt'%(type_id1,tol)

	region_list = []

	if output_filename=='':
		output_filename = 'region1.1.%d.tol%d.1.txt'%(type_id1,tol)
	data1, data2 = compare_with_regions_pre(filename1,filename2,output_filename,tol,label_name,
												save_mode=1,region_data=region_data)

	if os.path.exists(region_filename)==True:
		region_list1 = pd.read_csv(region_filename,sep='\t')
		data2 = pd.read_csv(filename2,header=None,sep='\t')
		print(region_filename,filename2)
	else:
		label_name1 = 'label_tol%d'%(tol)
		print('data1',data1.shape)
		print(list(data1))
		region_list1 = query_region2(data1,label_name=label_name1)
		region_list1.to_csv(region_filename,index=False,sep='\t')

	region_chrom, pair1 = np.asarray(region_list1['chrom']), np.asarray(region_list1.loc[:,['pos1','pos2']])

	# load ERCE files
	# data2 = pd.read_csv(filename2,header=None,sep='\t')
	colnames2 = list(data2)
	col1, col2, col3 = colnames2[0], colnames2[1], colnames2[2]
	chrom2, start2, stop2 = data2[col1], data2[col2], data2[col3]
		
	data3 = data1
	colnames1 = list(data3)
	# print(colnames1)
	# data3 = data3.sort_values(by=['serial'])
	num1 = data3.shape[0]
	print(num1)
	label1 = np.zeros(num1,dtype=np.int32)
	chrom1, start1, stop1 = np.asarray(data3['chrom']), np.asarray(data3['start']), np.asarray(data3['stop'])
	# quantile_vec1 = ['Q1','Q2']
	# attention1 = np.asarray(data3['predicted_attention'])
	# attention2 = np.asarray(data3[quantile_vec1[type_id]])	# ranking
	sel_column = ['predicted_attention',quantile_vec1[type_id]]
	attention1 = np.asarray(data3.loc[:,sel_column])

	chrom_vec = np.unique(chrom2)
	chrom_num = len(chrom_vec)
	chrom_size1 = len(chrom1)
	bin_size = stop1[1]-start1[1]

	num2 = len(chrom2)
	sel_num = len(sel_column)
	thresh_num = len(thresh_vec)
	sel_num1 = 2*sel_num + thresh_num
	score1 = -np.ones((num2,sel_num1*3),dtype=np.float32) # sampled region: mean, std
	score2 = -np.ones((num2,sel_num1),dtype=np.float32) # sampled region: mean, std
	vec1, vec1_1 = [], []

	for t_chrom in chrom_vec:
		# t_chrom = chrom_vec[i]
		b2 = np.where(chrom2==t_chrom)[0]
		t_num2 = len(b2)
		b1 = np.where(chrom1==t_chrom)[0]
		t_num1 = len(b1)

		# print(t_chrom,t_num1,t_num2)

		if t_num1==0 or t_num2==0:
			print('chromosome not found', t_chrom)
			continue
			
		t_chrom_size = t_num1
		# print('sample regions %d'%(sample_num),t_chrom,t_chrom_size)

		t_chrom1, t_start1, t_stop1, t_attention1 = np.asarray(chrom1[b1]), np.asarray(start1[b1]), np.asarray(stop1[b1]), np.asarray(attention1[b1])
		t_chrom_region, t_start_region, t_stop_region = np.asarray(chrom2[b2]), np.asarray(start2[b2]), np.asarray(stop2[b2])

		t_region_len = t_stop_region-t_start_region
		region_len_vec = np.unique(t_region_len)
		# print('region_len_vec',t_chrom,len(region_len_vec),region_len_vec)

		region_sample_dict = dict()
		b_1 = np.where(region_chrom==t_chrom)[0]
		region_list = pair1[b_1]

		for region_len in region_len_vec:
			region_len_tol = region_len + 2*tol*bin_size
			vec2 = compare_with_regions_sub3(t_chrom1,t_start1,t_stop1,t_attention1,
									sample_num,region_len_tol,region_list,bin_size,tol,
									thresh_vec)
			region_sample_dict[region_len] = vec2
			# print('region_len',region_len,region_len_tol,vec2.shape)

		for l in range(t_num2):

			t_chrom2, t_start2, t_stop2 = t_chrom_region[l], t_start_region[l], t_stop_region[l]
			# if l%100==0:
			# 	print(t_chrom2, t_start2, t_stop2)

			# print(t_stop,t_start)
			region_len_ori = t_stop2-t_start2
			region_len_ori1 = (region_len_ori)/bin_size

			tol1 = tol
			t_start_2 = max(0,t_start2-tol1*bin_size)
			t_stop_2 = min(t_stop1[-1],t_stop2+tol1*bin_size)
			len1 = (t_stop_2-t_start_2)/bin_size

			b1_ori = np.where((t_start1<t_stop_2)&(t_stop1>t_start_2))[0]
			if len(b1_ori)==0:
				continue

			b1_ori = b1[b1_ori]
			i1 = b2[l]
			for l1 in range(sel_num):
				id2 = 2*l1
				attention_score = attention1[b1_ori,l1]
				#attention_score = t_attention1[b1_ori,l1]
				score1[i1,id2:(id2+2)] = [np.max(attention_score),np.mean(attention_score)]

			t_vec3 = []
			attention_score = attention1[b1_ori,sel_num-1]
			#attention_score = t_attention1[b1_ori,sel_num-1]
			for l2 in range(thresh_num):
				id1 = np.where(attention_score>thresh_vec[l2])[0]
				score1[i1,(2*sel_num+l2)] = len(id1)/len1

			vec2 = region_sample_dict[region_len]
			# print(vec2.shape)
			sample_num1, num3 = vec2.shape[0], vec2.shape[1]
			assert num3==(sel_num1+2)
			vec3 = []
			for l1 in range(0,sel_num1):
				value1 = vec2[:,l1+2]
				t_score_mean1, t_score_std1 = np.mean(value1), np.std(value1)
				vec3.extend([t_score_mean1,t_score_std1])
				t1 = np.where(value1>score1[i1,l1]-1e-06)[0]
				score2[i1,l1] = len(t1)/sample_num1

			score1[i1,sel_num1:] = vec3

			# if i%100==0:
			# 	print(i,score1[i],len(vec2))
			if i1%1000==0:
				# print(t_chrom,i1,score1[i1],score2[i1],len(vec2),vec2.shape)
				print(t_chrom2,t_start2,t_stop2,(t_stop2-t_start2)/bin_size,i1,score1[i1],score2[i1],len(vec2),vec2.shape)
			# if l>10:
			# 	break

	# find chromosomes with estimation
	# serial1 = find_serial(chrom2,chrom_num=len(np.unique(chrom1)))
	# score2, score1 = score2[serial1], score1[serial1]
	# data2 = data2.loc[serial1,:]

	# fdr correction
	b1 = np.where(score2[:,0]>=0)[0]
	n1, n2 = score2.shape[0], score2.shape[1]
	score2_fdr = -np.ones((n1,n2))

	for i in range(n2):
		vec2 = multipletests(score2[b1,i],alpha=thresh_fdr,method='fdr_bh')
		score2_fdr[b1,i] = vec2[1]

	return score2,score2_fdr,score1,data2

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

def one_hot_encoding(seq_data,serial):

	enc=OneHotEncoder(categories=[['A','C','G','T']],sparse=False,dtype=np.int,handle_unknown='ignore')

	n_sample = len(seq_data)
	seq_len = len(seq_data[0])
	list1 = np.zeros((n_sample,seq_len,4),dtype=np.int8)
	list2 = -np.ones((n_sample,2),dtype=np.int64)
	# list1, list2 = [], []
	cnt1, cnt2 = 0, 0
	print('region length', seq_len, n_sample)
	print(serial)
	for i in range(n_sample):
		str1, serial1 = seq_data[i], serial[i]
		seq_len1 = len(str1)
		n1 = str1.count('N')
		if seq_len1!=seq_len:
			continue
		if n1>seq_len1*0.1:
			cnt1 = cnt1+1
			if n1==seq_len1:
				cnt2 = cnt2+1
				continue

		str1 = np.asarray(list(str1))
		encoding = enc.fit_transform(str1[:,np.newaxis])
		list1[i] = encoding
		list2[i] = [serial1,n1]
		if i%10000==0:
			print(i,serial1)

	list1 = np.asarray(list1)
	list2 = np.asarray(list2)
	b1 = np.where(list2[:,0]>=0)[0]
	list1 = list1[b1]
	list2 = list2[b1]
	print('one hot encoding',n_sample,cnt1,cnt2,len(list2),list1.shape,cnt1,cnt2)

	return list1, list2

def aver_overlap_value(position, start_vec, stop_vec, signal, idx):

	t_start1, t_stop1 = start_vec[idx], stop_vec[idx]
	t_len1 = t_stop1 - t_start1
	t_len2 = (position[1] - position[0])*np.ones(len(idx))
	temp1 = np.vstack((t_stop1-position[0],position[1]-t_start1,t_len1,t_len2))
	temp1 = np.min(temp1,axis=0)
	aver_value = np.dot(signal[idx],temp1)*1.0/np.sum(temp1)

	return aver_value	

def search_region(position, start_vec, stop_vec, m_idx, start_idx):

	id1 = start_idx
	vec1 = []
	while (id1<=m_idx) and (stop_vec[id1]<=position[0]):
		id1 += 1

	while (id1<=m_idx) and (stop_vec[id1]>position[0]) and (start_vec[id1]<position[1]):
		vec1.append(id1)
		id1 += 1

	if len(vec1)>0:
		start_idx1 = vec1[-1]
	else:
		start_idx1 = id1

	return np.asarray(vec1), start_idx1	

def search_region_include(position, start_vec, stop_vec, m_idx, start_idx):

	id1 = start_idx
	vec1 = []
	while (id1<=m_idx) and (start_vec[id1]<position[0]):
		id1 += 1

	while (id1<=m_idx) and (stop_vec[id1]<=position[1]) and (start_vec[id1]>=position[0]):
		vec1.append(id1)
		id1 += 1

	return np.asarray(vec1), id1

# load chromosome 1
def query_region1(data1,chrom_name,chrom_size,bin_size,type_id1=0):

	# data1 = pd.read_csv(filename,header=None)
	chrom, start, stop, signal = np.asarray(data1[0]), np.asarray(data1[1]), np.asarray(data1[2]), np.asarray(data1[3])
	region_len = stop-start
	id1 = np.where(chrom==chrom_name)[0]
	print("chrom",chrom_name,len(id1))
	t_stop = stop[id1[-1]]

	# bin_size = 200
	num_region = int(chrom_size*1.0/bin_size)
	serial_vec = np.zeros(num_region)
	signal_vec = np.zeros(num_region)
	start_vec = np.asarray(range(0,num_region))*bin_size
	stop_vec = start_vec + bin_size
	chrom1, start1, stop1, signal1 = chrom[id1], start[id1], stop[id1], signal[id1]
	
	threshold = 1e-04
	b1 = np.where(signal1<=threshold)[0]
	b2 = np.where(signal1>threshold)[0]
	
	region_len = stop1-start1
	len1 = np.sum(region_len[b1])
	len2 = np.sum(region_len[b2])
	ratio1 = len1*1.0/np.sum(region_len)
	ratio2 = 1-ratio1
	print('chrom, ratio1 ratio2',chrom_name, ratio1, ratio2, len(b1), len(b2))

	list1 = []
	count = 0
	start_idx = 0
	print("number of regions", len(b1))
	count2 = 0
	time_1 = time.time()
	m_idx = len(start_vec)-1
	for id2 in b1:
		# print(id2)
		t_start, t_stop = start1[id2], stop1[id2] # position of zero region
		position = [t_start,t_stop]
		id3 = []
		if start_idx<=m_idx:
			id3, start_idx = search_region_include(position, start_vec, stop_vec, m_idx, start_idx)
		# print(count,t_start,t_stop,t_stop-t_start,start_idx,len(id3))
		if len(id3)>0:
			# if count%500000==0:
			# 	print(count,t_start,t_stop,len(id3),start_idx,start_vec[id3[0]],stop_vec[id3[-1]])
			# 	print(count,t_start,t_stop,t_stop-t_start,id3[0],id3[-1],start_vec[id3[0]],stop_vec[id3[-1]],len(id3),len(id3)*bin_size)
			# if count>500:
			# 	break
			list1.extend(id3)
			count += 1
		else:
			count2 += 1

	time_2 = time.time()
	print("time: ", time_2-time_1)

	list2 = np.setdiff1d(range(0,num_region),list1)
	print("zero regions",len(list1), len(list2))
	print("zero regions 2", count, count2)

	# return False

	start_idx = 0
	count = 0
	count1, count2 = 0, 0
	time_1 = time.time()
	# start1_ori, stop1_ori = start1.copy(), stop1.copy()
	# start1, stop1 = start1[b2], stop1[b2]	# regions with signal values higher than the threshold
	list1, list2 = np.asarray(list1), np.asarray(list2)
	num2 = len(list2)

	# type_id1 = 0
	m_idx = len(start1)-1
	for id1 in list2:
		t_start, t_stop = start_vec[id1], stop_vec[id1]
		position = [t_start,t_stop]
		if start_idx<=m_idx:
			vec1, start_idx = search_region(position, start1, stop1, m_idx, start_idx)

		if len(vec1)>0:
			if type_id1==0:
				aver_value = aver_overlap_value(position, start1, stop1, signal1, vec1)
				signal_vec[id1] = aver_value
			else:
				signal_vec[id1] = np.max(signal1[vec1])
			serial_vec[id1] = 1
			count += 1
			# if count%500000==0:
			# 	id_1, id_2 = vec1[0], vec1[-1]
			# 	print(count,t_start,t_stop,signal_vec[id1],id_1,id_2,start1[id_1],stop1[id_1],start1[id_2],stop1[id_2],len(vec1))

		if start_idx>m_idx:
			break

		else:
			count2 += 1

	time_2 = time.time()
	print("time: ", time_2-time_1)
	print("serial, signal", np.max(serial_vec), np.max(signal_vec), count2)
	return serial_vec, signal_vec

	# return True

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

def generate_serial_start(filename1,chrom,start,stop,chrom_num,type_id=0):

	# chrom_vec = np.sort(np.unique(chrom))
	# print(chrom_vec)
	chrom_vec = []
	for i in range(1,chrom_num+1):
		chrom_vec.append('chr%d'%(i))
	if type_id==0:
		chrom_vec += ['chrX']
		chrom_vec += ['chrY']
	print(chrom_vec)
	print(chrom)
	print(len(chrom))
	
	# filename1 = '/volume01/yy3/seq_data/genome/hg38.chrom.sizes'
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	ref_chrom, chrom_size = np.asarray(data1[0]), np.asarray(data1[1])
	serial_start = 0
	serial_vec = -np.ones(len(chrom),dtype=np.int64)
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
			if len(b2)>0:
				serial = np.int64(start[b2]/bin_size)+serial_start
				serial_vec[b2] = serial
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
	
	# filename1 = './genome/hg38.chrom.sizes'
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

def get_model2a_sequential(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (input_shape,feature_dim))
	lr = config['lr']
	activation = config['activation']

	model = keras.models.Sequential()
	# model.add(Bidirectional(LSTM(input_shape=(10,feature_dim),units=output_dim,return_sequences = True,recurrent_dropout = 0.1)))
	model.add(Bidirectional(LSTM(units=output_dim,return_sequences = True,recurrent_dropout = 0.1),input_shape=(input_shape,feature_dim)))
	# model.add(LSTM(units=output_dim,return_sequences = True,recurrent_dropout = 0.1,input_shape=(input_shape,feature_dim)))
	# model.add(Input(shape = (input_shape,feature_dim)))
	# model.add(Bidirectional(LSTM(units=output_dim,return_sequences = True,recurrent_dropout = 0.1)))
	model.add(LayerNormalization())
	model.add(Flatten())

	if fc1_output_dim>0:
		model.add(Dense(units=fc1_output_dim))
		model.add(BatchNormalization())
		model.add(Activation(activation))
		model.add(Dropout(0.5))
	else:
		pass

	model.add(Dense(units=1))
	model.add(BatchNormalization())
	model.add(Activation("sigmoid"))

	adam = Adam(lr = lr)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()

	return model

def get_model2a1_sequential(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	# input1 = Input(shape = (input_shape,feature_dim))
	lr = config['lr']
	
	model = keras.models.Sequential()
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1),input_shape=(None, feature_dim)))
	model.add(LayerNormalization())

	if fc1_output_dim>0:
		model.add(Dense(units=fc1_output_dim))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
	else:
		pass

	model.add(Dense(units=1))
	model.add(BatchNormalization())
	model.add(Activation("sigmoid"))

	adam = Adam(lr = lr)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()

	return model

def get_model2a1_attention_1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']

	input1 = Input(shape = (n_steps,feature_dim))

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	activation = config['activation']

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	# x1 = Activation('tanh',name='activation')(x1)
	# x1 = Flatten()(x1)
	if activation!='':
		x1 = Activation(activation,name='activation')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# x_1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# x1 = x_1[0]
	# attention = x_1[1]
	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim,name='dense1')(x1)
		dense1 = BatchNormalization(name='batchnorm1')(dense1)
		dense1 = Activation(activation,name='activation1')(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
		output_dim1 = fc1_output_dim
	else:
		dense_layer_output = x1
		output_dim1 = 2*output_dim

	units_1 = config['units1']
	if units_1>0:
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)
		dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_1)
	else:
		dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(input1)
	
	attention1 = Flatten()(dense_layer_2)
	attention1 = Activation('softmax',name='attention1')(attention1)
	attention1 = RepeatVector(output_dim1)(attention1)
	attention1 = Permute([2,1])(attention1)
	layer_1 = Multiply()([dense_layer_output, attention1])
	dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)
	
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(dense_layer_output)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation("sigmoid",name='activation2')(output)

	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = lr)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()

	return model

def get_model2a1_attention_2(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	input1 = Input(shape = (n_steps,feature_dim))

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	# x1 = Activation('tanh',name='activation')(x1)	
	# x1 = Flatten()(x1)
	if activation!='':
		x1 = Activation(activation,name='activation')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# x_1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# x1 = x_1[0]
	# attention = x_1[1]
	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim,name='dense1')(x1)
		dense1 = BatchNormalization(name='batchnorm1')(dense1)
		dense1 = Activation(activation,name='activation1')(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
		output_dim1 = fc1_output_dim
	else:
		dense_layer_output = x1
		output_dim1 = 2*output_dim

	attention1 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_output)
	attention1 = Flatten()(attention1)
	attention1 = Activation('softmax',name='attention1')(attention1)
	attention1 = RepeatVector(output_dim1)(attention1)
	attention1 = Permute([2,1])(attention1)
	layer_1 = Multiply()([dense_layer_output, attention1])
	dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)
	
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(dense_layer_output)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation("sigmoid",name='activation2')(output)

	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = lr)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()

	return model

def compute_mean_std(run_id, filename1, config={}):

	data1 = pd.read_csv(filename1,sep='\t')
	fields = ['chrom','start','stop','signal']

	chrom, signal = np.asarray(data1['chrom']), np.asarray(data1['signal'])
	predicted_signal = np.asarray(data1['predicted_signal'])

	# if (np.min(predicted_signal)<-0.5) and (np.min(signal)>-0.5):
	# 	predicted_signal = 0.5*predicted_signal+0.5

	# if (np.min(predicted_signal)>-0.5) and (np.min(signal)<-0.5):
	# 	signal = 0.5*signal+0.5

	chrom_vec = np.unique(chrom)
	chrom_num = len(chrom_vec)
	chrom_vec1 = np.zeros(chrom_num,dtype=np.int32)

	for i in range(chrom_num):
		chrom_id = chrom_vec[i]
		id1 = chrom_id.find('chr')
		try:
			chrom_vec1[i] = int(chrom_id[id1+3:])
		except:
			chrom_vec1[i] = i+1

	id1 = np.argsort(chrom_vec1)
	chrom_vec = chrom_vec[id1]
	chrom_vec1 = chrom_vec1[id1]
	print(chrom_vec)
	print(chrom_vec1)

	vec1 = score_2a(signal, predicted_signal)
	mtx = np.zeros((chrom_num+1,len(vec1)))
	field_num = len(vec1)
	mtx[-1] = vec1

	for i in range(chrom_num):
		t_chrom = chrom_vec[i]
		b = np.where(chrom==t_chrom)[0]
		vec1 = score_2a(signal[b], predicted_signal[b])
		print(t_chrom,vec1)
		mtx[i] = vec1

	# fields = ['run_id','chrom','mse','pearsonr','pvalue1','explained_variance',
	# 			'mean_abs_err','median_abs_err','r2','spearmanr','pvalue2']
	fields = ['run_id','method','celltype','chrom','mse','pearsonr','pvalue1','explained_variance',
				'mean_abs_err','median_abs_err','r2','spearmanr','pvalue2']

	data1 = pd.DataFrame(columns=fields)
	num2 = chrom_num+1
	cell_type1, method1 = config['cell_type1'], config['method1']
	data1['run_id'] = [run_id]*num2
	data1['method'] = [method1]*num2
	data1['celltype'] = [cell_type1]*num2
	data1['chrom'] = list(chrom_vec)+['-1']
	mtx = np.asarray(mtx,dtype=np.float32)
	for i in range(field_num):
		data1[fields[i+4]] = mtx[:,i]

	if 'chrom_vec1_pre' in config:
		chrom_vec1_pre = config['chrom_vec1_pre']
		train_chrom_num = len(chrom_vec)-len(chrom_vec1_pre)
		data1['train_chrNum'] = [train_chrom_num]*num2
	
	if 'train_chrNum' in config:
		train_chrom_num = config['train_chrNum']
		data1['train_chrNum'] = [train_chrom_num]*num2

	return data1

# construct gumbel selector 1
def construct_gumbel_selector1(input1,config,number=1,type_id=1):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation1 = config['activation']
	# activation1 = 'relu'
	feature_dim_vec1 = config['feature_dim_vec']
	if 'local_conv_size' in config:
		local_conv_size = config['local_conv_size']
	else:
		local_conv_size = 3

	# input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		# encode the input, shape: (batch_size,n_steps,units_1)
		dense_layer_1 = Dense(units_1,name='dense1_gumbel_%d'%(number))(input1)
		dense_layer_1 = BatchNormalization()(dense_layer_1)
		dense_layer_1 = Activation(activation1,name = 'dense_gumbel_%d'%(number))(dense_layer_1)
	else:
		dense_layer_1 = input1

	# default: n_filter1:50, dim1:25, n_filter2: 50, dim2: 25, n_local_conv: 0, concat: 0
	# n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0], feature_dim_vec1[1], feature_dim_vec1[2], feature_dim_vec1[3], feature_dim_vec1[4], feature_dim_vec1[5]
	n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
	if n_filter1>0:
		# layer_1 = Conv1D(n_filter1, local_conv_size, padding='same', activation=activation1, strides=1, name = 'conv1_gumbel_%d'%(number))(dense_layer_1)
		layer_1 = Conv1D(n_filter1, local_conv_size, padding='same', strides=1, name = 'conv1_1_gumbel_%d'%(number))(dense_layer_1)
		layer_1 = BatchNormalization()(layer_1)
		layer_1 = Activation(activation1, name='conv1_gumbel_%d'%(number))(layer_1)
	else:
		layer_1 = dense_layer_1

	local_info = layer_1
	if n_local_conv>0:
		for i in range(n_local_conv):
			local_info = Conv1D(n_filter2, local_conv_size, padding='same', activation=activation1, strides=1, name = 'conv%d_gumbel_%d'%(i+2,number))(local_info)

	# global info, shape: (batch_size, feature_dim)
	if concat>0:
		x1 = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_%d'%(number))(layer_1)
		if dim1>0:
			global_info = Dense(dim1, name = 'new_dense_%d'%(number), activation=activation1)(x1)
		else:
			global_info = x1

		# concatenated feature, shape: (batch_size, n_steps, dim1+dim2)
		x2 = Concatenate_1()([global_info,local_info])
	else:
		x2 = local_info

	# x2 = Dropout(0.2, name = 'new_dropout_2')(x2)
	# current configuration: dense1 + conv1 + conv2
	if dim2>0:
		x2 = Conv1D(dim2, 1, padding='same', activation=activation1, strides=1, name = 'conv%d_gumbel_%d'%(n_local_conv+2,number))(x2)
	if ('batchnorm1' in config) and config['batchnorm1']==1:
		x2 = TimeDistributed(BatchNormalization(),name ='conv%d_gumbel_bn%d'%(n_local_conv+2,number))(x2)

	if 'regularizer1' in config:
		regularizer1 = config['regularizer1']
	else:
		# regularizer1 = 1e-05
		regularizer1 = 0

	if 'regularizer2' in config:
		regularizer2 = config['regularizer2']
	else:
		regularizer2 = 1e-05

	activation3 = config['activation3']
	print(activation3)

	# type_id: 1: not using regularization (need to be changed)
	# type_id: 2: using regularization and not using regularization
	if type_id==1:
		# x2 = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv5_gumbel_%d'%(number))(x2)
		# x2 = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'logits_T_%d'%(number))(x2)
		x2 = TimeDistributed(Dense(1),name='dense_1_%d'%(number))(x2)
		x2 = TimeDistributed(BatchNormalization(),name='batchnorm_1_%d'%(number))(x2)
		x2 = TimeDistributed(Activation(activation3),name='logits_T_%d'%(number))(x2)
	elif type_id==2:
		if activation3=='ReLU' or activation3=='relu':
			x2 = TimeDistributed(Dense(1,activation='linear',
									kernel_regularizer=regularizers.l2(regularizer2),
									activity_regularizer=regularizers.l1(regularizer1)),
									name='dense_1_%d'%(number))(x2)
			# x2 = Dense(1,name='dense_1_%d'%(number),activation='linear',
			# 				kernel_regularizer=regularizers.l2(regularizer2),
			# 				activity_regularizer=regularizers.l1(regularizer1))(x2)

			# x2 = TimeDistributed(Activation(activation2,name='activation_1_%d'%(number)))(x2)
			if not('batchnorm2' in config) or config['batchnorm2']==1:
				x2 = TimeDistributed(BatchNormalization(),name='batchnorm_1_%d'%(number))(x2)
			if activation3=='ReLU':
				# thresh1, thresh2 = 1.0-1e-07, 1e-07
				thresh1, thresh2 = 1.0, 0.0
				print(thresh1, thresh2)
				x2 = TimeDistributed(ReLU(max_value=thresh1,threshold=thresh2),name='logits_T_%d'%(number))(x2)
				# x2 = TimeDistributed(ReLU(max_value=thresh1),name='logits_T_%d'%(number))(x2)
			else:
				# x2 = TimeDistributed(BatchNormalization(),name='batchnorm_1_%d'%(number))(x2)
				x2 = TimeDistributed(Activation('relu'),name='logits_T_%d'%(number))(x2)
			# x2 = ReLU(max_value=thresh1,threshold=thresh2,name='logits_T_%d'%(number))(x2)
			# x2 = Activation(Lambda(lambda x: relu(x, max_value=1.0)))(x2)
		# elif activation2=='sigmoid':
		# 	x2 = TimeDistributed(Dense(1,name='dense_1_%d'%(number)))(x2)
		# 	x2 = TimeDistributed(BatchNormalization(name='batchnorm_1_%d'(number)))(x2)
		# 	x2 = TimeDistributed(Activation(activation2,name='activation_1_%d'%(number)))(x2)
		else:
			# if ('regularizer_1' in config) and (config['regularizer_1']==1):
			# 	x2 = TimeDistributed(Dense(1,
			# 						activation=activation3,
			# 						kernel_regularizer=regularizers.l2(regularizer2),
			# 						activity_regularizer=regularizers.l1(regularizer1),
			# 						),name='logits_T_%d'%(number))(x2)

			# x2 = TimeDistributed(Activation(activation2,name='activation_1_%d'%(number)))(x2)
			flag1 = 0
			# if ('regularizer_1' in config) and (config['regularizer_1']==1):
			# 	print(regularizer1,regularizer2)
			# 	if regularizer1>0:
			# 		print('regularization after activation',activation3)
			#		flag1 = 1
			if ('regularizer_1' in config):
				print(regularizer1,regularizer2)
				if config['regularizer_1']==1:
					if regularizer1>0:
						print('regularization after activation',activation3)
						flag1 = 1
				else:
					flag1 = config['regularizer_1']
			
			if flag1==1:
				x2 = TimeDistributed(Dense(1,
									activation=activation3,
									kernel_regularizer=regularizers.l2(regularizer2),
									activity_regularizer=regularizers.l1(regularizer1)
									),name='logits_T_%d'%(number))(x2)
				# x2 = TimeDistributed(Activation(activation3,
				# 					activity_regularizer=regularizers.l1(regularizer1)),
				# 					name='logits_T_%d'%(number))(x2)
			else:
				x2 = TimeDistributed(Dense(1,
									kernel_regularizer=regularizers.l2(regularizer2)
									),name='dense_1_%d'%(number))(x2)
				# if regularizer2>0:
				# 	x2 = TimeDistributed(Dense(1,
				# 					kernel_regularizer=regularizers.l2(regularizer2),
				# 					),name='dense_1_%d'%(number))(x2)
				# else:
				# 	x2 = TimeDistributed(Dense(1),name='dense_1_%d'%(number))(x2)
				# if flag1!=2:
				# 	x2 = TimeDistributed(BatchNormalization(),name='batchnorm_1_%d'%(number))(x2)
				x2 = TimeDistributed(BatchNormalization(),name='batchnorm_1_%d'%(number))(x2)
				x2 = TimeDistributed(Activation(activation3),name='logits_T_%d'%(number))(x2)
			# elif 'regularizer_1' in config and config['regularizer_1']==2:
			# 		x2 = TimeDistributed(Dense(1,
			# 							kernel_regularizer=regularizers.l2(regularizer2),
			# 							),name='dense_1_%d'%(number))(x2)
			# 		x2 = TimeDistributed(BatchNormalization(),name='batchnorm_1_%d'%(number))(x2)
			# 		x2 = TimeDistributed(Activation(activation3),name='logits_T_%d'%(number))(x2)
			# else:
			# 	x2 = TimeDistributed(Dense(1),name='dense_1_%d'%(number))(x2)
			# 	x2 = TimeDistributed(BatchNormalization(),name='batchnorm_1_%d'%(number))(x2)
			# 	x2 = TimeDistributed(Activation(activation3),name='logits_T_%d'%(number))(x2)
	else:
		pass

	return x2

def construct_gumbel_selector1_1(input1,config,number,type_id=2):

	if 'regularizer1' in config:
		regularizer1 = config['regularizer1']
	else:
		regularizer1 = 1e-05

	if 'regularizer2' in config:
		regularizer2 = config['regularizer2']
	else:
		regularizer2 = 1e-05

	activation3 = config['activation3']
	print(activation3)

	# type_id: 1: not using regularization (need to be changed)
	# type_id: 2: using regularization and not using regularization
	x2 = input1
	if type_id==1:
		# x2 = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv5_gumbel_%d'%(number))(x2)
		# x2 = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'logits_T_%d'%(number))(x2)
		x2 = TimeDistributed(Dense(1),name='dense_2_local_%d'%(number))(x2)
		x2 = TimeDistributed(BatchNormalization(),name='batchnorm_2_local_%d'%(number))(x2)
		x2 = TimeDistributed(Activation(activation3),name='logits_T_local_%d'%(number))(x2)
	elif type_id==2:
		if activation3=='ReLU' or activation3=='relu':
			x2 = TimeDistributed(Dense(1,activation='linear',
									kernel_regularizer=regularizers.l2(regularizer2),
									activity_regularizer=regularizers.l1(regularizer1)),
									name='dense_1_local_%d'%(number))(x2)

			if not('batchnorm2' in config) or config['batchnorm2']==1:
				x2 = TimeDistributed(BatchNormalization(),name='batchnorm_2_local_%d'%(number))(x2)
			if activation3=='ReLU':
				# thresh1, thresh2 = 1.0-1e-07, 1e-07
				thresh1, thresh2 = 1.0, 0.0
				print(thresh1, thresh2)
				x2 = TimeDistributed(ReLU(max_value=thresh1,threshold=thresh2),name='logits_T_local_%d'%(number))(x2)
				# x2 = TimeDistributed(ReLU(max_value=thresh1),name='logits_T_%d'%(number))(x2)
			else:
				# x2 = TimeDistributed(BatchNormalization(),name='batchnorm_1_%d'%(number))(x2)
				x2 = TimeDistributed(Activation('relu'),name='logits_T_local_%d'%(number))(x2)
		else:
			flag1 = 0
			if ('regularizer_1' in config):
				print(regularizer1,regularizer2)
				if config['regularizer_1']==1:
					if regularizer1>0:
						print('regularization after activation',activation3)
						flag1 = 1
				else:
					flag1 = config['regularizer_1']
			
			if flag1==1:
				x2 = TimeDistributed(Dense(1,
									activation=activation3,
									kernel_regularizer=regularizers.l2(regularizer2),
									activity_regularizer=regularizers.l1(regularizer1),
									),name='logits_T_local_%d'%(number))(x2)
			else:
				x2 = TimeDistributed(Dense(1,
									kernel_regularizer=regularizers.l2(regularizer2)),name='dense_2_local_%d'%(number))(x2)
				if flag1!=2:
					x2 = TimeDistributed(BatchNormalization(),name='batchnorm_2_local_%d'%(number))(x2)
				x2 = TimeDistributed(Activation(activation3),name='logits_T_local_%d'%(number))(x2)
	else:
		pass

	return x2

# construct gumbel selector 1
def construct_gumbel_selector1_sequential(input1,config,number=1,type_id=2):

	feature_dim, output_dim = config['feature_dim'], config['output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation1 = config['activation']
	# activation1 = 'relu'
	feature_dim_vec1 = config['feature_dim_vec']
	if 'local_conv_size' in config:
		local_conv_size = config['local_conv_size']
	else:
		local_conv_size = 3

	# input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		# encode the input, shape: (batch_size,n_steps,units_1)
		dense_layer_1 = TimeDistributed(Dense(units_1),name='dense1_gumbel_local_%d'%(number))(input1)
		dense_layer_1 = TimeDistributed(BatchNormalization())(dense_layer_1)
		dense_layer_1 = TimeDistributed(Activation(activation1),name = 'dense_gumbel_local_%d'%(number))(dense_layer_1)
	else:
		dense_layer_1 = input1

	# default: n_filter1:50, dim1:25, n_filter2: 50, dim2: 25, n_local_conv: 0, concat: 0
	# n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0], feature_dim_vec1[1], feature_dim_vec1[2], feature_dim_vec1[3], feature_dim_vec1[4], feature_dim_vec1[5]
	n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
	if n_filter1>0:
		# layer_1 = TimeDistributed(Conv1D(n_filter1, local_conv_size, padding='same', 
		# 			activation=activation1, strides=1), name = 'conv1_gumbel_local_%d'%(number))(dense_layer_1)
		x1 = TimeDistributed(Conv1D(n_filter1, local_conv_size, 
								padding='same', strides=1))(input1)
		x1 = TimeDistributed(BatchNormalization(),name='batchnorm_1_local_%d'%(number))(x1)
		layer_1 = TimeDistributed(Activation(activation1),name = 'conv1_gumbel_local_%d'%(number))(x1)
	else:
		layer_1 = dense_layer_1

	local_info = layer_1
	if n_local_conv>0:
		for i in range(n_local_conv):
			local_info = TimeDistributed(Conv1D(n_filter2, local_conv_size, padding='same', activation=activation1, strides=1), name = 'conv%d_gumbel_local_%d'%(i+2,number))(local_info)

	# global info, shape: (batch_size, feature_dim)
	if concat>0:
		x1 = TimeDistributed(GlobalMaxPooling1D(),name = 'new_global_max_pooling1d_%d'%(number))(layer_1)
		if dim1>0:
			global_info = TimeDistributed(Dense(dim1,activation=activation1), name = 'new_dense_%d'%(number))(x1)
		else:
			global_info = x1

		# concatenated feature, shape: (batch_size, n_steps, dim1+dim2)
		x2 = TimeDistributed(Concatenate_1(),name='concatenate_local_1')([global_info,local_info])
	else:
		x2 = local_info
		
	# x2 = Dropout(0.2, name = 'new_dropout_2')(x2)
	# current configuration: dense1 + conv1 + conv2
	if dim2>0:
		x2 = TimeDistributed(Conv1D(dim2, 1, padding='same', activation=activation1, strides=1), name = 'conv%d_gumbel_local_%d'%(n_local_conv+2,number))(x2)
	if ('batchnorm1' in config) and config['batchnorm1']==1:
		x2 = TimeDistributed(BatchNormalization(),name ='conv%d_gumbel_local_bn%d'%(n_local_conv+2,number))(x2)

	x2_local = construct_gumbel_selector1_1(x2,config,number=number+1,type_id=type_id)

	return x2_local

# construct gumbel selector 1: feature vector
def construct_basic1(input1,config,number=1,type_id=1):

	feature_dim, output_dim = config['feature_dim'], config['output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation1 = config['activation']
	# activation1 = 'relu'
	feature_dim_vec1 = config['feature_dim_vec_basic']
	if 'local_conv_size' in config:
		local_conv_size = config['local_conv_size']
	else:
		local_conv_size = 3

	# # input1 = Input(shape = (n_steps,feature_dim))
	# units_1 = config['units1']
	# if units_1>0:
	# 	# encode the input, shape: (batch_size,n_steps,units_1)
	# 	dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)	
	# else:
	# 	dense_layer_1 = input1

	# default: n_filter1:50, dim1:25, n_filter2: 50, dim2: 25, n_local_conv: 0, concat: 0
	n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0:6]
	print(n_filter1, n_filter2, n_local_conv, dim1, dim2, concat)
	if n_filter1>0:
		x1 = Conv1D(n_filter1, 1, padding='same', strides=1, name = 'conv1_basic1_%d'%(number))(input1)
		x1 = BatchNormalization(name='batchnorm_1_%d'%(number))(x1)
		layer_1 = Activation(activation1,name='conv1_basic_%d'%(number))(x1)
	else:
		layer_1 = input1

	local_info = layer_1
	if n_local_conv>0:
		for i in range(n_local_conv):
			local_info = Conv1D(n_filter2, local_conv_size, padding='same', activation=activation1, strides=1, name = 'conv%d_basic_%d'%(i+2,number))(local_info)

	# global info, shape: (batch_size, feature_dim)
	if concat>0:
		x1 = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_%d'%(number))(layer_1)
		if dim1>0:
			global_info = Dense(dim1, name = 'new_dense_%d'%(number), activation=activation1)(x1)
		else:
			global_info = x1

		# concatenated feature, shape: (batch_size, n_steps, dim1+dim2)
		x2 = Concatenate_1()([global_info,local_info])
	else:
		x2 = local_info

	# x2 = Dropout(0.2, name = 'new_dropout_2')(x2)
	# current configuration: dense1 + conv1 + conv2
	if dim2>0:
		x2 = Conv1D(dim2, 1, padding='same', activation=activation1, strides=1, name = 'conv%d_basic_%d'%(n_local_conv+2,number))(x2)
	if ('batchnorm1' in config) and config['batchnorm1']==1:
		x2 = TimeDistributed(BatchNormalization(),name ='conv%d_basic_bn%d'%(n_local_conv+2,number))(x2)

	return x2

# construct gumbel selector 1
def get_modelpre_basic2(input1,config,number=1,type_id=1):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	# lr = config['lr']
	activation1 = config['activation']
	# activation1 = 'relu'
	feature_dim_vec1 = config['feature_dim_vec']

	# input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		# encode the input, shape: (batch_size,n_steps,units_1)
		dense_layer_1 = Dense(units_1,name='dense_0')(input1)	
	else:
		dense_layer_1 = input1

	# default: n_filter1:50, dim1:25, n_filter2: 50, dim2: 25, n_local_conv: 0, concat: 0
	n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0], feature_dim_vec1[1], feature_dim_vec1[2], feature_dim_vec1[3], feature_dim_vec1[4], feature_dim_vec1[5]
	if n_filter1>0:
		# layer_1 = Conv1D(n_filter1, 1, padding='same', activation=activation1, strides=1, name = 'conv1_gumbel_%d'%(number))(dense_layer_1)
		layer_1 = Dense(n_filter1, activation=activation1, name = 'conv1_gumbel_%d'%(number))(dense_layer_1)
	else:
		layer_1 = dense_layer_1

	# local info
	if n_local_conv>0:
		layer_2 = Dense(n_filter2, activation=activation1, name = 'conv2_gumbel_%d'%(number))(layer_1)
		if n_local_conv>1:
			local_info = Dense(n_filter2, activation=activation1, name = 'conv3_gumbel_%d'%(number))(layer_2)
		else:
			local_info = layer_2
	else:
		local_info = layer_1

	x2 = local_info

	# x2 = Dropout(0.2, name = 'new_dropout_2')(x2)
	# current configuration: dense1 + conv1 + conv2
	if dim2>0:
		x2 = Dense(dim2, activation=activation1, name = 'conv4_%d'%(number))(x2)

	return x2

def get_model2a1_basic1(input1,config):

	feature_dim, output_dim, n_steps = config['feature_dim'], config['output_dim'], config['context_size']
	activation = config['activation']
	activation2 = config['activation2']
	activation_self = config['activation_self']
	if 'feature_dim3' in config:
		feature_dim3 = config['feature_dim3']
	else:
		feature_dim3 = []
	regularizer2 = config['regularizer2']
	# regularizer2_2 = config['regularizer2_2']
	if 'activation_basic' in config:
		activation_basic = config['activation_basic']
	else:
		activation_basic = 'sigmoid'

	# method 21: attention1:1, method 22: attention1:0
	if config['attention1']==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, 
												attention_activation=activation_self,
												name='attention1')(input1)
	else:
		layer_1 = input1

	# biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
	# 								units=output_dim,
	# 								return_sequences = True,
	# 								recurrent_dropout = 0.1),name='bilstm1')

	if 'regularizer2_2' in config:
		regularizer2_2 = config['regularizer2_2']
	else:
		regularizer2_2 = 1e-05
	biLSTM_layer1 = Bidirectional(LSTM(
									units=output_dim,
									kernel_regularizer=regularizers.l2(regularizer2_2),
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	# biLSTM_layer1 = Bidirectional(LSTM(
	# 								activation='tanh',
	# 								units=output_dim,
	# 								kernel_regularizer=regularizers.l2(regularizer2_2),
	# 								return_sequences = True,
	# 								recurrent_dropout = 0.1),name='bilstm1')
	# biLSTM_layer1 = Bidirectional(LSTM(
	# 								activation='linear',
	# 								units=output_dim,
	# 								kernel_regularizer=regularizers.l2(regularizer2_2),
	# 								return_sequences = True,
	# 								recurrent_dropout = 0.1),name='bilstm1')

	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation2!='':
		x1 = Activation(activation2,name='activation2_2')(x1)
	# x1 = Activation(activation2,name='activation2_2')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])

	if config['attention2']==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, 
											attention_activation=activation_self,
											name='attention2')(x1)

	# if config['concatenate_2']==1:
	# 	global_info = GlobalMaxPooling1D(name='global_pooling_1')(x1)
	# 	x1 =Concatenate_1(name='concatenate_1')([x1,global_info])

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])

	cnt1 = 0
	for t_feature_dim3 in feature_dim3:
		cnt1 += 1
		x1 = TimeDistributed(Dense(t_feature_dim3,
								kernel_regularizer=regularizers.l2(regularizer2)),
								name = 'conv1_3_%d'%(cnt1))(x1)
		x1 = TimeDistributed(BatchNormalization(),name='bnorm1_3_%d'%(cnt1))(x1)
		x1 = TimeDistributed(Activation('relu'),name='activation1_3_%d'%(cnt1))(x1)

	if 'output_dim_1' in config:
		output_dim1 = config['output_dim_1']
	else:
		output_dim1 = 1
	output = Dense(output_dim1,name='dense2')(x1)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation(activation_basic,name='activation2')(output)

	return output

def get_model2a1_basic1_2(input1,config):

	feature_dim, output_dim, n_steps = config['feature_dim'], config['output_dim'], config['context_size']
	activation = config['activation']
	activation2 = config['activation2']
	activation_self = config['activation_self']
	if 'feature_dim3' in config:
		feature_dim3 = config['feature_dim3']
	else:
		feature_dim3 = []
	regularizer2 = config['regularizer2']
	# regularizer2_2 = config['regularizer2_2']
	if 'activation_basic' in config:
		activation_basic = config['activation_basic']
	else:
		activation_basic = 'sigmoid'

	# method 21: attention1:1, method 22: attention1:0
	if config['attention1']==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, 
												attention_activation=activation_self,
												name='attention1')(input1)
	else:
		layer_1 = input1

	# biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
	# 								units=output_dim,
	# 								return_sequences = True,
	# 								recurrent_dropout = 0.1),name='bilstm1')

	if 'regularizer2_2' in config:
		regularizer2_2 = config['regularizer2_2']
	else:
		regularizer2_2 = 1e-05
	# biLSTM_layer1 = Bidirectional(LSTM(
	# 								activation='tanh',
	# 								units=output_dim,
	# 								kernel_regularizer=regularizers.l2(regularizer2_2),
	# 								return_sequences = True,
	# 								recurrent_dropout = 0.1),name='bilstm1')
	biLSTM_layer1 = Bidirectional(LSTM(
									activation='tanh',
									units=output_dim,
									kernel_regularizer=regularizers.l2(regularizer2_2),
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')

	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation2!='':
		x1 = Activation(activation2,name='activation2_2')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])

	if config['attention2']==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, 
											attention_activation=activation_self,
											name='attention2')(x1)

	# if config['concatenate_2']==1:
	# 	global_info = GlobalMaxPooling1D(name='global_pooling_1')(x1)
	# 	x1 =Concatenate_1(name='concatenate_1')([x1,global_info])

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])

	cnt1 = 0
	for t_feature_dim3 in feature_dim3:
		cnt1 += 1
		x1 = TimeDistributed(Dense(t_feature_dim3,
								kernel_regularizer=regularizers.l2(regularizer2)),
								name = 'conv1_3_%d'%(cnt1))(x1)
		x1 = TimeDistributed(BatchNormalization(),name='bnorm1_3_%d'%(cnt1))(x1)
		x1 = TimeDistributed(Activation('relu'),name='activation1_3_%d'%(cnt1))(x1)

	if 'output_dim_1' in config:
		output_dim1 = config['output_dim_1']
	else:
		output_dim1 = 1
	output = Dense(output_dim1,name='dense2')(x1)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation(activation_basic,name='activation2')(output)

	return output

def get_model2a1_basic2(input1,config):

	feature_dim, output_dim, n_steps = config['feature_dim'], config['output_dim'], config['context_size']
	activation = config['activation']
	activation_self = config['activation_self']
	if 'activation_basic' in config:
		activation_basic = config['activation_basic']
	else:
		activation_basic = 'sigmoid'

	# method 21: attention1:1, method 22: attention1:0
	if 'n_layer_basic' in config:
		n_layer = config['n_layer_basic']
	else:
		n_layer = 2
		
	x1 = input1
	for i in range(0,n_layer):
		x1 = Dense(output_dim,name='dense%d'%(i+2))(x1)
		x1 = BatchNormalization(name='batchnorm%d'%(i+2))(x1)
		x1 = Activation(activation_self,name='activation%d'%(i+2))(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense_basic2')(x1)
	output = BatchNormalization(name='batchnorm_basic2')(output)
	output = Activation(activation_basic,name='activation_basic2')(output)

	return output

# get_model2a1_basic1_2 from utility_1_5
def get_model2a1_basic1_2_ori(input1,config):

	feature_dim, output_dim, n_steps = config['feature_dim'], config['output_dim'], config['context_size']
	activation = config['activation']
	activation2 = config['activation2']
	activation_self = config['activation_self']
	if 'activation_basic' in config:
		activation_basic = config['activation_basic']
	else:
		activation_basic = 'sigmoid'

	# method 21: attention1:1, method 22: attention1:0
	if config['attention1']==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, 
												attention_activation=activation_self,
												name='attention1')(input1)
	else:
		layer_1 = input1

	if ('layer_norm' in config) and (config['layer_norm']>0):
		if (config['layer_norm']==1):
			activation_2 = "linear"
		else:
			if activation2!='':
				activation_2 = activation2
			else:
				activation_2 = "tanh"

		biLSTM_layer1 = Bidirectional(LSTM(
								input_shape=(n_steps, feature_dim),
								units=output_dim,
								activation=activation_2,
								return_sequences = True,
								recurrent_dropout = 0.1),name='bilstm1')
		x1 = biLSTM_layer1(layer_1)
		# x1 = BatchNormalization()(x1)
		x1 = LayerNormalization(name='layernorm1')(x1)

		if (config['layer_norm']==1) and (activation2!=''):
			x1 = Activation(activation2,name='activation1')(x1)

		print('layer_norm',config['layer_norm'],activation2,activation_2)

	else:
		print('layer_norm',config['layer_norm'])
		biLSTM_layer1 = Bidirectional(LSTM(
									input_shape=(n_steps, feature_dim),
									units=output_dim,
									activation=activation2,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
		x1 = biLSTM_layer1(layer_1)
		
	print("get_model2a1_basic1_2",activation2)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if config['attention2']==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, 
											attention_activation=activation_self,
											name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(x1)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation(activation_basic,name='activation2')(output)

	return output


# multiple convolution layers, self-attention
# method 31, 35
def get_model2a1_attention1_1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	if not('loss_function' in config):
		loss_function = 'mean_squared_error'
	else:
		loss_function = config['loss_function']

	input1 = Input(shape = (n_steps,feature_dim))

	typeid = 0
	number = 1
	layer_1 = construct_gumbel_selector1(input1,config,number,typeid)

	output = get_model2a1_basic1(layer_1,config)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	# adam = Adam(lr = lr, clipnorm=1.0, clipvalue=1.0)
	adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = loss_function)

	model.summary()

	return model

# method 56: single network for predicting signals
# with multiple fully connected layers
def get_model2a1_attention_1_2_2_single(config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	
	if not('loss_function' in config):
		loss_function = 'mean_squared_error'
	else:
		loss_function = config['loss_function']

	input1 = Input(shape = (feature_dim,))

	units1 = config['units1']
	config['units1'] = config['units2']
	dense_layer_output1 = get_modelpre_basic2(input1,config)
	output = get_model2a1_basic2(dense_layer_output1,config)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	model.compile(adam,loss = loss_function)

	model.summary()

	return model

# convolution
def get_model2a1_convolution_pre_1(input1,config):

	# input1 = Input(shape = (input_shape,4))
	# conv_1 = [[[128,20,1e-04,1,1,'relu',10,10,0.2],[64,20,1e-04,1,1,'relu',10,10,0.2]],
	#			[[32,0.1,0.5]],[1],[[50,1,'relu',0]]]
	conv_list_ori = config['conv_1'] # 0: conv_layers: 1: bilstm 2: processing output of lstm 3: dense_layers 
	activation_self = config['activation_self']

	x1 = input1
	conv_list1 = conv_list_ori[0]
	cnt1 = 0
	for conv_1 in conv_list1:
		cnt1 = cnt1+1
		n_filters, kernel_size1, regularizer2, stride, dilation_rate1, bnorm, boundary, activation, pool_length1, stride1, drop_out_rate = conv_1
		x1 = Conv1D(filters = n_filters, kernel_size = kernel_size1, strides = stride, activation = "linear",
					padding = boundary,
					kernel_regularizer=regularizers.l2(regularizer2),
					dilation_rate = dilation_rate1,
					name = 'conv1_%d'%(cnt1))(x1)
		# x1 = Conv1D(filters = n_filters, kernel_size = kernel_size1, activation = "linear",
		# 			kernel_regularizer=regularizers.l2(regularizer2),
		# 			activity_regularizer=regularizers.l1(regularizer1))(x1)
		if bnorm>0:
			x1 = BatchNormalization(name='bnorm1_%d'%(cnt1))(x1)
		print(n_filters,kernel_size1,activation,pool_length1,drop_out_rate)
		x1 = Activation(activation,name='activation1_%d'%(cnt1))(x1)
		if pool_length1>1:
			x1 = MaxPooling1D(pool_size = pool_length1, strides = stride1, name='pooling_%d'%(cnt1))(x1)
		if drop_out_rate>0:
			x1 = Dropout(drop_out_rate,name='dropout1_%d'%(cnt1))(x1)

	# if config['attention1']==1:
	# 	x1, attention1 = SeqSelfAttention(return_attention=True, 
	# 										attention_activation=activation_self,
	# 										name='attention1')(x1)

	conv_list2 = conv_list_ori[1]
	cnt1 = 0
	for conv_1 in conv_list2:
		cnt1 = cnt1+1
		output_dim, recurrent_dropout_rate, drop_out_rate = conv_1[0:3]
		biLSTM_layer1 = Bidirectional(LSTM(units=output_dim,
									return_sequences = True,
									recurrent_dropout = recurrent_dropout_rate),name='bilstm%d'%(cnt1))
		x1 = biLSTM_layer1(x1)
		x1 = LayerNormalization(name='layernorm2_%d'%(cnt1))(x1)
		x1 = Dropout(drop_out_rate,name='dropout2_%d'%(cnt1))(x1)

	connection = conv_list_ori[2]
	cnt1 = 0
	for conv_1 in connection[0]:
		cnt1 = cnt1+1
		fc1_output_dim, bnorm, activation, drop_out_rate = conv_1
		x1 = Dense(fc1_output_dim,name='dense3_%d_1'%(cnt1))(x1)
		if bnorm>0:
			x1 = BatchNormalization(name='bnorm3_%d_1'%(cnt1))(x1)
		x1 = Activation(activation,name='activation3_%d_1'%(cnt1))(x1)
		if drop_out_rate>0:
			x1 = Dropout(drop_out_rate,name='dropout3_%d_1'%(cnt1))(x1)

	if config['attention2']==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, 
											attention_activation=activation_self,
											name='attention2')(x1)

	flag1 = connection[-1][0]
	if flag1==1:
		x1 = Flatten(name='flatten1')(x1)
	else:
		x1 = GlobalMaxPooling1D(name ='global_max_pooling1d_1')(x1)

	conv_list3 = conv_list_ori[3]
	cnt1 = 0
	for conv_1 in conv_list3:
		cnt1 = cnt1+1
		fc1_output_dim, bnorm, activation, drop_out_rate = conv_1
		x1 = Dense(fc1_output_dim,name='dense3_%d'%(cnt1))(x1)
		if bnorm>0:
			x1 = BatchNormalization(name='bnorm3_%d'%(cnt1))(x1)
		x1 = Activation(activation,name='activation3_%d'%(cnt1))(x1)
		if drop_out_rate>0:
			x1 = Dropout(drop_out_rate,name='dropout3_%d'%(cnt1))(x1)

	dense_layer_output = x1

	return dense_layer_output

# convolution layer + selector + LSTM + pooling
# 2D feature (n_step_local,feature_dim) to 1D
# from get_model2a1_basic5_convolution
def get_model2a1_convolution_pre(input_local,select_config):

	return_sequences_flag, sample_local, pooling_local = select_config['local_vec_1']
	# print(feature_dim1, feature_dim2, return_sequences_flag)

	conv_list_ori = select_config['local_conv_list_ori'] # 0: conv_layers: 1: bilstm 2: processing output of lstm 3: dense_layers 
	cnt1 = 0
	boundary_vec = ['same','valid']
	x1 = input_local
	conv_list1 = conv_list_ori[0]
	for conv_1 in conv_list1:
		cnt1 = cnt1+1
		if len(conv_1)==0:
			continue
		n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate = conv_1
		x1 = Conv1D(filters = n_filters, kernel_size = kernel_size1, strides = stride, padding=boundary_vec[boundary], activation = "linear",
					kernel_regularizer=regularizers.l2(regularizer2),
					dilation_rate = dilation_rate1,
					name = 'conv1_pre_%d'%(cnt1))(x1)
		# x1 = Conv1D(filters = n_filters, kernel_size = kernel_size1, activation = "linear",
		# 			kernel_regularizer=regularizers.l2(regularizer2),
		# 			activity_regularizer=regularizers.l1(regularizer1))(x1)
		x1 = BatchNormalization(name='bnorm1_pre_%d'%(cnt1))(x1)
		print(n_filters,kernel_size1,activation,pool_length1,drop_out_rate)
		x1 = Activation(activation,name='activation1_pre_%d'%(cnt1))(x1)

		if pool_length1>1:
			x1 = MaxPooling1D(pool_size = pool_length1, strides = stride1, name='pooling_pre_%d'%(cnt1))(x1)
		if drop_out_rate>0:
			x1 = Dropout(drop_out_rate,name='dropout1_pre_%d'%(cnt1))(x1)

	layer_1 = x1
	if sample_local>=1:
		logits_T_local = construct_gumbel_selector1_sequential(layer_1,select_config,number=1,type_id=2)
		
		if sample_local==1:
			tau, k = select_config['tau'], select_config['n_select']
			typeid_sample, activation3 = select_config['typeid_sample'], select_config['activation3']
			print('sample_attention',tau,k,activation3,typeid_sample)
			if activation3=='linear':
				typeid_sample = 1
			elif activation3=='tanh':
				typeid_sample = 5
			else:
				pass
			attention1 = Sample_Concrete1(tau,k,n_step_local,typeid_sample,name='Sample_Concrete1_local')(logits_T_local) # output shape: (batch_size, n_step_local, 1)
		else:
			attention1 = logits_T_local
	
		layer_1 = Multiply()([layer_1, attention1])

	conv_list2 = conv_list_ori[1]
	cnt1 = 0
	layer_1 = x1
	for conv_1 in conv_list2:
		cnt1 = cnt1+1
		if len(conv_1)==0:
			continue
		output_dim, activation2, regularizer2, recurrent_dropout_rate, drop_out_rate, layer_norm = conv_1[0:6]
		biLSTM_layer1 = Bidirectional(LSTM(units=output_dim,
									activation = activation2,
									return_sequences = True,
									kernel_regularizer = keras.regularizers.l2(regularizer2),
									recurrent_dropout = recurrent_dropout_rate),name='bilstm2_%d'%(cnt1))
		x1 = biLSTM_layer1(x1)
		if layer_norm>0:
			x1 = LayerNormalization(name='layernorm2_%d'%(cnt1))(x1)
		x1 = Dropout(drop_out_rate,name='dropout2_%d'%(cnt1))(x1)

	if return_sequences_flag==True:

		# x1 = BatchNormalization()(x1)
		# x1 = TimeDistributed(LayerNormalization(),name='layernorm_local_1')(x1)
		if select_config['concatenate_1']==1:
			# x1 = TimeDistributed(Concatenate(axis=-1),name='concatenate_local_1')([x1,layer_1])
			x1 = Concatenate(axis=-1,name='concatenate_local_1')([x1,layer_1])

		connection = conv_list_ori[2]
		cnt1 = 0
		for conv_1 in connection:
			cnt1 = cnt1+1
			if len(conv_1)==0:
				continue
			fc1_output_dim, bnorm, activation, drop_out_rate = conv_1
			x1 = Dense(fc1_output_dim,name='dense3_%d_1'%(cnt1))(x1)
			if bnorm>0:
				x1 = BatchNormalization(name='bnorm3_%d_1'%(cnt1))(x1)
			x1 = Activation(activation,name='activation3_%d_1'%(cnt1))(x1)
			if drop_out_rate>0:
				x1 = Dropout(drop_out_rate,name='dropout3_%d_1'%(cnt1))(x1)

		if select_config['attention2_local']==1:
			activation_self = select_config['activation_self']
			x1, attention2 = SeqSelfAttention(return_attention=True, 
												attention_activation=activation_self,
												name='attention_local_1')(x1)

		if pooling_local==1:
			x1 = GlobalMaxPooling1D(name='global_pooling_local_1')(x1)
		else:
			x1 = Flatten(name='Flatten_local_1')(x1)

	conv_list3 = conv_list_ori[3]
	cnt1 = 0
	for conv_1 in conv_list3:
		cnt1 = cnt1+1
		if len(conv_1)==0:
			continue
		fc1_output_dim, bnorm, activation, drop_out_rate = conv_1
		x1 = Dense(fc1_output_dim,
					kernel_regularizer=regularizers.l2(1e-05),
					name='dense3_%d'%(cnt1))(x1)
		if bnorm>0:
			x1 = BatchNormalization(name='bnorm3_%d'%(cnt1))(x1)
		x1 = Activation(activation,name='activation3_%d'%(cnt1))(x1)
		if drop_out_rate>0:
			x1 = Dropout(drop_out_rate,name='dropout3_%d'%(cnt1))(x1)

	return x1

# convolution (original function)
def get_model2a1_convolution(config):

	size1 = config['n_step_local_ori']
	# n_steps = config['context_size']
	learning_rate = config['lr']
	# activation = config['activation']
	# activation_self = config['activation_self']
	# activation3 = config['activation3']
	if not('loss_function' in config):
		loss_function = 'mean_squared_error'
	else:
		loss_function = config['loss_function']

	input1 = Input(shape = (size1,4))
	dense_layer_output = get_model2a1_convolution_pre(input1,config)

	conv_2 = config['conv_2'] # conv_2: [1,1,'sigmoid']
	# drop_out_rate, n_dim, bnorm, activation = conv_2
	# x1 = Dropout(drop_out_rate)(x1)
	n_dim, bnorm, activation = conv_2[0:3]
	# output = Dense(1,activation= 'sigmoid')(dense1)
	# output = Dense(1,activation= 'sigmoid')(dense_layer_output)
	output = Dense(n_dim)(dense_layer_output)
	if bnorm>0:
		output = BatchNormalization()(output)
	output = Activation(activation)(output)
	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	model.compile(adam,loss = loss_function)
	# model.compile(adam,loss = 'kullback_leibler_divergence')

	model.summary()
	
	return model

# network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling and multiple convolution layers
def get_model2a1_attention_1_2_2_sample(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	if not('loss_function' in config):
		loss_function = 'mean_squared_error'
	else:
		loss_function = config['loss_function']

	input1 = Input(shape = (n_steps,feature_dim))

	number = 1
	typeid = 2
	logits_T = construct_gumbel_selector1(input1,config,number,typeid)

	# k = 10
	if not('sample_attention' in config) or config['sample_attention']==1:
		tau = 0.5
		k = 5
		print('sample_attention',tau,k,typeid,activation3,typeid_sample)
		if 'tau' in config:
			tau = config['tau']
		if 'n_select' in config:
			k = config['n_select']
		if typeid<2:
			attention1 = Sample_Concrete(tau,k,n_steps)(logits_T)
		else:
			if activation3=='linear':
				typeid_sample = 1
			elif activation3=='tanh':
				typeid_sample = 5
			elif activation3=='sigmoid':
				typeid_sample = 3
			else:
				pass
			attention1 = Sample_Concrete1(tau,k,n_steps,typeid_sample)(logits_T) # output shape: (batch_size, n_steps, 1)
	else:
		attention1 = logits_T

	# encode the input 2
	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	if config['select2']==1:
		units1 = config['units1']
		config['units1'] = 0
		typeid = 0
		number = 2
		dense_layer_output1 = construct_gumbel_selector1(dense_layer_output1,config,number,typeid)
		config['units1'] = units1

	layer_1 = Multiply()([dense_layer_output1, attention1])

	output = get_model2a1_basic1(layer_1,config)
	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	model.compile(adam,loss = loss_function)

	model.summary()

	return model

# network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling and multiple convolution layers
def get_model2a1_attention_1_2_2_sample_1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	if not('loss_function' in config):
		loss_function = 'mean_squared_error'
	else:
		loss_function = config['loss_function']

	input1 = Input(shape = (n_steps,feature_dim))

	number = 1
	typeid = 2
	logits_T = construct_gumbel_selector1(input1,config,number,typeid)

	# k = 10
	if not('sample_attention' in config) or config['sample_attention']==1:
		tau = 0.5
		k = 5
		print('sample_attention',tau,k,typeid,activation3,typeid_sample)
		if 'tau' in config:
			tau = config['tau']
		if 'n_select' in config:
			k = config['n_select']
		if typeid<2:
			attention1 = Sample_Concrete(tau,k,n_steps)(logits_T)
		else:
			if activation3=='linear':
				typeid_sample = 1
			elif activation3=='tanh':
				typeid_sample = 5
			elif activation3=='sigmoid':
				typeid_sample = 3
			else:
				pass
			attention1 = Sample_Concrete1(tau,k,n_steps,typeid_sample)(logits_T) # output shape: (batch_size, n_steps, 1)
	else:
		attention1 = logits_T

	# encode the input 2
	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	if config['select2']==1:
		units1 = config['units1']
		config['units1'] = 0
		typeid = 0
		number = 2
		dense_layer_output1 = construct_gumbel_selector1(dense_layer_output1,config,number,typeid)
		config['units1'] = units1

	layer_1 = Multiply()([dense_layer_output1, attention1])

	output = get_model2a1_basic1(layer_1,config)

	model = Model(input = input1, output = output)
	adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	model.compile(adam,loss = loss_function)

	model.summary()

	return model

def find_optimizer(config):

	init_lr, decay_rate1 = 0.005, 0.96
	if 'init_lr' in config:
		init_lr = config['init_lr']
	if 'decay_rate1' in config:
		decay_rate1 = config['decay_rate1']
	
	lr_schedule = keras.optimizers.schedules.ExponentialDecay(
		initial_learning_rate=init_lr,
		decay_steps=50,
		decay_rate=decay_rate1,
		staircase=True)

	lr = config['lr']
	lr_id = 1-config['lr_schedule']
	vec1 = [lr,lr_schedule]

	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	print(config['optimizer'])
	if config['optimizer']=='SGD':
		optimizer = keras.optimizers.SGD(learning_rate=vec1[lr_id], nesterov=True, clipnorm=CLIPNORM1)
	elif config['optimizer']=='RMSprop':
		optimizer = keras.optimizers.RMSprop(learning_rate=vec1[lr_id], clipnorm=CLIPNORM1)
	elif config['optimizer']=='Adadelta':
		optimizer = keras.optimizers.Adadelta(learning_rate=lr, clipnorm=CLIPNORM1)
	elif config['optimizer']=='Adagrad':
		optimizer = keras.optimizers.Adagrad(learning_rate=lr, clipnorm=CLIPNORM1)
	elif config['optimizer']=='Nadam':
		optimizer = keras.optimizers.Nadam(learning_rate=lr, clipnorm=CLIPNORM1)
	else:
		optimizer = keras.optimizers.Adam(learning_rate=vec1[lr_id], clipnorm=CLIPNORM1)

	return optimizer

# convolution layer + selector + LSTM + pooling
# 2D feature (n_step_local,feature_dim) to 1D
def get_model2a1_basic5_convolution(input_local,select_config):

	# n_step_local,feature_dim = select_config['input_shape_1']
	n_step_local = select_config['n_step_local']
	feature_dim1, feature_dim2, feature_dim3, return_sequences_flag, sample_local, pooling_local = select_config['local_vec_1']
	# print(feature_dim1, feature_dim2, return_sequences_flag)

	# conv_list1 = config['local_vec_1'] # 0: conv_layers: 1: bilstm 2: processing output of lstm 3: dense_layers 
	conv_list1 = select_config['local_conv_list1'] # 0: conv_layers: 1: bilstm 2: processing output of lstm 3: dense_layers 

	# input_local = Input(shape=(n_step_local,feature_dim))
	# lstm_1 = Bidirectional(LSTM(feature_dim1, name = 'lstm_1'), 
	# 		name = 'bidirectional')(embedded_sequences)

	# layer_1 = TimeDistributed(Conv1D(feature_dim1,1,padding='same',activation=None,strides=1),name='conv_local_1')(input_local)
	# layer_1 = TimeDistributed(BatchNormalization(),name='batchnorm_local_1')(layer_1)
	# layer_1 = TimeDistributed(Activation('relu'),name='activation_local_1')(layer_1)

	cnt1 = 0
	boundary_vec = ['same','valid']
	x1 = input_local
	for conv_1 in conv_list1:
		cnt1 = cnt1+1
		n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate = conv_1
		x1 = TimeDistributed(Conv1D(filters = n_filters, kernel_size = kernel_size1, strides = stride, padding=boundary_vec[boundary], activation = "linear",
					kernel_regularizer=regularizers.l2(regularizer2),
					dilation_rate = dilation_rate1),
					name = 'conv1_pre_%d'%(cnt1))(x1)
		# x1 = Conv1D(filters = n_filters, kernel_size = kernel_size1, activation = "linear",
		# 			kernel_regularizer=regularizers.l2(regularizer2),
		# 			activity_regularizer=regularizers.l1(regularizer1))(x1)
		x1 = TimeDistributed(BatchNormalization(),name='bnorm1_pre_%d'%(cnt1))(x1)
		print(n_filters,kernel_size1,activation,pool_length1,drop_out_rate)
		x1 = TimeDistributed(Activation(activation),name='activation1_pre_%d'%(cnt1))(x1)

		if pool_length1>1:
			x1 = TimeDistributed(MaxPooling1D(pool_size = pool_length1, strides = stride1), name='pooling_pre_%d'%(cnt1))(x1)
		if drop_out_rate>0:
			x1 = TimeDistributed(Dropout(drop_out_rate),name='dropout1_pre_%d'%(cnt1))(x1)

	layer_1 = x1
	if sample_local>=1:
		logits_T_local = construct_gumbel_selector1_sequential(layer_1,select_config,number=1,type_id=2)
		
		if sample_local==1:
			tau, k = select_config['tau'], select_config['n_select']
			typeid_sample, activation3 = select_config['typeid_sample'], select_config['activation3']
			print('sample_attention',tau,k,activation3,typeid_sample)
			if activation3=='linear':
				typeid_sample = 1
			elif activation3=='tanh':
				typeid_sample = 5
			else:
				pass
			attention1 = TimeDistributed(Sample_Concrete1(tau,k,n_step_local,typeid_sample),name='Sample_Concrete1_local')(logits_T_local) # output shape: (batch_size, n_steps, n_step_local, 1)
		else:
			attention1 = logits_T_local
	
		layer_1 = Multiply()([layer_1, attention1])
		
	# biLSTM_layer_1 = Bidirectional(LSTM(input_shape=(n_step_local, n_filters), 
	# 								units=feature_dim1,
	# 								return_sequences = return_sequences_flag,
	# 								kernel_regularizer = keras.regularizers.l2(1e-5),
	# 								dropout=0.1,
	# 								recurrent_dropout = 0.1),name='bilstm_local_1_1')

	regularizer2 = select_config['regularizer2_2']
	biLSTM_layer_1 = Bidirectional(LSTM(units=feature_dim1,
									return_sequences = return_sequences_flag,
									kernel_regularizer = keras.regularizers.l2(regularizer2),
									dropout=0.1,
									recurrent_dropout = 0.1),name='bilstm_local_1_1')

	x1 = TimeDistributed(biLSTM_layer_1,name='bilstm_local_1')(layer_1)

	if return_sequences_flag==True:

		# x1 = BatchNormalization()(x1)
		# x1 = TimeDistributed(LayerNormalization(),name='layernorm_local_1')(x1)
		if select_config['concatenate_1']==1:
			# x1 = TimeDistributed(Concatenate(axis=-1),name='concatenate_local_1')([x1,layer_1])
			x1 = Concatenate(axis=-1,name='concatenate_local_1')([x1,layer_1])

		if feature_dim2>0:
			cnt1 += 1
			x1 = TimeDistributed(Dense(feature_dim2,
										kernel_regularizer=regularizers.l2(1e-05)),
										name = 'conv1_pre_%d'%(cnt1))(x1)
			x1 = TimeDistributed(BatchNormalization(),name='bnorm1_pre_%d'%(cnt1))(x1)
			x1 = TimeDistributed(Activation('relu'),name='activation1_pre_%d'%(cnt1))(x1)

		if select_config['attention2_local']==1:
			x1, attention2 = TimeDistributed(SeqSelfAttention(return_attention=True, attention_activation=activation_self),name='attention_local_1')(x1)

		if pooling_local==1:
			x1 = TimeDistributed(GlobalMaxPooling1D(),name='global_pooling_local_1')(x1)
		else:
			x1 = TimeDistributed(Flatten(),name='Flatten_local_1')(x1)

	for t_feature_dim3 in feature_dim3:
		cnt1 += 1
		x1 = TimeDistributed(Dense(t_feature_dim3,
								kernel_regularizer=regularizers.l2(1e-05)),
								name = 'conv1_pre_%d'%(cnt1))(x1)
		x1 = TimeDistributed(BatchNormalization(),name='bnorm1_pre_%d'%(cnt1))(x1)
		x1 = TimeDistributed(Activation('relu'),name='activation1_pre_%d'%(cnt1))(x1)

	return x1

# convolution layer + pooling + dropout + dilated convolution
# 2D feature (n_step_local,feature_dim) to 1D
def get_model2a1_basic5_convolution1(input_local,select_config):

	# n_step_local,feature_dim = select_config['input_shape_1']
	# n_step_local = select_config['n_step_local']
	# feature_dim1, feature_dim2, feature_dim3, return_sequences_flag, sample_local, pooling_local = select_config['local_vec_1']
	# print(feature_dim1, feature_dim2, return_sequences_flag)

	# conv_list1 = config['local_vec_1'] # 0: conv_layers; 1: bilstm;  2: processing output of lstm; 3: dense_layers 
	conv_list1 = select_config['local_conv_list1'] # 0: conv_layers; 1: bilstm 2; processing output of lstm; 3: dense_layers 
	conv_list2 = select_config['local_conv_list2'] # 0: dilated convolution layers 

	# input_local = Input(shape=(n_step_local,feature_dim))
	# lstm_1 = Bidirectional(LSTM(feature_dim1, name = 'lstm_1'), 
	# 		name = 'bidirectional')(embedded_sequences)

	# layer_1 = TimeDistributed(Conv1D(feature_dim1,1,padding='same',activation=None,strides=1),name='conv_local_1')(input_local)
	# layer_1 = TimeDistributed(BatchNormalization(),name='batchnorm_local_1')(layer_1)
	# layer_1 = TimeDistributed(Activation('relu'),name='activation_local_1')(layer_1)

	cnt1 = 0
	boundary_vec = ['same','valid']
	x1 = input_local
	for conv_1 in conv_list1:
		cnt1 = cnt1+1
		n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate = conv_1
		x1 = TimeDistributed(Conv1D(filters = n_filters, kernel_size = kernel_size1, strides = stride, padding=boundary_vec[boundary], 
					activation = "linear",
					kernel_regularizer=regularizers.l2(regularizer2),
					dilation_rate = dilation_rate1),
					name = 'conv1_pre_%d'%(cnt1))(x1)
		# x1 = Conv1D(filters = n_filters, kernel_size = kernel_size1, activation = "linear",
		# 			kernel_regularizer=regularizers.l2(regularizer2),
		# 			activity_regularizer=regularizers.l1(regularizer1))(x1)
		x1 = TimeDistributed(BatchNormalization(),name='bnorm1_pre_%d'%(cnt1))(x1)
		print(n_filters,kernel_size1,activation,pool_length1,drop_out_rate)
		x1 = TimeDistributed(Activation(activation),name='activation1_pre_%d'%(cnt1))(x1)

		if pool_length1>1:
			x1 = TimeDistributed(MaxPooling1D(pool_size = pool_length1, strides = stride1), name='pooling_pre_%d'%(cnt1))(x1)
		if drop_out_rate>0:
			x1 = TimeDistributed(Dropout(drop_out_rate),name='dropout1_pre_%d'%(cnt1))(x1)

	layer_1 = x1
	if len(conv_list1)>0:
		x1 = TimeDistributed(Flatten(),name='Flatten_local_1')(x1)

	# dilated convolution
	cnt2 = 0
	for conv_1 in conv_list2:
		# dilation rate: 1, 2, 4, 8, 16, 32
		if cnt2>=1:
			x2 = Concatenate(axis=-1,name='concatenate_local_%d'%(cnt2))([x1,x2])
		else:
			x2 = x1
		n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate = conv_1
		# x1 = TimeDistributed(Conv1D(filters = n_filters, kernel_size = kernel_size1, strides = stride, padding=boundary_vec[boundary], activation = "linear",
		# 			kernel_regularizer=regularizers.l2(regularizer2),
		# 			dilation_rate = dilation_rate1),
		# 			name = 'conv2_pre_%d'%(cnt2))(x2)
		x1 = Conv1D(filters = n_filters, kernel_size = kernel_size1, strides = stride, padding=boundary_vec[boundary], 
					activation = "linear",
					kernel_regularizer=regularizers.l2(regularizer2),
					dilation_rate = dilation_rate1,
					name = 'conv2_pre_%d'%(cnt2))(x2)
		# x1 = Conv1D(filters = n_filters, kernel_size = kernel_size1, activation = "linear",
		# 			kernel_regularizer=regularizers.l2(regularizer2),
		# 			activity_regularizer=regularizers.l1(regularizer1))(x1)
		x1 = BatchNormalization(name='bnorm2_pre_%d'%(cnt2))(x1)
		print(n_filters,kernel_size1,activation,pool_length1,drop_out_rate)
		x1 = Activation(activation,name='activation2_pre_%d'%(cnt2))(x1)

		# if pool_length1>1:
		# 	x1 = TimeDistributed(MaxPooling1D(pool_size = pool_length1, strides = stride1), name='pooling_pre_%d'%(cnt1))(x1)
		if drop_out_rate>0:
			x1 = Dropout(drop_out_rate,name='dropout2_pre_%d'%(cnt2))(x1)

		cnt2 += 1

	x2 = Concatenate(axis=-1,name='concatenate_local_%d'%(cnt2))([x1,x2])

	return x2

# network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling and multiple convolution layers
def get_model2a1_attention_1_2_2_sample5(config):

	feature_dim, output_dim = config['feature_dim'], config['output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	loss_function = config['loss_function']

	# n_step_local_ori, n_step_local = config['n_step_local_ori'], config['n_step_local']
	n_step_local_ori = config['n_step_local_ori']
	# input_shape_1 = [n_step_local,feature_dim]
	# return_sequences_flag1 = True
	# config.update({'input_shape_1':input_shape_1})
	# encoder_1 = get_model2a1_basic5(config)
	# encoder_1.summary()

	input_region = Input(shape=(n_steps,n_step_local_ori,feature_dim))
	# layer_2 = TimeDistributed(encoder_1,name='encoder_1')(input_region) # shape: (n_steps,feature_dim2*2)
	# print(layer_2.shape)

	# feature_dim1, feature_dim2, return_sequences_flag = config['local_vec_1']
	# if return_sequences_flag==True:
	# 	if config['attention2_local']==1:
	# 		layer_2, attention2 = TimeDistributed(SeqSelfAttention(return_attention=True, attention_activation=activation_self),name='attention_local_1')(layer_2)

	# 	layer_2 = TimeDistributed(GlobalMaxPooling1D(),name='global_pooling_local_1')(layer_2)

	# layer_2 = get_model2a1_basic5_1(input_region,config)
	layer_2 = get_model2a1_basic5_convolution(input_region,config)
	print(layer_2.shape)

	config['sample_attention'] = 1
	if config['sample_attention']>=1:
		number, typeid = 3, 2
		units1 = config['units1']
		config['units1'] = 0
		logits_T = construct_gumbel_selector1(layer_2,config,number,typeid) # shape: (n_steps,1)
		config['units1'] = units1

		# k = 10
		if config['sample_attention']==1:
			tau, k, typeid_sample = config['tau'], config['n_select'], config['typeid_sample']
			print('sample_attention',tau,k,typeid,activation3,typeid_sample)
			
			if activation3=='linear':
				typeid_sample = 1
			elif activation3=='tanh':
				typeid_sample = 5
			else:
				pass
			attention1 = Sample_Concrete1(tau,k,n_steps,typeid_sample)(logits_T) # output shape: (batch_size, n_steps, 1)
		else:
			attention1 = logits_T

		# encode the input 2
		if config['select2']==1:
			dense_layer_output1 = construct_basic1(layer_2,config)
		else:
			dense_layer_output1 = layer_2

		dense_layer_output2 = Multiply()([dense_layer_output1, attention1])
	else:
		dense_layer_output2 = layer_2

	config['activation2'] = ''
	output = get_model2a1_basic1(dense_layer_output2,config)

	# output = Activation("softmax")(output)
	model = Model(input = input_region, output = output)

	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	optimizer = Adam(learning_rate = lr, clipnorm=CLIPNORM1)
	# optimizer = find_optimizer(config)

	model.compile(optimizer=optimizer, loss = loss_function)

	model.summary()
	
	return model

# network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling and multiple convolution layers
def get_model2a1_attention_1_2_2_sample5_1(config):

	feature_dim, output_dim = config['feature_dim'], config['output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	loss_function = config['loss_function']

	# n_step_local_ori, n_step_local = config['n_step_local_ori'], config['n_step_local']
	n_step_local_ori = config['n_step_local_ori']
	# input_shape_1 = [n_step_local,feature_dim]
	# return_sequences_flag1 = True
	# config.update({'input_shape_1':input_shape_1})
	# encoder_1 = get_model2a1_basic5(config)
	# encoder_1.summary()

	input_region = Input(shape=(n_steps,n_step_local_ori,feature_dim))
	# layer_2 = TimeDistributed(encoder_1,name='encoder_1')(input_region) # shape: (n_steps,feature_dim2*2)
	# print(layer_2.shape)

	# feature_dim1, feature_dim2, return_sequences_flag = config['local_vec_1']
	# if return_sequences_flag==True:
	# 	if config['attention2_local']==1:
	# 		layer_2, attention2 = TimeDistributed(SeqSelfAttention(return_attention=True, attention_activation=activation_self),name='attention_local_1')(layer_2)

	# 	layer_2 = TimeDistributed(GlobalMaxPooling1D(),name='global_pooling_local_1')(layer_2)

	# layer_2 = get_model2a1_basic5_1(input_region,config)
	layer_2 = get_model2a1_basic5_convolution(input_region,config)
	print(layer_2.shape)

	config['sample_attention'] = 1
	if config['sample_attention']>=1:
		number, typeid = 3, 2
		units1 = config['units1']
		config['units1'] = 0
		logits_T = utility_1_5.construct_gumbel_selector2_ori(layer_2,config,number,typeid) # shape: (n_steps,1)
		config['units1'] = units1

		# k = 10
		if config['sample_attention']==1:
			tau, k, typeid_sample = config['tau'], config['n_select'], config['typeid_sample']
			print('sample_attention',tau,k,typeid,activation3,typeid_sample)
			
			if activation3=='linear':
				typeid_sample = 1
			elif activation3=='tanh':
				typeid_sample = 5
			else:
				pass
			attention1 = Sample_Concrete1(tau,k,n_steps,typeid_sample)(logits_T) # output shape: (batch_size, n_steps, 1)
		else:
			attention1 = logits_T

		# encode the input 2
		if config['select2']==1:
			dense_layer_output1 = construct_basic1(layer_2,config)
		else:
			dense_layer_output1 = layer_2

		dense_layer_output2 = Multiply()([dense_layer_output1, attention1])
	else:
		dense_layer_output2 = layer_2

	config['activation2'] = ''
	output = utility_1_5.get_model2a1_basic1_2(dense_layer_output2,config)

	# output = Activation("softmax")(output)
	model = Model(input = input_region, output = output)

	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	optimizer = Adam(learning_rate = lr, clipnorm=CLIPNORM1)
	# optimizer = find_optimizer(config)

	model.compile(optimizer=optimizer, loss = loss_function)

	model.summary()
	
	return model

# dilated convolutions
def get_model2a1_attention_1_2_2_sample6(config):

	feature_dim, output_dim = config['feature_dim'], config['output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	loss_function = config['loss_function']

	n_step_local_ori = config['n_step_local_ori']
	input_region = Input(shape=(n_steps,n_step_local_ori,feature_dim))

	layer_2 = get_model2a1_basic5_convolution1(input_region,config)
	print(layer_2.shape)

	if 'feature_dim3' in config:
		feature_dim3 = config['feature_dim3']
	else:
		feature_dim3 = []
	regularizer2 = config['regularizer2']

	if 'activation_basic' in config:
		activation_basic = config['activation_basic']
	else:
		activation_basic = 'sigmoid'

	cnt1 = 0
	x1 = layer_2
	for t_feature_dim3 in feature_dim3:
		cnt1 += 1
		x1 = TimeDistributed(Dense(t_feature_dim3,
								kernel_regularizer=regularizers.l2(regularizer2)),
								name = 'conv1_3_%d'%(cnt1))(x1)
		x1 = TimeDistributed(BatchNormalization(),name='bnorm1_3_%d'%(cnt1))(x1)
		x1 = TimeDistributed(Activation('relu'),name='activation1_3_%d'%(cnt1))(x1)

	if 'output_dim_1' in config:
		output_dim1 = config['output_dim_1']
	else:
		output_dim1 = 1
	output = Dense(output_dim1,name='dense2')(x1)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation(activation_basic,name='activation2')(output)

	# output = Activation("softmax")(output)
	model = Model(input = input_region, output = output)
	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	optimizer = Adam(learning_rate = lr, clipnorm=CLIPNORM1)
	# optimizer = find_optimizer(config)
	model.compile(optimizer=optimizer, loss = loss_function)

	model.summary()
	
	return model

# dilated convolutions, sequence features
def get_model2a1_attention_1_2_2_sample6_1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	if not('loss_function' in config):
		loss_function = 'mean_squared_error'
	else:
		loss_function = config['loss_function']

	input1 = Input(shape = (n_steps,feature_dim))

	number = 1
	# if 'typeid2' in config:
	# 	typeid = config['typeid2']
	# else:
	# 	typeid = 2
	typeid = 2
	# logits_T = construct_gumbel_selector1(input1,config,number,typeid)

	# k = 10
	# encode the input 2
	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	layer_1 = dense_layer_output1

	config['local_conv_list1'] = []
	layer_2 = get_model2a1_basic5_convolution1(layer_1,config)
	print(layer_2.shape)

	# output = get_model2a1_basic1(dense_layer_output2,config)
	if 'feature_dim3' in config:
		feature_dim3 = config['feature_dim3']
	else:
		feature_dim3 = []
	regularizer2 = config['regularizer2']
	# regularizer2_2 = config['regularizer2_2']
	if 'activation_basic' in config:
		activation_basic = config['activation_basic']
	else:
		activation_basic = 'sigmoid'

	cnt1 = 0
	x1 = layer_2
	for t_feature_dim3 in feature_dim3:
		cnt1 += 1
		x1 = TimeDistributed(Dense(t_feature_dim3,
								kernel_regularizer=regularizers.l2(regularizer2)),
								name = 'conv1_3_%d'%(cnt1))(x1)
		x1 = TimeDistributed(BatchNormalization(),name='bnorm1_3_%d'%(cnt1))(x1)
		x1 = TimeDistributed(Activation('relu'),name='activation1_3_%d'%(cnt1))(x1)

	if 'output_dim_1' in config:
		output_dim1 = config['output_dim_1']
	else:
		output_dim1 = 1
	output = Dense(output_dim1,name='dense2')(x1)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation(activation_basic,name='activation2')(output)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	optimizer = Adam(learning_rate = lr, clipnorm=CLIPNORM1)
	# optimizer = find_optimizer(config)
	model.compile(optimizer=optimizer, loss = loss_function)

	model.summary()

	return model

def get_model2a1_attention(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (None,feature_dim))
	lr = config['lr']
	activation = config['activation']

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	# x1 = Activation('tanh',name='activation')(x1)
	if activation!='':
		x1 = Activation(activation,name='activation')(x1)
	# x1 = Flatten()(x1)

	x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# x_1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# x1 = x_1[0]
	# attention = x_1[1]
	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim,name='dense1')(x1)
		dense1 = BatchNormalization(name='batchnorm1')(dense1)
		dense1 = Activation(activation,name='activation1')(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
	else:
		dense_layer_output = x1

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(dense_layer_output)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation("sigmoid",name='activation2')(output)
	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = lr)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()

	return model

# select sample
def sample_select2a(x_mtx, y, idx_sel_list, tol=5, L=5):

	num1 = len(idx_sel_list)
	# L = 5
	size1 = 2*L+1
	feature_dim = x_mtx.shape[1]
	vec1_list = np.zeros((num1,size1))
	vec2_list = np.zeros((num1,size1))
	# feature_list = np.zeros((num1,size1*feature_dim))
	feature_list = np.zeros((num1,size1,feature_dim))
	signal_list = np.zeros((num1,size1))
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
		vec2_list[i] = vec1
		feature_list[i] = t_feature
		signal_list[i] = y[vec1]

		if i%50000==0:
			print(i,t_feature.shape,vec1,vec1_list[i])

	signal_list = np.expand_dims(signal_list, axis=-1)

	return feature_list, signal_list, vec1_list, vec2_list

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

	for i in range(0,num1):
		s1, s2 = seq_list[i][0], seq_list[i][1]+1
		serial = ref_serial[s1:s2]
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

		if i%10==0:
			print(i,num2,vec1_local[s1],vec1_serial[s1])

	# signal_mtx = signal_mtx[:,np.newaxis]
	signal_mtx = np.expand_dims(signal_mtx, axis=-1)

	# signal_mtx = np.expand_dims(signal_ntx, axis=-1)

	return feature_mtx, signal_mtx, vec1_serial, vec1_local

def read_predict(y, vec, idx, flanking1=3, type_id=0, base1=0.25):

	num1, context_size = vec.shape[0], vec.shape[1]
	if len(idx)==0:
		idx = range(0,num1)

	a1 = np.asarray(range(0,context_size))
	a2 = np.ones((num1,1))
	mtx1 = np.outer(a2,a1)
	# weight = 0.5*np.ones(context_size)
	weight = np.ones(context_size)
	L = int((context_size-1)*0.5)

	if type_id==1:
		base1 = base1
		for i in range(0,L+1):
			weight[i] = base1+(1-base1)*i/L
		for i in range(L,context_size):
			weight[i] = 1-(1-base1)*(i-L)/L

		if flanking1<=L:
			idx_range = np.asarray(range(L-flanking1,L+flanking1+1))
			weight[idx_range] = 1

	# weight = weight/np.sum(weight)
	weight_vec = np.outer(a2,weight)
	# print(num1,context_size,L,idx_range)
	# print(weight)

	serial1 = vec[:,L]
	t1 = np.sum(serial1!=idx)
	if t1>0:
		print("error! read predict %d"%(t1))
		return

	## idx_vec, pos_vec, weight_vec = np.ravel(vec), np.ravel(mtx1), np.ravel(mtx2)
	# idx_vec, weight_vec = np.ravel(vec), np.ravel(mtx2)
	# y1 = np.ravel(y)
	# value = np.zeros(num1)
	# for i in range(0,num1):
	# 	b1 = np.where(idx_vec==idx[i])[0]
	# 	if len(b1)==0:
	# 		print("error! %d %d"%(i,idx[i]))
	# 	t_weight = weight_vec[b1]
	# 	t_weight = t_weight*1.0/np.sum(t_weight)
	# 	value[i] = np.dot(y1[b1],t_weight)

	value = np.zeros(num1)
	y = y.reshape((y.shape[0],y.shape[1]))
	for i in range(0,num1):
		b1 = (vec==idx[i])
		# if len(b1)==0:
		# 	print("error! %d %d"%(i,idx[i]))
		t_weight = weight_vec[b1]
		t_weight = t_weight*1.0/np.sum(t_weight)
		value[i] = np.dot(y[b1],t_weight)

	return value

def read_predict_1(y, vec, idx, flanking1=3, type_id=0, base1=0.25):

	num1, context_size = vec.shape[0], vec.shape[1]
	if len(idx)==0:
		idx = range(0,num1)
	sample_num1 = len(idx)

	L = int((context_size-1)*0.5)
	serial1 = vec[:,L]
	# t1 = np.sum(serial1!=idx)
	# if t1>0:
	# 	print("error! read predict %d"%(t1))
	# 	return
	assert list(serial1)==list(idx)

	## idx_vec, pos_vec, weight_vec = np.ravel(vec), np.ravel(mtx1), np.ravel(mtx2)
	# idx_vec, weight_vec = np.ravel(vec), np.ravel(mtx2)
	# y1 = np.ravel(y)
	# value = np.zeros(num1)
	# for i in range(0,num1):
	# 	b1 = np.where(idx_vec==idx[i])[0]
	# 	if len(b1)==0:
	# 		print("error! %d %d"%(i,idx[i]))
	# 	t_weight = weight_vec[b1]
	# 	t_weight = t_weight*1.0/np.sum(t_weight)
	# 	value[i] = np.dot(y1[b1],t_weight)

	# y = y.reshape((y.shape[0],y.shape[1]))
	dim1 = y.shape[-1]
	value = np.zeros((sample_num1,dim1),dtype=np.float32)
	if type_id==0:
		for i in range(0,sample_num1):
			b1 = (vec==idx[i])
			value[i] = np.mean(y[b1],axis=0)
	else:
		a1 = np.asarray(range(0,context_size))
		a2 = np.ones((num1,1))
		mtx1 = np.outer(a2,a1)
		# weight = 0.5*np.ones(context_size)
		weight = np.ones(context_size)

		base1 = base1
		for i in range(0,L+1):
			weight[i] = base1+(1-base1)*i/L
		for i in range(L,context_size):
			weight[i] = 1-(1-base1)*(i-L)/L

		if flanking1<=L:
			idx_range = np.asarray(range(L-flanking1,L+flanking1+1))
			weight[idx_range] = 1

		# weight = weight/np.sum(weight)
		weight_vec = np.outer(a2,weight)
		# print(num1,context_size,L,idx_range)
		# print(weight)
		for i in range(0,sample_num1):
			b1 = (vec==idx[i])
			if len(b1)==0:
				print("error! %d %d"%(i,idx[i]))
			t_weight = weight_vec[b1]
			t_weight = t_weight*1.0/np.sum(t_weight)
			t_weight = np.tile(t_weight,[dim1,1]).T
			# value[i] = np.dot(y[b1],t_weight)
			value[i] = np.sum(y[b1]*t_weight,axis=0)

	return value

def read_predict_weighted(y, vec, idx, flanking1=3):

	num1, context_size = vec.shape[0], vec.shape[1]
	if len(idx)==0:
		idx = range(0,num1)

	a1 = np.asarray(range(0,context_size))
	a2 = np.ones((num1,1))
	mtx1 = np.outer(a2,a1)
	base1 = 0.25
	weight = base1*np.ones(context_size)
	L = int((context_size-1)*0.5)
	for i in range(0,context_size):
		if i<=L:
			weight[i] = base1+(1-base1)*i/L
		else:
			weight[i] = 1-(1-base1)*(i-L)/L
	idx_range = np.asarray(range(L-flanking1,L+flanking1+1))
	weight[idx_range] = 1
	mtx2 = np.outer(a2,weight)
	print(num1,context_size,L,idx_range)
	print(weight)

	serial1 = vec[:,L]
	t1 = np.sum(serial1!=idx)
	if t1>0:
		print("error! %d"%(t1))

	idx_vec, pos_vec, weight_vec = np.ravel(vec), np.ravel(mtx1), np.ravel(mtx2)
	y1 = np.ravel(y)
	value = np.zeros(num1)
	for i in range(0,num1):
		b1 = np.where(idx_vec==idx[i])[0]
		if len(b1)==0:
			print("error! %d %d"%(i,idx[i]))
		t_weight = weight_vec[b1]
		t_weight = t_weight*1.0/np.sum(t_weight)
		value[i] = np.dot(y1[b1],t_weight)

	return value

def dot_layer(inputs):
	x,y = inputs

	return K.sum(x*y,axis = -1,keepdims=True)

def corr(y_true, y_pred):
	return np.min(np.corrcoef(y_true,y_pred))

def score_function(y_test, y_pred, y_proba):

	auc = roc_auc_score(y_test,y_proba)
	aupr = average_precision_score(y_test,y_proba)
	precision = precision_score(y_test,y_pred)
	recall = recall_score(y_test,y_pred)
	accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))
	F1 = 2*precision*recall/(precision+recall)
	# print(auc,aupr,precision,recall)
	
	return accuracy, auc, aupr, precision, recall, F1

def score_function_group(y_test, y_pred, y_proba, group_label):

	group_label_vec = np.unique(group_label)
	num1 = len(group_label_vec)
	y_test_group = np.zeros(num1,dtype=np.int32)
	y_pred_group = np.zeros(num1,dtype=np.int32)
	y_prob_group = np.zeros(num1,dtype=np.float32)

	for i in range(num1):
		t_label = group_label_vec[i]
		id1 = np.where(group_label==t_label)[0]
		id2 = (y_test[id1]!=y_test[id1[0]])
		if np.sum(id2)>0:
			print('error!')
			return -1
		y_test_group[i] = y_test[id1[0]]
		y_pred_group[i] = np.max(y_pred[id1])
		y_prob_group[i] = np.max(y_proba[id1])
		# print(t_label,id1,y_test[id1],y_pred[id1],y_proba[id1])

	# print(auc,aupr,precision,recall)
	accuracy, auc, aupr, precision, recall, F1 = score_function(y_test_group,y_pred_group,y_prob_group)
	
	return accuracy, auc, aupr, precision, recall, F1

def load_samples(chrom_vec,chrom,y_label_ori,y_group1,y_signal_ori1,filename2,filename2a,kmer_size,kmer_dict1,generate):

	x_mtx_vec, y_label_vec, y_group_vec, y_signal_ori_vec = [], [], [], []
	for chrom_id in chrom_vec:
		chrom_id1 = 'chr%s'%(chrom_id)
		sel_idx = np.where(chrom==chrom_id1)[0]
		print(('sel_idx:%d')%(len(sel_idx)))

		if generate==0:
			filename2 = 'training_mtx/training2_mtx_%s.npy'%(chrom_id)
			if(os.path.exists(filename2)==True):
				x_mtx = np.load(filename2)
				x_kmer = np.load('training2_kmer_%s.npy'%(chrom_id))
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

def score_2(y, y_predicted):

	score1 = mean_squared_error(y, y_predicted)
	score2 = pearsonr(y, y_predicted)

	return score1, score2

def load_kmer_single(species_name):

	path1 = '/volume01/yy3/seq_data/dl/replication_timing'
	filename1 = '%s/training2_kmer_%s.npy'%(path1,species_name)
	filename2 = '%s/training2_kmer_%s.serial.npy'%(path1,species_name)

	data1 = np.load(filename1)
	t_serial = np.load(filename2)

	filename3 = '%s/estimate_rt/estimate_rt_%s.txt'%(path1,species_name)
	filename3a = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path1,species_name)
	temp1 = pd.read_csv(filename3,sep='\t')
	temp2 = np.read_csv(filename3a,sep='\t')
	colname1, colname2 = list(temp1), list(temp2)
	chrom1, start1, stop1, serial1 = temp1[colname1[0]], temp1[colname1[1]], temp1[colname1[2]], temp1[colname1[3]]
	chrom2, start2, stop2, serial2 = temp2[colname2[0]], temp2[colname2[1]], temp2[colname2[2]], temp2[colname2[3]]

	map_idx = mapping_Idx(serial1,serial2)
	
	data1_sub = data1[map_idx]
	print(data1.shape, data1_sub.shape)

	return data1_sub, map_idx

# the mapped indices of selected regions
def load_map_idx(species_name):

	path1 = './'
	filename3 = '%s/estimate_rt/estimate_rt_%s.txt'%(path1,species_name)
	filename3a = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path1,species_name)
	temp1 = pd.read_csv(filename3,sep='\t')
	temp2 = pd.read_csv(filename3a,sep='\t')
	colname1, colname2 = list(temp1), list(temp2)
	chrom1, start1, stop1, serial1 = temp1[colname1[0]], temp1[colname1[1]], temp1[colname1[2]], temp1[colname1[3]]
	chrom2, start2, stop2, serial2 = temp2[colname2[0]], temp2[colname2[1]], temp2[colname2[2]], temp2[colname2[3]]

	map_idx = mapping_Idx(serial1,serial2)

	return serial1, serial2, map_idx

def dimension_reduction(x_ori,feature_dim,shuffle,sub_sample,type_id):

	if shuffle==1 and sub_sample>0:
		idx = np.random.permutation(x_ori.shape[0])
	else:
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
		# X_pca_reconst = pca.inverse_transform(x)
	elif type_id==1:
		# Incremental PCA
		n_batches = 10
		inc_pca = IncrementalPCA(n_components=feature_dim)
		for X_batch in np.array_split(x_ori, n_batches):
			inc_pca.partial_fit(X_batch)
		x = inc_pca.transform(x_ori)
		# X_ipca_reconst = inc_pca.inverse_transform(x)
	elif type_id==2:
		# Kernel PCA
		kpca = KernelPCA(kernel="rbf",n_components=feature_dim, gamma=None, fit_inverse_transform=True, random_state = 0, n_jobs=50)
		kpca.fit(x_ori[id1,:])
		x = kpca.transform(x_ori)
		# X_kpca_reconst = kpca.inverse_transform(x)
	elif type_id==3:
		# Sparse PCA
		sparsepca = SparsePCA(n_components=feature_dim, alpha=0.0001, random_state=0, n_jobs=50)
		sparsepca.fit(x_ori[id1,:])
		x = sparsepca.transform(x_ori)
	elif type_id==4:
		# SVD
		SVD_ = TruncatedSVD(n_components=feature_dim,algorithm='randomized', random_state=0, n_iter=5)
		SVD_.fit(x_ori[id1,:])
		x = SVD_.transform(x_ori)
		# X_svd_reconst = SVD_.inverse_transform(x)
	elif type_id==5:
		# Gaussian Random Projection
		GRP = GaussianRandomProjection(n_components=feature_dim,eps = 0.5, random_state=2019)
		GRP.fit(x_ori[id1,:])
		x = GRP.transform(x_ori)
	elif type_id==6:
		# Sparse random projection
		SRP = SparseRandomProjection(n_components=feature_dim,density = 'auto', eps = 0.5, random_state=2019, dense_output = False)
		SRP.fit(x_ori[id1,:])
		x = SRP.transform(x_ori)
	elif type_id==7:
		# MDS
		mds = MDS(n_components=feature_dim, n_init=12, max_iter=1200, metric=True, n_jobs=4, random_state=2019)
		x = mds.fit_transform(x_ori[id1])
	elif type_id==8:
		# ISOMAP
		isomap = Isomap(n_components=feature_dim, n_jobs = 4, n_neighbors = 5)
		isomap.fit(x_ori[id1,:])
		x = isomap.transform(x_ori)
	elif type_id==9:
		# MiniBatch dictionary learning
		miniBatchDictLearning = MiniBatchDictionaryLearning(n_components=feature_dim,batch_size = 1000,alpha = 1,n_iter = 25,  random_state=2019)
		if sub_sample>0:
			miniBatchDictLearning.fit(x_ori[id1,:])
			x = miniBatchDictLearning.transform(x_ori)
		else:
			x = miniBatchDictLearning.fit_transform(x_ori)
	elif type_id==10:
		# ICA
		fast_ICA = FastICA(n_components=feature_dim, algorithm = 'parallel',whiten = True,max_iter = 100,  random_state=2019)
		if sub_sample>0:
			fast_ICA.fit(x_ori[id1])
			x = fast_ICA.transform(x_ori)
		else:
			x = fast_ICA.fit_transform(x_ori)
		# X_fica_reconst = FastICA.inverse_transform(x)
	elif type_id==11:
		# t-SNE
		tsne = TSNE(n_components=feature_dim,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
		x = tsne.fit_transform(x_ori)
	elif type_id==12:
		# Locally linear embedding
		lle = LocallyLinearEmbedding(n_components=feature_dim, n_neighbors = np.max((int(feature_dim*1.5),500)),method = 'modified', n_jobs = 20,  random_state=2019)
		lle.fit(x_ori[id1,:])
		x = lle.transform(x_ori)
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

	return x

def feature_transform(x_train, x_test, feature_dim_kmer, feature_dim, shuffle, sub_sample_ratio, type_id, normalize):

	x_ori1 = np.vstack((x_train,x_test))
	dim1 = x_ori1.shape[1]
	dim2 = dim1-feature_dim_kmer
	print("feature_dim_kmer",feature_dim_kmer,dim2)
	x_ori = x_ori1[:,dim2:]
	if normalize>=1:
		sc = StandardScaler()
		x_ori = sc.fit_transform(x_ori)	# normalize data
	# x_train_sub = sc.fit_transform(x_ori[0:num_train,:])
	# x_test_sub = sc.transform(x_ori[num_train+num_test,:])
	# x_train_sub = sc.fit_transform(x_ori[0:num_train,:])
	# x_test_sub = sc.transform(x_ori[num_train+num_test,:])
	num_train, num_test = x_train.shape[0], x_test.shape[0]
	vec1 = ['PCA','Incremental PCA','Kernel PCA','Sparse PCA','SVD','GRP','SRP','MDS','ISOMAP','Minibatch','ICA','tSNE','LLE','Encoder']
	start = time.time()
	if sub_sample_ratio<1:
		sub_sample = int(x_ori.shape[0]*sub_sample_ratio)
	else:
		sub_sample = -1
	x = dimension_reduction(x_ori,feature_dim,shuffle,sub_sample,type_id)
	stop = time.time()
	print("feature transform %s"%(vec1[type_id]),stop - start)
	x1 = np.hstack((x_ori1[:,0:dim2],x))
	if normalize>=2:
		sc = StandardScaler()
		x1 = sc.fit_transform(x1)
	x_train1, x_test1 = x1[0:num_train], x1[num_train:num_train+num_test]
	print(x_train.shape,x_train1.shape,x_test.shape,x_test1.shape)

	return x_train1, x_test1

# select sample
def sample_select(x_mtx, idx_sel_list, tol=5, L=5):

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



