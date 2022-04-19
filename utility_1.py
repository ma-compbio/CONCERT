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
from keras.layers import LSTM, Bidirectional
from keras.layers import BatchNormalization, Dropout, Concatenate, Embedding
from keras.layers import Activation,Dot,dot
from keras.layers import TimeDistributed, RepeatVector, Permute, merge, Multiply
from keras.models import Sequential, Model, clone_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.constraints import unitnorm
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU
clipped_relu = lambda x: relu(x, max_value=1.0)

import sklearn as sk
from sklearn.svm import SVR
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

import os.path
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
plt.switch_backend('Agg')
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD, FastICA, MiniBatchDictionaryLearning
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import KFold, train_test_split

from timeit import default_timer as timer
import time

from scipy import stats
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from scipy.stats import pearsonr, spearmanr
from scipy.stats import wilcoxon, mannwhitneyu,kstest,ks_2samp
from scipy.stats import chisquare
from scipy import signal
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import h5py

import os.path
from optparse import OptionParser
from timeit import default_timer as timer
import time
clipped_relu = lambda x: relu(x, max_value=1.0)

import multiprocessing as mp
import threading
import sys
# import utility_1_5

n_epochs = 100
drop_out_rate = 0.5
learning_rate = 0.001
validation_split_ratio = 0.1
BATCH_SIZE = 128	# previous: 128
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
def find_region_sub2(filename1,output_filename1,config={}):

	data1 = pd.read_csv(filename1,sep='\t')
	colnames = list(data1)
	colnames_1 = ['sel1','sel2','sel2.1','sel3']
	label_1 = np.asarray(data1.loc[:,colnames_1])
	sel1, sel2, sel2_1 = label_1[:,0], label_1[:,1], label_1[:,2]
	b1, b2, b3 = np.where(sel1>0)[0], np.where(sel2>0)[0], np.where(sel2_1>0)[0]
	b_1, b_2, b_3 = np.where(sel1!=0)[0], np.where(sel2!=0)[0], np.where(sel2_1!=0)[0]

	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	sel_column = ['Q2']
	predicted_attention, predicted_attention1 = np.asarray(data1['predicted_attention']), np.asarray(data1[sel_column])
	thresh = config['thresh_select']
	id1 = np.where(predicted_attention1[b3]>thresh)[0]
	id1 = b3[id1]

	id2 = np.union1d(b2,id1) 	# local peaks
	id3 = np.union1d(b1,id2)	# local peaks or high scores

	id5 = np.intersect1d(b2,id3) # local peaks
	id6 = np.intersect1d(b1,id5) # local peaks and high values
	print('select',len(b1),len(b2),len(b3),len(id1),len(id2),len(id3),len(id5),len(id6))

	sample_num = data1.shape[0]
	t_label = np.zeros(sample_num,dtype=np.int8)
	list1 = [b1,b2,id3,id6]
	for i in range(len(list1)):
		t_label[list1[i]] = i+1

	sel_id = np.where(t_label>0)[0]
	data1 = data1.loc[sel_id,['chrom','start','stop','serial','predicted_attention','Q2']]
	data1.reset_index(drop=True,inplace=True)
	data1['label'] = t_label[t_label>0]

	data1.to_csv(output_filename1,index=False,sep='\t')

	return True

# write genomie loci to bed file
def find_region_sub2_1(filename1,output_filename1,config={}):

	data1 = pd.read_csv(filename1,sep='\t')
	colnames = list(data1)
	
	# high value, local peak (distance>=1), local peak with signal difference (distance>=1), wavelet local peak
	colnames_1 = ['sel1','sel2','sel2.1','sel3']
	num1 = len(colnames)
	# local peak with different distances
	for i in range(num1):
		t_colname = colnames[i]
		if t_colname.find('sel2.0')>=0:
			# print(1,t_colname)
			colnames_1.append(t_colname)
		# else:
		# 	print(0,t_colname)

	colnames_2 = colnames_1[0:3]+colnames_1[4:]
	num2 = len(colnames_2)
	print('colnames_2',colnames_2)

	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	sel_column = 'Q2'
	predicted_attention, predicted_attention1 = np.asarray(data1['predicted_attention']), np.asarray(data1[sel_column])
	thresh = config['thresh_select']

	region_num = data1.shape[0]
	mask = np.zeros((region_num,num2),dtype=np.int8)
	thresh = config['thresh_select']
	thresh_2 = config['thresh_2']
	for i in range(num2):
		t_colname = colnames_2[i]
		t_column = np.asarray(data1[t_colname])
		b1 = np.where(t_column>0)[0]
		if t_colname=='sel2.1':
			id1 = np.where(predicted_attention1[b1]>thresh)[0]
			b1 = b1[id1]
		if t_colname=='sel1':
			id1 = np.where(predicted_attention1[b1]>thresh_2)[0]
			b1 = b1[id1]
		mask[b1,i] = 1

	label_value = np.zeros(region_num,dtype=np.int32)
	for i1 in range(num2):
		label_value = label_value + (10**i1)*mask[:,i1]

	t_label = label_value

	sel_id = np.where(t_label>0)[0]
	data1 = data1.loc[sel_id,['chrom','start','stop','serial','predicted_attention','Q2']]
	data1.reset_index(drop=True,inplace=True)
	data1['label'] = t_label[t_label>0]

	data1.to_csv(output_filename1,index=False,sep='\t')

	return True

# merge neighboring important genomic loci into regions
def find_region_sub3(filename1,output_filename1,config={}):

	data1 = pd.read_csv(filename1,sep='\t')
	chrom = np.asarray(data1['chrom'])
	t_score = np.asarray(data1['Q2'])

	t_label = np.asarray(data1['label'])
	colnames = list(data1)

	thresh1 = config['thresh_select']
	b1 = np.where(t_score>thresh1)[0]

	b2 = np.where(t_label>=3)[0]
	b1 = np.intersect1d(b1,b2)

	data1 = data1.loc[b1,:]
	data1.reset_index(drop=True,inplace=True)
	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	t_score = np.asarray(data1['Q2'])
	t_percentile = np.asarray(data1['label_1'])

	id1 = [int(chrom1[3:]) for chrom1 in chrom]
	idx_sel_list = np.column_stack((id1,serial))

	seq_list = generate_sequences(idx_sel_list,gap_tol=5)
	data_1 = output_generate_sequences(chrom,start,stop,serial,idx_sel_list,seq_list,output_filename='temp1.txt',save_mode=0)
	serial1, serial2 = np.asarray(data_1['serial1']), np.asarray(data_1['serial2'])
	num1 = len(serial1)
	list_1, list_2, list_3 = [], [], []
	for i in range(num1):
		b1 = np.where((serial<=serial2[i])&(serial>=serial1[i]))[0]
		list1 = [str(serial[i1]) for i1 in b1]
		list2 = ['%.4f'%(t_score1) for t_score1 in t_score[b1]]
		list3 = ['%.4f'%(t_percent1) for t_percent1 in t_percentile[b1]] # signal percentile
		d1 = ','
		str1 = d1.join(list1)
		str2 = d1.join(list2)
		list_1.append(str1)
		list_2.append(str2)
		list_3.append(d1.join(list3))
	
	data_1['loci'] = list_1
	data_1['score'] = list_2
	data_1['signal_percentile'] = list_3

	data_1.to_csv(output_filename1,index=False,sep='\t')

	return data_1

# generate serial for bed file
def find_region_sub3_1(filename1,genome_file='',chrom_num=19):

	data1 = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1)

	if len(colnames)<5:
		chrom, start, stop, signal = np.asarray(data1[0]), np.asarray(data1[1]), np.asarray(data1[2]), np.asarray(data1[3])
		
		serial, start_vec = generate_serial_start(genome_file,chrom,start,stop,chrom_num=chrom_num,type_id=0)
		data1['serial'] = serial
		id1 = np.where(serial>=0)[0]
		data1 = data1.loc[id1,colnames[0:3]+['serial']+colnames[3:]]
		b1 = filename1.find('.bedGraph')
		output_filename = filename1[0:b1]+'.bed'
		data1 = data1.sort_values(by=['serial'])
		data1.to_csv(output_filename,header=False,index=False,sep='\t')
	# else:
	# 	chrom, start, stop, signal = np.asarray(data1[0]), np.asarray(data1[1]), np.asarray(data1[2]), np.asarray(data1[4])
	# 	serial = np.asarray(data1[3])

	return data1

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

# generate domain labels
# input: filename1: original RT data with domain labels
# 		 filename_list: list of files with estimated importance scores
def find_region_sub3_3(filename1,filename_list):

	fields = ['chrom','start','stop','serial','signal','label','q1','q2','q_1','q_2','local_peak1','local_peak2']
	num1 = len(fields)
	data1_ori = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1_ori)
	if len(colnames)<num1:
		print('generate domain labels')
		data1_ori = find_region_sub3_2(filename1)

	chrom, start, stop = np.asarray(data1_ori[0]), np.asarray(data1_ori[1]), np.asarray(data1_ori[2])
	serial = np.asarray(data1_ori[3])
	label_ori = np.asarray(data1_ori[colnames[5]])
	print(serial)
	
	for filename2 in filename_list:
		data2 = pd.read_csv(filename2,sep='\t')
		t_chrom, t_start, t_stop = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop'])
		serial1, serial2 = np.asarray(data2['serial1']), np.asarray(data2['serial2'])
		# print(serial1[0:2])
		id1 = mapping_Idx(serial,serial1)
		id2 = mapping_Idx(serial,serial2)
		assert np.sum(id1<0)==0
		assert np.sum(id2<0)==0

		label1, label2 = label_ori[id1], label_ori[id2]
		b1 = np.max([label1,label2],axis=0)
		b2 = np.min([label1,label2],axis=0)
		sample_num = len(t_chrom)
		
		label_1 = np.int64(np.column_stack((b1,b2)))
		num1 = np.sum(b1>0)
		print(num1,sample_num,num1/sample_num)

		# column1 = ['chrom','start','stop','serial1','serial2','region_len','loci','score']
		column1 = list(data2)
		data2['label1'] = label_1[:,0]
		data2['label2'] = label_1[:,1]
		if not('label1' in column1):
			data2 = data2.loc[:,column1[0:5]+['label1','label2']+column1[5:]]
		data2.to_csv(filename2,index=False,sep='\t')
	
	return True

# generate domain labels
# input: filename1: ERCE file
#		 filename_list: list of files with estimated importance scores
def find_region_sub3_5(filename1,run_idList,filename_list,type_id1,tol=2,row=1):

	if row==1:
		data1 = pd.read_csv(filename1,sep='\t')
	else:
		data1 = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1)
	print(colnames)
	chrom = np.asarray(data1[colnames[0]])
	b1 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]

	data1 = data1.loc[b1,:]
	data1.reset_index(drop=True,inplace=True)
	chrom, start, stop = np.asarray(data1[colnames[0]]), np.asarray(data1[colnames[1]]), np.asarray(data1[colnames[2]])
	region_serial = np.asarray(data1[colnames[3]])
	region_num = len(chrom)
	print(run_idList)
	num1 = len(run_idList)
	vec1 = []
	query_serial = [1,2]
	id1 = mapping_Idx(region_serial,query_serial)

	for i in range(num1):
		run_id = run_idList[i]
		filename2 = filename_list[i]
		data2 = pd.read_csv(filename2,sep='\t')
		chrom2, start2, stop2 = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop'])
		serial1, serial2 = np.asarray(data2['serial1']), np.asarray(data2['serial2'])
		label_1 = np.asarray(data2['label1'])

		region_num2 = len(chrom2)
		t_label = np.zeros(region_num,dtype=np.int64) # label for ERCR
		t_label1 = np.zeros(region_num2,dtype=np.int64)	# label for genomic loci
		bin_size = 5000
		for l in range(region_num):
			t_chrom, t_start, t_stop = chrom[l], start[l], stop[l]
			t_start = t_start-tol*bin_size
			t_stop = t_stop+tol*bin_size
			b1 = np.where((chrom2==t_chrom)&(start2<t_stop)&(stop2>t_start))[0]
			t_label[l] = len(b1)
			if len(b1)>0:
				t_label1[b1] = region_serial[l]

		data1[run_id] = t_label
		print(id1)
		print(chrom[id1],start[id1],stop[id1],t_label[id1])
		if tol==0:
			t_column = 'label'
		else:
			t_column = 'label_%d'%(tol)
		data2[t_column] = t_label1
		b2 = np.where(t_label>0)[0]
		recall = len(b2)/region_num
		b3 = np.where(label_1>0)[0]
		b_3 = np.where(t_label1[b3]>0)[0]
		precision = len(b_3)/len(b3)
		vec1.append([run_id,recall,precision,len(b2),len(b_3),len(b3)]+list(t_label[id1]))
		data2.to_csv(filename2,index=False,sep='\t')

	vec1 = np.asarray(vec1)
	fields = ['run_id','recall','precision','ERCE_num','early_num1','early_num','chr8_1','chr8_2','chr16_1','chr16_2','chr16_3']
	data_2 = pd.DataFrame(columns=fields)
	data_2['run_id'] = np.int64(vec1[:,0])
	data_2.loc[:,fields[3:]] = np.int64(vec1[:,3:])
	data_2.loc[:,fields[1:3]] = vec1[:,1:3]
	file_path1 = './data_5'
	b1 = filename1.find('.bed')
	if b1<0:
		b1 = filename1.find('.txt')
	output_filename1 = filename1[0:b1]+'.%d.tol%d.1.txt'%(type_id1,tol)
	data_2.to_csv(output_filename1,index=False,sep='\t')

	output_filename1 = filename1[0:b1]+'.%d.tol%d.1.bed'%(type_id1,tol)
	data1.to_csv(output_filename1,index=False,sep='\t')

	return vec1

# generate domain labels
# input: filename1: ERCE file
#		 filename_list: list of files with estimated importance scores
def find_region_sub3_5_1(filename1,run_idList,filename_list,type_id1,
							tol=2,row=1,output_filename_annot='1'):

	if row==1:
		data1 = pd.read_csv(filename1,sep='\t')
	else:
		data1 = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1)
	print(colnames)
	chrom = np.asarray(data1[colnames[0]])
	b1 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]

	data1 = data1.loc[b1,:]
	data1.reset_index(drop=True,inplace=True)
	chrom, start, stop = np.asarray(data1[colnames[0]]), np.asarray(data1[colnames[1]]), np.asarray(data1[colnames[2]])
	region_serial = np.asarray(data1[colnames[3]])
	region_num = len(chrom)
	print(run_idList)
	num1 = len(run_idList)
	vec1 = []

	for i in range(num1):
		run_id = run_idList[i]
		filename2 = filename_list[i]
		data2 = pd.read_csv(filename2,sep='\t')
		chrom2, start2, stop2 = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop'])
		colnames_2 = list(data2)
		if 'label1' in colnames_2:
			label_1 = np.asarray(data2['label1'])
		else:
			label_1 = np.asarray(data2['label'])

		region_num2 = len(chrom2)
		t_label = np.zeros(region_num,dtype=np.int64) # label for ERCR
		t_label1 = np.zeros(region_num2,dtype=np.int64)	# label for genomic loci
		bin_size = 5000
		for l in range(region_num):
			t_chrom, t_start, t_stop = chrom[l], start[l], stop[l]
			t_start = t_start-tol*bin_size
			t_stop = t_stop+tol*bin_size
			b1 = np.where((chrom2==t_chrom)&(start2<t_stop)&(stop2>t_start))[0]
			t_label[l] = len(b1)
			if len(b1)>0:
				t_label1[b1] = region_serial[l]

		data1[run_id] = t_label
	
		t_column = 'label_%d'%(tol)
		data2[t_column] = t_label1
		b2 = np.where(t_label>0)[0]
		recall = len(b2)/region_num
		b3 = np.where(label_1>0)[0]
		b_3 = np.where(t_label1[b3]>0)[0]
		precision = len(b_3)/len(b3)
		vec1.append([run_id,recall,precision,len(b2),len(b_3),len(b3)])
		data2.to_csv(filename2,index=False,sep='\t')

	vec1 = np.asarray(vec1)
	fields = ['run_id','recall','precision','region_num','num1','num']
	data_2 = pd.DataFrame(columns=fields)
	data_2['run_id'] = np.int64(vec1[:,0])
	data_2.loc[:,fields[3:]] = np.int64(vec1[:,3:])
	data_2.loc[:,fields[1:3]] = vec1[:,1:3]
	# file_path1 = './data_5'
	b1 = filename1.find('.bed')
	if b1<0:
		b1 = filename1.find('.txt')
	# output_filename1 = filename1[0:b1]+'.%d.tol%d.%s.2.txt'%(type_id1,tol,output_filename_annot)
	output_filename1 = filename1[0:b1]+'.tol%d.%s.2.txt'%(tol,output_filename_annot)
	data_2.to_csv(output_filename1,index=False,sep='\t',mode='a')

	# output_filename1 = filename1[0:b1]+'.%d.tol%d.%s.2.bed'%(type_id1,tol,output_filename_annot)
	output_filename1 = filename1[0:b1]+'.tol%d.%s.2.bed'%(tol,output_filename_annot)
	data1.to_csv(output_filename1,index=False,sep='\t')

	return vec1

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
def find_region_sub3_6(filename_list,tol=0, thresh1=2):
	
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

def find_region():

	filename_list1 = []
	output_filename1 = []

# compare importance score
def compare_test_1(run_idList,celltype_id,filename1,annot1,typeid1=0,flanking=250,file_path2='',annot_vec=[]):

	type_id = int(typeid1)

	celltype_id = int(celltype_id)
	celltype_vec = ['GM12878','K562','H1-hESC','HCT116','H9','HEK293','IMR-90','RPE-hTERT','U2OS']
	celltype = celltype_vec[celltype_id]

	file_path = './'
	file_path1 = './data1'
	if file_path2=='':
		file_path2 = './data2'

	filename2 = '%s/%s.smooth.sorted.dnase.%d.bed'%(file_path1,celltype,flanking)
	region_data = pd.read_csv(filename1,header=None,sep='\t')
	annot_data = pd.read_csv(filename2,header=None,sep='\t')
	print(filename1,filename2)

	region_chrom, region_start, region_stop = np.asarray(region_data[0]), np.asarray(region_data[1]), np.asarray(region_data[2])
	region_num = len(region_chrom)

	chrom = np.asarray(annot_data[0])
	b2 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]
	annot_data = annot_data.loc[b2,:]
	annot_data.reset_index(drop=True,inplace=True)

	chrom, annot_serial = np.asarray(annot_data[0]), np.asarray(annot_data[3])
	annot_label2 = np.asarray(annot_data[5]) # DNase
	b2 = np.where(annot_label2>0)[0]
	annot_serial_1 = annot_serial[b2]
	print(len(annot_serial_1))

	num1 = len(run_idList)
	for i in range(num1):
		run_id = run_idList[i]
		filename3 = '%s/test_vec2_%d_52_[5].1_0.1.txt'%(file_path,run_id)
		pred_data = pd.read_csv(filename3,sep='\t')
		serial_ori = np.asarray(pred_data['serial'])
		# pred_attention = np.asarray(pred_data['predicted_attention'])

		id_1 = mapping_Idx(serial_ori,annot_serial_1)
		id_1 = id_1[id_1>=0]

		t_pred_data = pred_data.loc[id_1,['chrom','start','stop','serial','predicted_attention','Q2']]
		t_pred_data.reset_index(drop=True,inplace=True)

		serial = serial_ori[id_1]
		print(len(serial_ori),len(serial))

		chrom1, start1, stop1 = np.asarray(t_pred_data['chrom']), np.asarray(t_pred_data['start']), np.asarray(t_pred_data['stop'])

		chrom_vec = np.unique(chrom1)
		dict1 = dict()
		for chrom_id in chrom_vec:
			dict1[chrom_id] = np.where(chrom1==chrom_id)[0]

		list1 = []
		print(region_num)
		for l in range(region_num):
			if not(region_chrom[l] in chrom_vec):
				continue
			id1 = dict1[region_chrom[l]]
			b1 = np.where((start1[id1]<region_stop[l])&(stop1[id1]>region_start[l]))[0]
			b1 = id1[b1]
			if len(b1)>0:
				list1.extend(b1)
			if l%10000==0:
				print(len(b1),l)

		id1 = np.asarray(list1)

		sample_num = t_pred_data.shape[0]
		label1 = np.zeros(sample_num,dtype=np.int8)
		label1[id1] = 1
		print(len(label1),np.sum(label1>0),np.sum(label1==0))
		if len(annot_vec)==0:
			vec1 = ['ERCE(mapped)','Background']
		else:
			vec1 = annot_vec
		vec1 = np.asarray(vec1)
		t_label1 = vec1[1-label1]
		t_pred_data['label'] = t_label1
		t_pred_data = t_pred_data.sort_values(by=['label'])

		params = {'axes.labelsize': 20,
				'axes.titlesize': 26,
				'xtick.labelsize':20,
				'ytick.labelsize':20}
		pylab.rcParams.update(params)

		sel_id = 'Q2'
		fig = plt.figure(figsize=(25,12))
		plt.subplot(1,2,1)
		sns.boxplot(x='label', y=sel_id, data=t_pred_data, order=vec1, showfliers=True)
		plt.ylabel('Predicted attention')
		plt.xlabel('Label')

		plt.subplot(1,2,2)
		for t_label in vec1:
			b1 = np.where(t_label1==t_label)[0]
			# t_mtx = data_3[data_3['label']==t_label]
			t_mtx = t_pred_data.loc[b1,:]
			sns.distplot(t_mtx[sel_id], hist = True, kde = True,
						 kde_kws = {'shade':True, 'linewidth': 3},
						 label = t_label)
			plt.xlabel('Predicted attention')

		# type_id: 1, element not low DNase and DNase
		# type_id: 2, element not low DNase or DNase
		# annot1 = 'tf'
		output_filename = '%s/fig1/test1_%s_%d_%d.%d.%s.pdf'%(file_path2,celltype,run_id,flanking,type_id,annot1)
		plt.savefig(output_filename,dpi=300)

		t_pred_data_1 = t_pred_data
		output_filename = '%s/vbak2/test1_%s_%d_%d.%d.%s.2.txt'%(file_path2,celltype,run_id,flanking,type_id,annot1)
		t_pred_data_1.to_csv(output_filename,index=False,sep='\t')

# compare importance score
def compare_test_2(run_idList,celltype_id,filename1,annot1,typeid1=0,flanking=250,typeid2=0):

	# type_id = int(typeid1)
	type_id = int(typeid1)
	type_id2 = int(typeid2)

	# celltype_id = int(celltype_id)-1
	celltype_id = int(celltype_id)
	# celltype_vec = ['GM12878','K562','H1-hESC','HCT116','H9','HEK293','IMR-90','RPE-hTERT','U2OS']
	celltype_vec = ['GM12878','K562','H1-hESC','HCT116','H9','HEK293','IMR-90','RPE-hTERT','U2OS']
	celltype = celltype_vec[celltype_id]

	file_path = './'
	file_path1 = './data1'
	file_path2 = './data2'
	file_path2 = file_path1

	filename2 = '%s/%s.smooth.sorted.dnase.%d.bed'%(file_path1,celltype,flanking)
	region_data = pd.read_csv(filename1,header=None,sep='\t')
	annot_data = pd.read_csv(filename2,header=None,sep='\t')
	print(filename2,list(region_data),list(annot_data))

	region_label1 = np.asarray(region_data[3]) # element type
	region_label2 = np.asarray(region_data[7]) # DNase
	region_serial = np.asarray(region_data[4]) # serial
	region_num_1 = len(region_label2)
	print(region_label1[0:5],region_label2[0:5],region_serial[0:5])

	id1_ori = np.where((region_label2>0)|(region_label1!='Low-DNase'))[0]

	label_vec1 = ['CTCF-only,CTCF-bound','DNase-H3K4me3','DNase-H3K4me3,CTCF-bound','DNase-only',
					'PLS,CTCF-bound','dELS,CTCF-bound','pELS,CTCF-bound']
	region_label_vec1 = ['CTCF-only,CTCF-bound','DNase-H3K4me3','DNase-H3K4me3,CTCF-bound',
							'DNase-only','Low-DNase','PLS','PLS,CTCF-bound','dELS','dELS,CTCF-bound',
							'pELS','pELS,CTCF-bound']
	region_label_vec2 = [['CTCF-only,CTCF-bound','DNase-H3K4me3,CTCF-bound','PLS,CTCF-bound','dELS,CTCF-bound','pELS,CTCF-bound'],
							['DNase-H3K4me3','DNase-H3K4me3,CTCF-bound','DNase-only'],
							['DNase-H3K4me3','DNase-H3K4me3,CTCF-bound'],['PLS','PLS,CTCF-bound'],
							['dELS','dELS,CTCF-bound'],
							['pELS','pELS,CTCF-bound'],['Low-DNase']]

	region_label_vec3 = [['CTCF-only,CTCF-bound'],
							['DNase-H3K4me3','DNase-H3K4me3,CTCF-bound'],
							['PLS','PLS,CTCF-bound'],
							['dELS','dELS,CTCF-bound'],
							['pELS','pELS,CTCF-bound']]
	name_vec3 = ['CTCF-only','DNase-H3K4me3','PLS','dELS','pELS']
	t1 = []
	for t_region_label_vec in region_label_vec3:
		t1.extend(t_region_label_vec)
	region_label_vec3_1 = [t1]
	
	if type_id2==2:
		region_label_vec3 = region_label_vec3_1
		name_vec3 = ['group']
	num3 = len(region_label_vec3)

	list1 = []
	for t_label in label_vec1:
		print(t_label)
		b1 = np.where(region_label1==t_label)[0]
		list1.extend(region_serial[b1])
	serial_pre = np.unique(np.asarray(list1))
	print('serial_pre', len(serial_pre))

	region_data_ori = region_data.copy()
	region_label_vec = np.unique(region_label1)
	region_num2 = len(id1_ori)
	print(region_num_1,region_num2)
	print(region_label_vec)

	chrom = np.asarray(annot_data[0])
	b2 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]
	annot_data = annot_data.loc[b2,:]
	annot_data.reset_index(drop=True,inplace=True)

	chrom, annot_serial = np.asarray(annot_data[0]), np.asarray(annot_data[3])
	annot_label2 = np.asarray(annot_data[5]) # DNase
	b2 = np.where(annot_label2>0)[0]
	annot_serial_1_ori = annot_serial[b2]
	print('annot_serial_1_ori',len(annot_serial_1_ori),annot_label2[b2[0:5]])

	annot_serial_1 = np.union1d(annot_serial_1_ori,serial_pre)
	print('annot_serial_1',len(annot_serial_1),len(serial_pre))

	num1 = len(region_label_vec3)
	list2 = []
	for i in range(num1):
		list2.extend(region_label_vec3[i])

	region_sel_idvec = []
	for t_region_label in list2:
		print(t_region_label)
		id2 = np.where(region_label1==t_region_label)[0]
		region_sel_idvec.extend(id2)
	region_serial_sel = np.asarray(region_serial[region_sel_idvec])
	print('region_serial_sel',len(region_serial),len(region_serial_sel),len(region_serial_sel)/len(region_serial))

	for l in range(num3):
		t_region_label_vec = region_label_vec3[l]
		
		list_1 = []
		print(t_region_label_vec)
		for t_region_label in t_region_label_vec:
			print(t_region_label)
			id2 = np.where(region_label1==t_region_label)[0]
			id_1 = np.intersect1d(id1_ori,id2)
			list_1.extend(id_1)

		t_region_label = name_vec3[l]
		id_1 = np.asarray(list_1)

		region_data = region_data_ori.loc[id_1,:]
		region_data.reset_index(drop=True,inplace=True)
		t_region_num1 = len(id_1)
		print(t_region_label,region_num_1,t_region_num1,len(id2),len(id1_ori),
				t_region_num1/region_num_1,t_region_num1/region_num2,t_region_num1/len(id2))

		region_chrom, region_start, region_stop = np.asarray(region_data[0]), np.asarray(region_data[1]), np.asarray(region_data[2])
		region_num = len(region_chrom)
		t_region_serial = np.asarray(region_data[4])

		num1 = len(run_idList)
		for i in range(num1):
			run_id = run_idList[i]
			filename3 = '%s/test_vec2_%d_52_[5].1_0.1.txt'%(file_path,run_id)
			pred_data = pd.read_csv(filename3,sep='\t')
			serial_ori = np.asarray(pred_data['serial'])

			# serial of open chromation regions, mapped to estimation file
			id_1 = mapping_Idx(serial_ori,annot_serial_1)
			b1 = np.where(id_1>=0)[0]
			id_1 = id_1[b1]

			# serial of open chromatin regions, mapped to reference file
			id_2 = mapping_Idx(annot_serial,annot_serial_1)
			assert np.sum(id_2<0)==0

			# retain genomic loci with open chromatin regions
			t_pred_data = pred_data.loc[id_1,['chrom','start','stop','serial','predicted_attention','Q2']]
			t_pred_data.reset_index(drop=True,inplace=True)

			# area of open chromatin regions in each genomic locus
			t_pred_data['regionLabel'] = annot_label2[id_2[b1]]

			# serial of mapped genomic loci with open chromatin regions in estimation file
			serial = serial_ori[id_1]
			print(run_id,len(serial_ori),len(serial))

			chrom1, start1, stop1 = np.asarray(t_pred_data['chrom']), np.asarray(t_pred_data['start']), np.asarray(t_pred_data['stop'])
			t_serial1 = np.asarray(t_pred_data['serial'])
			id_map = mapping_Idx(t_serial1,t_region_serial) # map elements to geomic loci in estimation file
			b2 = np.where(id_map>=0)[0]
			id_map = id_map[b2]

			id_map_merge = mapping_Idx(t_serial1,region_serial_sel) # map union of elements to geomic loci in estimation file
			b3 = np.where(id_map_merge>=0)[0]
			id_map_merge = id_map_merge[b3]

			sample_num = t_pred_data.shape[0]
			label1 = np.zeros(sample_num,dtype=np.int8)
			label1[id_map_merge] = 2
			label1[id_map] = 1
			print(run_id,len(label1),np.sum(label1==1),np.sum(label1>0),np.sum(label1==0))
			vec1 = ['candidate CRE','background']
			vec1 = ['candidate CRE(other)','candidate CRE','background']
			vec1 = np.asarray(vec1)
			vec2 = vec1[1:3]

			t_label1 = vec1[2-label1]
			t_pred_data['label'] = t_label1
			t_pred_data['label1'] = label1
			t_pred_data = t_pred_data.sort_values(by=['label1'])
			label1_ori = label1.copy()
			
			for comp_id in [0]:
				t_pred_data1 = t_pred_data.copy()
				if comp_id==0:
					sel_id = np.where(label1<2)[0]
					t_pred_data1 = t_pred_data.loc[sel_id,:]
					t_pred_data1.reset_index(drop=True,inplace=True)
					t_label1 = t_label1[sel_id]
				else:
					label1 = (label1_ori==1)
					print(np.sum(label1==1),np.sum(label1==0),len(label1))
					t_label1 = vec2[1-label1]
					t_pred_data1['label'] = t_label1

				params = {'axes.labelsize': 20,
						'axes.titlesize': 26,
						'xtick.labelsize':20,
						'ytick.labelsize':20}
				pylab.rcParams.update(params)

				sel_id = 'Q2'
				fig = plt.figure(figsize=(25,12))
				plt.subplot(1,2,1)
				sns.boxplot(x='label', y=sel_id, data=t_pred_data1, order=vec2, showfliers=True)
				plt.ylabel('Predicted attention')
				plt.xlabel('Label')

				plt.subplot(1,2,2)
				t_label1 = np.asarray(t_pred_data1['label'])
				for t_label in vec2:
					b1 = np.where(t_label1==t_label)[0]
					# t_mtx = data_3[data_3['label']==t_label]
					print(t_label,len(b1))
					t_mtx = t_pred_data1.loc[b1,:]
					print('mean,median:',np.mean(t_mtx[sel_id]),np.median(t_mtx[sel_id]))
					sns.distplot(t_mtx[sel_id], hist = True, kde = True,
								 kde_kws = {'shade':True, 'linewidth': 3},
								 label = t_label)
					plt.xlabel('Predicted attention')

				# type_id: 1, element not low DNase and DNase
				# type_id: 2, element not low DNase or DNase
				# annot1 = 'tf'
				annot1 = t_region_label
				# group
				output_filename = '%s/fig1/test1_%s_%d_%d.%d.%d.%s.2.pdf'%(file_path2,celltype,run_id,flanking,type_id,comp_id,annot1)
				plt.savefig(output_filename,dpi=300)

			t_pred_data_1 = t_pred_data.loc[:,['chrom','start','stop','serial','label1',sel_id,'regionLabel']]
			# output_filename = '%s/test1_%s_%d_%d.%d.%s.1.txt'%(file_path2,celltype,run_id,flanking,type_id,annot1)
			# group
			output_filename = '%s/test1_%s_%d_%d.%d.%s.2.txt'%(file_path2,celltype,run_id,flanking,type_id,annot1)
			t_pred_data_1.to_csv(output_filename,index=False,sep='\t')

# compare with phyloP score
# input: filename1: phyloP score
#        filename2: RT estimation score
def compare_phyloP_sub1(filename1,run_idList,thresh=-10,output_filename='',config={}):
	
	filename_centromere = config['filename_centromere']
	file_path = config['file_path']
	feature_id, interval, sel_id = config['feature_id'], config['interval'], config['sel_id']

	if os.path.exists(filename1)==True:
		print("loading data...")
		data1 = np.load(filename1,allow_pickle=True)
		data_1 = data1[()]
		x_train1_trans_ori, train_sel_list_ori = np.asarray(data_1['x1']), np.asarray(data_1['idx'])

		print('train_sel_list',train_sel_list_ori.shape)
		print('x_train1_trans',x_train1_trans_ori.shape)

		serial_ori = train_sel_list_ori[:,1]
		dim1 = x_train1_trans_ori.shape[1]
	else:
		print('%s does not exist'%(filename1))
		return -1

	thresh1 = thresh
	if thresh1>0:
		thresh1 = np.quantile(signal_ori,thresh)
		print(np.max(signal_ori),np.min(signal_ori),thresh1)

	if thresh1>-10:
		id2 = np.where(signal_ori>thresh1)[0]
		serial_ori, x_train1_trans_ori = serial_ori[id2], x_train1_trans_ori[id2]

	data_ori = [serial_ori,x_train1_trans_ori]
	list1 = []
	num1 = len(run_idList)
	for l in range(num1):
		run_id1 = run_idList[l]
		for run_id in run_id1:
			filename2 = '%s/test_vec2_%d_52_[%d].1_0.1.txt'%(file_path,run_id,feature_id)
			data2 = compare_phyloP_sub2(data_ori,filename2,
										interval=interval,
										sel_id=sel_id,
										output_filename='',filename_centromere=filename_centromere)
			num1 = data2.shape[0]
			data2['celltype'] = [l]*num1
			data2['run_id'] = [run_id]*num1
			colnames = list(data2)
			data2 = data2.loc[:,['celltype','run_id','chrom','id']+colnames[0:-4]]
			list1.append(data2)

	data_2 = pd.concat(list1, axis=0, join='outer', ignore_index=True)
	data_2.to_csv(output_filename,index=False,sep='\t')

	return data_2

def compare_phyloP_sub2(data_ori,filename2,interval=0.05,sel_id='Q2',output_filename='',filename_centromere=''):

	serial_ori, x_train1_trans_ori = data_ori[0], data_ori[1]

	data2_1 = pd.read_csv(filename2,sep='\t')
	if filename_centromere!='':
		data2_1 = region_filter(data2_1,filename_centromere)

	serial_1 = np.asarray(data2_1['serial'])
	id1 = mapping_Idx(serial_ori,serial_1)
	b1 = np.where(id1>=0)[0]
	id1 = id1[b1]

	serial_ori, x_train1_trans_ori = serial_ori[id1], x_train1_trans_ori[id1]
	data2_1 = data2_1.loc[b1,:]
	data2_1.reset_index(drop=True,inplace=True)
	serial_1, predicted_attention = np.asarray(data2_1['serial']), np.asarray(data2_1[sel_id])
	chrom = np.asarray(data2_1['chrom'])

	phyloP_score = x_train1_trans_ori[:,range(2,21)]
	vec1 = phyloP_score[:,[17,15,16]]
	# phyloP_max, phyloP_min, phyloP_mean, phyloP_median = phyloP_score[:,15], phyloP_score[:,16], phyloP_score[:,17], phyloP_score[:,18]
	print(len(serial_ori),len(serial_1),phyloP_score.shape)

	# interval = 0.05
	num1 = int(1/interval)
	vec2 = np.zeros((num1,3))
	thresh_vec = 1-interval*np.arange(1,num1+1)
	thresh_vec = np.column_stack((thresh_vec,thresh_vec+interval))
	thresh_vec[0,1] += 1e-05

	dict1 = dict()
	dict2 = dict()
	chrom_vec = np.unique(chrom)
	chrom_num = len(chrom_vec)
	vec3 = np.zeros((num1,chrom_num,3))
	for l in range(num1):
		t1, t2 = thresh_vec[l]
		b1 = np.where((predicted_attention>=t1)&(predicted_attention<t2))[0]
		t_serial = serial_1[b1]
		if len(t_serial)==0:
			continue
		id1 = mapping_Idx(serial_ori,t_serial)
		vec2[l] = np.mean(vec1[id1],axis=0)
		for i in range(chrom_num):
			chrom_id = chrom_vec[i]
			b2 = np.where(chrom[b1]==chrom_id)[0]
			b_1 = b1[b2]
			t_serial = serial_1[b_1]
			if len(t_serial)==0:
				continue
			id1 = mapping_Idx(serial_ori,t_serial)
			vec3[l,i] = np.mean(vec1[id1],axis=0)
			
	vec3 = vec3.reshape((-1,3))
	vec2 = np.vstack((vec2,vec3))
	t1 = np.arange(chrom_num)+1
	t1 = np.repeat(t1,num1,axis=0)
	t2 = np.hstack(([-1]*num1,t1))
	num2 = len(t2)

	print(vec2.shape)
	data_2 = pd.DataFrame(columns=['max','mean','median'],data=vec2)
	data_2['id'] = np.ravel(np.tile(np.arange(num1),[1,chrom_num+1]))
	data_2['chrom'] = np.hstack(([-1]*num1,t1))
	if output_filename!='':
		data_2.to_csv(output_filename,index=False,sep='\t')

	return data_2

# compare with RT state
# input: filename_ori: original RT state estimation
#		 filename1: mapped RT state estimation
#        filename_list: list of RT estimation file and output filename
def compare_RT_sub1_ori(filename_ori,filename1,filename_list,thresh=-10,output_filename='',config={}):
	
	# filename_centromere = config['filename_centromere']
	# file_path = config['file_path']
	# feature_id, interval, sel_id = config['feature_id'], config['interval'], config['sel_id']

	data1_ori = pd.read_csv(filename_ori,header=None,sep='\t',
								names=['chrom','start','stop','serial','state','group','group1'])
	chrom_ori, serial_ori, state_ori = np.asarray(data1_ori['chrom']), np.asarray(data1_ori['serial']), np.asarray(data1_ori['state'])
	group_id_ori, group_id1_ori = np.asarray(data1_ori['group']), np.asarray(data1_ori['group1'])
	state_vec, group, group1 = np.unique(state_ori), np.unique(group_id_ori), np.unique(group_id1_ori)
	state_num, group_num, group_num1 = len(state_vec), len(group), len(group1)
	print(data1_ori.shape,state_num,group_num,group_num1)
	data1_ori_1 = data1_ori.copy()

	data1 = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1)
	col1, col2, col3, col4 = colnames[0], colnames[1], colnames[2], colnames[3]
	chrom = np.asarray(data1[col1])
	id1 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]
	data1 = data1.loc[id1,:]
	data1.reset_index(drop=True,inplace=True)
	chrom, start, stop, serial = np.asarray(data1[col1]), np.asarray(data1[col2]), np.asarray(data1[col3]), np.asarray(data1[col4])
	print(data1_ori.shape,data1.shape)

	num1 = len(filename_list)
	for i in range(num1):
		filename2, output_filename = filename_list[i]
		serial1, score_vec1 = compare_RT_sub2_ori(chrom,start,stop,serial,filename2)
		b1 = mapping_Idx(serial_ori,serial1)
		assert np.sum(b1<0)==0

		data1_ori = data1_ori_1.loc[b1,:]
		data1_ori.reset_index(drop=True,inplace=True)
		state, group_id, group_id1 = state_ori[b1], group_id_ori[b1], group_id1_ori[b1]

		data1_ori['score'] = score_vec1
		vec1 = np.zeros(state_num)
		for i in range(state_num):
			t_state = state_vec[i]
			b1 = np.where(state==t_state)[0]
			vec1[i] = np.mean(score_vec1[b1])

		vec2, vec3 = np.zeros(group_num), np.zeros(group_num1)
		for i in range(group_num):
			b1 = np.where(group_id==group[i])[0]
			vec2[i] = np.mean(score_vec1[b1])

		for i in range(group_num1):
			b1 = np.where(group_id1==group1[i])[0]
			vec3[i] = np.mean(score_vec1[b1])

		mean_value = np.column_stack((state_vec,vec1))
		print(filename2)
		print(mean_value)
		print(group,vec2)
		print(group1,vec3)
		data1_ori.to_csv(output_filename,index=False,sep='\t',float_format='%.7f')

	return data1_ori

# compare with RT state
# input:chrom,start,stop,serial: positions of mapped RT estimtion regions 
#		filename2: RT estimation score
def compare_RT_sub2_ori(chrom,start,stop,serial,filename2,sel_id='Q2',filename_centromere=''):

	data2 = pd.read_csv(filename2,sep='\t')
	if filename_centromere!='':
		data2 = region_filter(data2,filename_centromere)

	chrom2, start2, stop2, predicted_attention = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop']), np.asarray(data2[sel_id])
	chrom_vec, chrom_vec2 = np.unique(chrom), np.unique(chrom2)
	region_len = stop-start
	region_len2 = stop2-start2
	dict2 = dict()
	for chrom_id in chrom_vec2:
		dict2[chrom_id] = np.where(chrom2==chrom_id)[0]

	sample_num = len(chrom)
	score_vec1 = -100*np.ones(sample_num)
	for chrom_id in chrom_vec:
		b1 = np.where(chrom==chrom_id)[0]
		# dict1[chrom_id] = b1
		if not(chrom_id in chrom_vec2):
			continue
		b2 = dict2[chrom_id]
		print(chrom_id,len(b1),len(b2))
		for t_id in b1:
			t_chrom, t_start, t_stop = chrom[t_id], start[t_id], stop[t_id]
			id1 = np.where((start2[b2]<t_stop)&(stop2[b2]>t_start))[0]
			if len(id1)>0:
				overlap = []
				id2 = b2[id1]
				for t_id2 in id2:
					temp1 = np.min([t_stop-start2[t_id2],stop2[t_id2]-t_start,region_len[t_id],region_len2[t_id2]])
					overlap.append(temp1)
				id_1 = np.argmax(overlap)
				ratio = overlap/np.sum(overlap)
				score_vec1[t_id] = predicted_attention[id2[id_1]]
				# score_vec1[t_id] = np.dot(ratio,predicted_attention[id2])

	id1 = np.where(score_vec1>-100)[0]
	serial1 = serial[id1]
	score_vec1 = score_vec1[id1]

	return (serial1, score_vec1)

# compare with RT state
# input: filename_ori: original RT state estimation on genome 1
#		 filename1: mapped positions from genome 2 to genome 1
#        filename2: original RT signal
def compare_RT_sub1_pre1(filename_ori,filename1,filename2,thresh=-10,output_filename='',config={}):
	
	# filename_centromere = config['filename_centromere']
	# file_path = config['file_path']
	# feature_id, interval, sel_id = config['feature_id'], config['interval'], config['sel_id']

	data1_ori = pd.read_csv(filename_ori,header=None,sep='\t',
								names=['chrom','start','stop','serial','state','group','group1'])
	chrom_ori, serial_ori, state_ori = np.asarray(data1_ori['chrom']), np.asarray(data1_ori['serial']), np.asarray(data1_ori['state'])
	group_id_ori, group_id1_ori = np.asarray(data1_ori['group']), np.asarray(data1_ori['group1'])
	
	state_vec, group, group1 = np.unique(state_ori), np.unique(group_id_ori), np.unique(group_id1_ori)
	state_num, group_num, group_num1 = len(state_vec), len(group), len(group1)
	print(data1_ori.shape,state_num,group_num,group_num1)

	data1 = pd.read_csv(filename1,header=None,sep='\t')
	colnames = list(data1)
	col1, col2, col3, col4 = colnames[0], colnames[1], colnames[2], colnames[3]
	chrom = np.asarray(data1[col1])
	id1 = np.where((chrom!='chrX')&(chrom!='chrY')&(chrom!='chrM'))[0]
	data1 = data1.loc[id1,:]
	data1.reset_index(drop=True,inplace=True)
	chrom, start, stop, serial = np.asarray(data1[col1]), np.asarray(data1[col2]), np.asarray(data1[col3]), np.asarray(data1[col4])
	print(data1_ori.shape,data1.shape)

	data2 = pd.read_csv(filename2,header=None,sep='\t')
	colnames = list(data2)
	col1, col4, col5 = colnames[0], colnames[3], colnames[4]
	chrom_1, serial_1 = np.asarray(data2[col1]), np.asarray(data2[col4])
	signal_1 = np.asarray(data2[col5])
	sample_num = len(serial_1)
	state = -np.ones(sample_num,dtype=np.int32)
	group_id, group_id1 = np.asarray(['-10']*sample_num), np.asarray(['-10']*sample_num)

	# index in file1, index in file2
	id_vec2, id_vec1 = overlapping_with_regions_sub1(data1,data1_ori,tol=0,mode=1)
	print(len(id_vec2),len(id_vec1))
	print(id_vec2[0:10])
	print(id_vec1[0:10])
	mapped_serial = serial[id_vec2]
	b1 = mapping_Idx(mapped_serial,serial_1)
	b2 = np.where(b1>=0)[0]
	id1 = id_vec1[b1[b2]]
	state[b2] = state_ori[id1]
	group_id[b2] = group_id_ori[id1]
	group_id1[b2] = group_id1_ori[id1]

	fields = ['chrom','start','stop','serial','signal','domain','q1','q2','q_1','q_2','local_peak1','local_peak2',
					'state','group1','group2']
	if len(colnames)>=len(fields):
		data2 = data2.loc[:,colnames[0:-3]]
	data2['state'], data2['group'], data2['group1'] = state, group_id, group_id1

	data2.to_csv(filename2,index=False,header=False,sep='\t',float_format='%.7f')

	return data2

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
	# print(mean_value)
	# print(group,vec2)
	# print(group1,vec3)
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

# compare with rmsk
# calculate enrichment of each rmsk family in groups of genomic loci of different estimated attention
# input: filename1: file of overlapping between rmsk and genomic loci
# 		 filename2: genomic loci file
#		 filename2_1: RT estimation file
#        interval: the interval of estimated attention to group the genomic loci
#		 sel_id: the type of predicted attention
def compare_rmsk_sub2_pre1(run_id,filename1,filename2,filename2_1,thresh=-10,interval=0.05,
							output_filename='',sel_id='Q2',filename_centromere=''):
	
	data1 = pd.read_csv(filename1,sep='\t')
	serial = np.asarray(data1['serial'])

	data2 = pd.read_csv(filename2,header=None,sep='\t')
	serial_ori, signal_ori = np.asarray(data2[3]), np.asarray(data2[4])

	thresh1 = thresh
	if thresh>0:
		thresh1 = np.quantile(signal_ori,thresh)
		print(np.max(signal_ori),np.min(signal_ori),thresh1)

	if thresh1>-10:
		id2 = np.where(signal_ori>thresh1)[0]
		serial_ori, signal_ori = serial_ori[id2], signal_ori[id2]

	id1 = mapping_Idx(serial_ori,serial)
	# assert np.sum(id1<0)==0
	b1 = np.where(id1>=0)[0]
	print(len(b1),len(id1))
	data1 = data1.loc[b1,:]
	data1.reset_index(drop=True,inplace=True)
	print(data1.shape)

	data2_1 = pd.read_csv(filename2_1,sep='\t')
	if filename_centromere!='':
		data2_1 = region_filter(data2_1,filename_centromere)

	serial_1, predicted_attention = np.asarray(data2_1['serial']), np.asarray(data2_1[sel_id])
	id_1 = mapping_Idx(serial_ori,serial_1)
	# assert np.sum(id_1<0)==0
	b_1 = np.where(id_1>=0)[0]
	serial_1, predicted_attention = serial_1[b_1], predicted_attention[b_1]

	sample_num, sample_num1 = data2_1.shape[0], len(serial_1)
	print(sample_num1,sample_num,sample_num1/sample_num)

	label, overlap = np.asarray(data1['label']), np.asarray(data1['overlap'])
	serial = np.asarray(data1['serial'])
	label_vec = np.unique(label)
	label_num = len(label_vec)

	# interval = 0.05
	num1 = int(1/interval)
	vec2 = np.zeros((num1,label_num))
	thresh_vec = 1-interval*np.arange(1,num1+1)
	thresh_vec = np.column_stack((thresh_vec,thresh_vec+interval))
	thresh_vec[0,1] += 1e-05

	dict1 = dict()
	for l in range(num1):
		t1, t2 = thresh_vec[l]
		b1 = np.where((predicted_attention>=t1)&(predicted_attention<t2))[0]
		t_serial = serial_1[b1]
		dict1[l] = t_serial

	for i in range(label_num):
		id1 = np.where(label==label_vec[i])[0]
		serial1 = serial[id1]
		vec1 = overlap[id1]
		background = np.sum(vec1)/sample_num1

		cnt1 = np.zeros(num1)
		for l in range(num1):
			t_serial = dict1[l]

			# print(label_vec[i],sample_num1,sample_num,len(serial1),len(t_serial))
			if len(t_serial)==0:
				continue
			id2 = mapping_Idx(serial1,t_serial)
			id2 = id2[id2>=0]
			t_vec1 = vec1[id2]
			print(label_vec[i],thresh_vec[l],len(id2),len(t_serial))
			vec2[l,i] = np.sum(t_vec1)/(len(t_serial)*background)
			cnt1[l] = len(id2)/len(id1)
		print(np.sum(cnt1),cnt1)

	data_2 = pd.DataFrame(columns=label_vec,data=vec2)
	file_path = 'data1'
	if output_filename=='':
		output_filename = '%s/table2.%d.%s.thresh%s.txt'%(file_path,run_id,str(interval),str(thresh))

	data_2.to_csv(output_filename,index=False,sep='\t')

	return data_2

# compare with rmsk
# calculate enrichment of each rmsk family in groups of genomic loci of different estimated attention
# input: filename1: file of overlapping between rmsk and genomic loci
# 		 filename2: RT prediction file
#        interval: the interval of estimated attention to group the genomic loci
#		 sel_id: the type of predicted attention
def compare_rmsk_sub2_pre(filename1,filename2,interval=0.05,sel_id='Q2'):
	
	data1 = pd.read_csv(filename1,header=None,sep='\t')
	colnames1 = list(data1)
	serial1 = np.asarray(data1[colnames1[3]])
	overlap = np.asarray(data1[colnames1[4]])
	overlap_len = np.sum(overlap)

	data2 = pd.read_csv(filename2,sep='\t')
	colnames2 = list(data2)
	serial = np.asarray(data2['serial'])

	# sel_id = 'Q2'
	predicted_attention = np.asarray(data2[sel_id])
	serial_vec = np.unique(serial1)
	region_num = len(serial_vec)
	vec1 = np.zeros(region_num)
	for i in range(region_num):
		t_serial = serial_vec[i]
		b2 = (serial1==t_serial)
		vec1[i] = np.sum(overlap[b2])
		if i%1000==0:
			print(i)
			sys.stdout.flush()
		# if i>2000:
		# 	break

	id1 = mapping_Idx(serial,serial_vec)
	b1 = np.where(id1>=0)[0]
	id1 = id1[b1]
	serial_1 = serial_vec[b1]
	vec1 = vec1[b1]
	score_1 = predicted_attention[id1]

	# interval = 0.05
	num1 = int(1/interval)
	vec2 = np.zeros(num1)
	thresh_vec = 1-interval*np.arange(1,num1+1)
	thresh_vec = np.column_stack((thresh_vec,thresh_vec+interval))
	thresh_vec[0,1] += 1e-05

	background = np.sum(vec1)/len(serial)

	for i in range(num1):
		t1, t2 = thresh_vec[i]
		b1 = np.where((predicted_attention>=t1)&(predicted_attention<t2))[0]
		t_serial = serial[b1]
		id2 = mapping_Idx(serial_1,t_serial)
		# b2 = np.where(id2>=0)[0]
		id2 = id2[id2>=0]
		t_vec1 = vec1[id2]
		vec2[i] = np.sum(t_vec1)/(len(b1)*background)

	return (serial_1,score_1,vec1,vec2)

# compare with rmsk
# input: run_id: experiment
#		 filename1: rmsk file
#        filename2: RT prediction file
def compare_rmsk_sub2_ori(run_id,filename1,filename2,output_filename1='',output_filename2='',interval=0.05,sel_id='Q2'):

	data1 = pd.read_csv(filename1,sep='\t')
	repFamily = np.asarray(data1['repFamily'])
	repClass = np.asarray(data1['repClass'])

	num1 = len(repFamily)
	file_path = 'data1'
	list1, list2 = [], []
	list3_label, list3_1, list3_2, list3_3 = [], [], [], []
	for i in range(num1):
		t_repFamily, t_repClass = repFamily[i], repClass[i]
		if (t_repFamily.find('?')<0) and (t_repClass.find('?')<0):
			filename1 = '%s/%s.%s.overlap.txt'%(file_path,t_repFamily,t_repClass)
			print(t_repFamily,t_repClass)
			if os.path.exists(filename1)==False:
				continue
			serial_1,score_1,vec1,vec2 = compare_rmsk_sub2_pre(filename1,filename2,interval=interval,sel_id=sel_id)
			list1.append(t_repFamily)
			list2.append(vec2)
			list3_1.extend(serial_1)
			list3_2.extend(vec1)
			list3_3.extend(score_1)
			list3_label.extend([t_repFamily]*len(serial_1))

	fields = ['label','serial','overlap','score']
	data_1 = pd.DataFrame(columns=fields)
	data_1['label'] = np.asarray(list3_label)
	data_1['serial'], data_1['overlap'], data_1['score'] = np.asarray(list3_1), np.asarray(list3_2), np.asarray(list3_3)

	print(len(list1),list1)
	data_2 = pd.DataFrame(columns=list1,data=np.asarray(list2).T)

	if output_filename1=='':
		output_filename1 = '%s/table1.%d.%s.txt'%(file_path,run_id,str(interval))
	if output_filename2=='':
		output_filename2 = '%s/table2.%d.%s.txt'%(file_path,run_id,str(interval))

	data_1.to_csv(output_filename1,index=False,sep='\t')
	data_2.to_csv(output_filename2,index=False,sep='\t')

	return data_1, data_2

# compare with rmsk
# input: run_id: experiment
#		 filename1: rmsk file
#        filename2: RT prediction file
def compare_rmsk_sub2_basic(filename1,filename2,output_filename1=''):

	data1 = pd.read_csv(filename1,sep='\t')
	repFamily = np.asarray(data1['repFamily'])
	repClass = np.asarray(data1['repClass'])

	num1 = len(repFamily)
	file_path = 'data1'
	list1, list2 = [], []
	list3_label, list3_1, list3_2, list3_3 = [], [], [], []

	data2 = pd.read_csv(filename2,sep='\t')
	colnames2 = list(data2)
	serial = np.asarray(data2[colnames2[3]])

	for i in range(num1):
		t_repFamily, t_repClass = repFamily[i], repClass[i]
		if (t_repFamily.find('?')<0) and (t_repClass.find('?')<0):
			filename1 = '%s/%s.%s.overlap.txt'%(file_path,t_repFamily,t_repClass)
			print(t_repFamily,t_repClass)
			if os.path.exists(filename1)==False:
				continue

			data1 = pd.read_csv(filename1,header=None,sep='\t')
			colnames1 = list(data1)
			serial1 = np.asarray(data1[colnames1[3]])
			overlap = np.asarray(data1[colnames1[4]])
			overlap_len = np.sum(overlap)

			serial_vec = np.unique(serial1)
			region_num = len(serial_vec)
			vec1 = np.zeros(region_num)
			for l in range(region_num):
				t_serial = serial_vec[l]
				b2 = (serial1==t_serial)
				vec1[l] = np.sum(overlap[b2])
				if l%20000==0:
					print(l)
					sys.stdout.flush()

			id1 = mapping_Idx(serial,serial_vec)
			b1 = np.where(id1>=0)[0]
			id1 = id1[b1]
			serial_1 = serial_vec[b1]
			vec1 = vec1[b1]

			list3_1.extend(serial_1)
			list3_2.extend(vec1)
			list3_label.extend([t_repFamily]*len(serial_1))

	fields = ['label','serial','overlap']
	data_1 = pd.DataFrame(columns=fields)
	data_1['label'] = np.asarray(list3_label)
	data_1['serial'], data_1['overlap'] = np.asarray(list3_1), np.asarray(list3_2)

	# print(len(list1),list1)
	# data_2 = pd.DataFrame(columns=list1,data=np.asarray(list2).T)

	if output_filename1=='':
		output_filename1 = '%s/table1.rmsk.txt'%(file_path)
	
	data_1.to_csv(output_filename1,index=False,sep='\t')

	return data_1

# compare with rmsk
# input: run_id: experiment
#		 filename1: rmsk file
#        filename2: RT prediction file
def compare_rmsk_sub2(filename1,filename2,run_idlist,filename_list_est,output_filename='',
						thresh=-10,interval=0.05,sel_id='Q2',config={}):

	num1 = len(filename_list)
	filename_centromere = config['filename_centromere']

	for i in range(num1):
		run_id, filename2_1 = run_idlist[i], filename_list_est[i]
		data_2 = compare_rmsk_sub2_pre1(run_id,filename1,filename2,filename2_1,thresh=thresh,interval=interval,
								output_filename=output_filename,sel_id='Q2',filename_centromere=filename_centromere)

	return True

def compare_rmsk_sub2_plot(filename,data1=[],output_filename='test1.pdf'):

	if len(data1)==0:
		data1 = pd.read_csv(filename,sep='\t')

	figure = plt.figure(figsize=(32,28))

	plt.title('Estimated importance score')
	# sns.violinplot(x='label', y=sel_id, data=data_3)
	sel_id = 'score'
	sns.boxplot(x='label', y=sel_id, data=data1, showfliers=True)
	# sns.boxplot(x='label', y=sel_id, data=data_3, showfliers=True)

	# ax.get_xaxis().set_ticks([])
	# ax = plt.gca()
	# ax.get_xaxis().set_visible(False)
	# ax.xaxis.label.set_visible(False)
	plt.show()
	plt.savefig(output_filename,dpi=300)

# compare with rmsk
# weighted average of scores for each rmsk family
# weight is based on enrichment in each genomic locus
# input: filename: enrichment of each rmsk in each genomic locus
def compare_rmsk_sub2_1(filename,data1=[],output_filename=''):

	data1 = pd.read_csv(filename,sep='\t')
	label, overlap, score = np.asarray(data1['label']), np.asarray(data1['overlap']), np.asarray(data1['score'])
	label_vec = np.unique(label)
	label_num = len(label_vec)
	vec1 = np.zeros(label_num)

	for i in range(label_num):
		t_label = label_vec[i]
		b1 = np.where(label==t_label)[0]
		t1 = overlap[b1]
		weight = t1/np.sum(t1)
		t_score = np.sum(weight*score[b1])
		vec1[i] = t_score
		print(t_label,t_score,np.max(weight),np.min(weight))

	data2 = pd.DataFrame(columns=['label','score'])
	data2['label'] = label_vec
	data2['score'] = vec1
	data2 = data2.sort_values(by=['score'])

	if output_filename=='':
		output_filename = 'test1_element.txt'
	data2.to_csv(output_filename,index=False,sep='\t')

	return data2

# compare with rmsk
# calculate enrichment of each rmsk family in groups of genomic loci of different estimated attention
# input: filename1: file of overlapping between rmsk and genomic loci
#		 filename2: file of original RT data
#        filename2_1: file of processed RT data
#        interval: the interval of estimated attention to group the genomic loci
def compare_rmsk_sub2_2(filename1,filename2,filename2_1,thresh=0,interval=0.05,output_filename='',sel_id='Q2'):

	data1 = pd.read_csv(filename1,sep='\t')
	serial = np.asarray(data1['serial'])
	print(data1.shape)

	data2 = pd.read_csv(filename2,header=None,sep='\t')
	serial_ori, signal_ori = np.asarray(data2[3]), np.asarray(data2[4])
	id1 = mapping_Idx(serial_ori,serial)
	assert np.sum(id1<0)==0
	thresh1 = thresh
	id2 = np.where(signal_ori[id1]>thresh1)[0]
	data1 = data1.loc[id2,:]
	data1.reset_index(drop=True,inplace=True)
	print(data1.shape)

	data2_1 = pd.read_csv(filename2_1,sep='\t')
	serial_1, predicted_attention = np.asarray(data2_1['serial']), np.asarray(data2_1[sel_id])
	id1 = mapping_Idx(serial_ori,serial_1)
	assert np.sum(id1<0)==0
	id2 = np.where(signal_ori[id1]>thresh1)[0]
	serial_1, predicted_attention = serial_1[id2], predicted_attention[id2]
	sample_num, sample_num1 = data2_1.shape[0], len(serial_1)
	print(sample_num1,sample_num,sample_num1/sample_num)

	label, overlap, score = np.asarray(data1['label']), np.asarray(data1['overlap']), np.asarray(data1['score'])
	serial = np.asarray(data1['serial'])
	label_vec = np.unique(label)
	label_num = len(label_vec)
	vec1 = np.zeros(label_num)
	
	# interval = 0.05
	num1 = int(1/interval)
	vec2 = np.zeros((num1,label_num))
	thresh_vec = 1-interval*np.arange(1,num1+1)
	thresh_vec = np.column_stack((thresh_vec,thresh_vec+interval))
	thresh_vec[0,1] += 1e-05

	mtx_1 = np.zeros((label_num,num1,3))
	for l in range(num1):
		t1, t2 = thresh_vec[l]
		b1 = np.where((predicted_attention>=t1)&(predicted_attention<t2))[0]
		t_serial = serial_1[b1]

		for i in range(label_num):
			id1 = np.where(label==label_vec[i])[0]
			serial1 = serial[id1]
			vec1 = overlap[id1]
			background = np.sum(vec1)/sample_num1

			id2 = mapping_Idx(serial1,t_serial)
			id2 = id2[id2>=0]
			t_vec1 = vec1[id2]
			# print(label_vec[i],len(id2),len(t_serial))
			vec2[l,i] = np.sum(t_vec1)/(len(t_serial)*background)
			mtx_1[i,l] = [np.sum(t_vec1),len(t_serial),np.sum(vec1)]

	data_2 = pd.DataFrame(columns=label_vec,data=vec2)
	if output_filename=='':
		output_filename = 'table2.%d.%s.thresh%s.txt'%(run_id,str(interval),str(thresh))

	data_2.to_csv(output_filename,index=False,sep='\t')

	b = filename1.find('.txt')
	output_filename = '%s.RE_ori.npy'%(filename1[0:b])
	np.save(output_filename,mtx_1,allow_pickle=True)

	sel_num = 6
	list1, list2 = [], []
	list_1, list_2 = [], []
	for i in range(label_num):
		t1 = mtx_1[i]
		s1 = t1[0,2]
		temp1 = np.sum(t1[0:sel_num,0:2],axis=0)
		t_overlap, t_num = temp1[0], temp1[1]
		t_overlap1, t_num1 = s1-t_overlap, sample_num1-t_num
		t_expectation = np.asarray([s1*t_num/sample_num1,s1*t_num1/sample_num1])
		unit = 1.0
		stats, pvalue = chisquare([t_overlap/unit,t_overlap1/unit], f_exp=t_expectation/unit)
		print(label_vec[i],stats,pvalue)
		list1.append(stats)
		list2.append(pvalue)

		t_overlap_1, t_num_1 = t1[:,0], t1[:,1]
		t_expectation1 = s1*t_num_1/sample_num1
		stats_1, pvalue_1 = chisquare(t_overlap_1/unit, f_exp=t_expectation1/unit)
		print(label_vec[i],stats_1,pvalue_1)
		list_1.append(stats_1)
		list_2.append(pvalue_1)

	data_3 = pd.DataFrame(columns=['label','stats','pvalue','stats1','pvalue1'])
	data_3['label'] = label_vec
	data_3['stats'], data_3['pvalue'] = list1, list2
	data_3['stats1'], data_3['pvalue1'] = list_1, list_2

	b = filename1.find('.txt')
	output_filename = '%s.pvalue.txt'%(filename1[0:b])
	data_3.to_csv(output_filename,index=False,sep='\t')

	return data_2

def chromatin_accessibility():
	x = 1

# using function detecting outliers:
# https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting
def is_outlier(points, thresh=3.5):
	"""
	Returns a boolean array with True if points are outliers and False 
	otherwise.

	Parameters:
	-----------
		points : An numobservations by numdimensions array of observations
		thresh : The modified z-score to use as a threshold. Observations with
			a modified z-score (based on the median absolute deviation) greater
			than this value will be classified as outliers.

	Returns:
	--------
		mask : A numobservations-length boolean array.

	References:
	----------
		Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
		Handle Outliers", The ASQC Basic References in Quality Control:
		Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
	"""
	if len(points.shape) == 1:
		points = points[:,None]
	median = np.median(points, axis=0)
	diff = np.sum((points - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation

	return modified_z_score > thresh


class DataGenerator1(keras.utils.Sequence):

	def __init__(self, list_IDs, labels, file_path, batch_size=32, dim=(101,5000,4), 
				 shuffle=True):
		'Initialization'
		self.file_path = file_path
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		# self.context_size = dim[0]
		self.n_step = dim[0]
		self.region_unit_size = dim[1]
		self.n_channels = dim[2]
		self.shuffle = shuffle
		self.counter = 0
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		start = time.time()
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
		stop = time.time()
		print('shuffle',stop-start)
		print('epoch',stop-self.counter)
		self.counter = stop

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, self.n_step, self.region_unit_size, self.n_channels),dtype=np.float32)
		y = np.empty((self.batch_size, self.n_step, 1))
		num1 = len(list_IDs_temp)

		start = time.time()
		# Generate data
		# for i, ID in enumerate(list_IDs_temp):
		for i in range(num1):
			# Store sample
			# X[i] = np.load(ID,allow_pickle=True)
			label_id,label_serial,t_filename,local_id = list_IDs_temp[i]
			# id1, t_filename = ID
			t_filename1 = '%s/%s'%(self.file_path,t_filename)
			with h5py.File(t_filename1,'r') as fid:
				# serial2 = fid["serial"][:]
				t_mtx = fid["vec"][:]
				#print(t_mtx.shape)
				#print(local_id)
				X[i] = t_mtx[local_id-1]
				# feature_mtx = feature_mtx[:,0:kmer_dim_ori]
			# Store class
			y[i] = self.labels[label_id]

		stop = time.time()
		print('Batch size: %d samples'%(X.shape[0]),stop-start)
		print(X.shape,y.shape)
		return X, y

class DataGenerator2(keras.utils.Sequence):

	def __init__(self, list_IDs, labels, train_data, file_path, batch_size=32, dim=(101,5000,4), 
				 shuffle=True):
		'Initialization'
		self.file_path = file_path
		self.batch_size = batch_size
		self.train_data = train_data
		self.labels = labels
		self.list_IDs = list_IDs
		# self.context_size = dim[0]
		self.n_step = dim[0]
		self.region_unit_size = dim[1]
		self.n_channels = dim[2]
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		# X = np.empty((self.batch_size, self.n_step, self.region_unit_size, self.n_channels),dtype=np.float32)
		# y = np.empty((self.batch_size, self.n_step, 1))
		# num1 = len(list_IDs_temp)

		# Generate data
		# for i, ID in enumerate(list_IDs_temp):
		# for i in range(num1):
		# 	# Store sample
		# 	# X[i] = np.load(ID,allow_pickle=True)
		# 	# label_id,label_serial,t_filename,local_id = list_IDs_temp[i]
		# 	label_id = list_IDs_temp[i]
		# 	# id1, t_filename = ID
		# 	# t_filename1 = '%s/%s'%(self.file_path,t_filename)
		# 	# with h5py.File(t_filename1,'r') as fid:
		# 	# 	# serial2 = fid["serial"][:]
		# 	# 	t_mtx = fid["vec"][:]
		# 	# 	#print(t_mtx.shape)
		# 	# 	#print(local_id)
		# 	# 	X[i] = t_mtx[local_id-1]
		# 	# 	# feature_mtx = feature_mtx[:,0:kmer_dim_ori]
		# 	# Store class
		# 	X[i] = self.train_data[label_id]
		# 	y[i] = self.labels[label_id]
		start = time.time()
		X = self.train_data[list_IDs_temp]
		y = self.labels[list_IDs_temp]
		stop = time.time()
		print('Batch size: %d samples 2'%(X.shape[0]),stop-start)
		print(X.shape,y.shape)

		return X, y

class DataGenerator3(keras.utils.Sequence):

	def __init__(self, list_IDs, labels, train_data, batch_size=32, dim=(101,5000,4), 
				 shuffle=True):
		'Initialization'
		self.batch_size = batch_size
		self.train_data = train_data
		self.labels = labels
		self.list_IDs = list_IDs
		# self.context_size = dim[0]
		self.n_step = dim[0]
		self.region_unit_size = dim[1]
		self.n_channels = dim[2]
		self.shuffle = shuffle
		self.indexes = np.arange(len(self.list_IDs))
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		start = time.time()
		# Find list of IDs
		# list_IDs_temp = [self.list_IDs[k] for k in indexes]
		sel_id = self.list_IDs[indexes]

		# Generate data
		# X, y = self.__data_generation(list_IDs_temp)

		X = self.train_data[sel_id]
		y = self.labels[indexes]
		stop = time.time()
		print('Batch size:', X.shape, y.shape, stop-start)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
			print('indexes shuffle')

	# def __data_generation(self, list_IDs_temp):
	# 	'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		# X = np.empty((self.batch_size, self.n_step, self.region_unit_size, self.n_channels),dtype=np.float32)
		# y = np.empty((self.batch_size, self.n_step, 1))
		# num1 = len(list_IDs_temp)

		# Generate data
		# for i, ID in enumerate(list_IDs_temp):
		# for i in range(num1):
		# 	# Store sample
		# 	# X[i] = np.load(ID,allow_pickle=True)
		# 	# label_id,label_serial,t_filename,local_id = list_IDs_temp[i]
		# 	label_id = list_IDs_temp[i]
		# 	# id1, t_filename = ID
		# 	# t_filename1 = '%s/%s'%(self.file_path,t_filename)
		# 	# with h5py.File(t_filename1,'r') as fid:
		# 	# 	# serial2 = fid["serial"][:]
		# 	# 	t_mtx = fid["vec"][:]
		# 	# 	#print(t_mtx.shape)
		# 	# 	#print(local_id)
		# 	# 	X[i] = t_mtx[local_id-1]
		# 	# 	# feature_mtx = feature_mtx[:,0:kmer_dim_ori]
		# 	# Store class
		# 	X[i] = self.train_data[label_id]
		# 	y[i] = self.labels[label_id]
		# start = time.time()
		# X = self.train_data[list_IDs_temp]
		# y = self.labels[list_IDs_temp]
		# stop = time.time()
		# print('Batch size: %d samples 3'%(X.shape[0]),stop-start)
		# print(X.shape,y.shape)

		# return X, y

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

def compare1(run_idList,filename_list=[],output_filename='',thresh=0.05):

	num1 = len(run_idList)
	print(num1)
	cnt1 = 100
	vec1 = []
	sel_num = 2
	sel_num1 = 2*sel_num
	file_path = 'data1'
	method1 = 52
	feature_id1 = -1
	annot1 = 'init1'

	for i in range(num1):
		run_id = run_idList[i]

		if len(filename_list)>0:
			filename = filename_list[i]
		else:
			filename = '%s/test_vec2_%d_%d_[%d].1_0.%s.txt'%(file_path,run_id,method1,feature_id1,annot1)
		
		if os.path.exists(filename)==False:
			print(filename)
			continue

		data1 = pd.read_csv(filename,sep='\t')
		colnames = list(data1)
		chrom, start = data1[colnames[0]], data1[colnames[1]]
		score3, score4 = np.asarray(data1['score3']), np.asarray(data1['score4'])
		empirical1, empirical2 = np.asarray(data1['empirical1']), np.asarray(data1['empirical2'])
		region_num = data1.shape[0]
		empirical1[empirical1<0] = 2
		empirical2[empirical2<0] = 2
		print('region_num',region_num,num1)

		if i==0:
			mask = np.zeros((region_num,num1),dtype=np.int8)
			mtx1 = np.zeros((region_num,num1,sel_num1))
			mtx2 = np.zeros((region_num,sel_num1))

		b1 = np.where((empirical1<thresh)|(empirical2<thresh))[0]
		b1_1 = np.where((empirical1<thresh)&(empirical2<thresh))[0]
		b_1 = np.where(empirical1<thresh)[0]
		b_2 = np.where(empirical2<thresh)[0]
		mask[b1,i] = 1
		mtx1[:,i,0], mtx1[:,i,1] = score3, score4
		mtx1[:,i,2], mtx1[:,i,3] = empirical1, empirical2
		
		num2 = len(b1)
		t1 = [run_id,num2,len(b1_1),len(b_1),len(b_2)]
		vec1.append(t1)
		print(t1)
		
		if num2>cnt1:
			cnt1 = num2
			data_1 = data1.copy()

	print(mask.shape)
	s1 = np.sum(mask,axis=1)
	b1 = np.where(s1>0)[0]
	num2 = len(b1)
	ratio1 = num2/region_num
	print(num2,region_num,ratio1)
	# data_1 = data_1.loc[b1,colnames]
	data_1['num1'] = s1

	for i in range(sel_num):
		mtx2[:,i] = np.max(mtx1[:,:,i],axis=1)
	for i in range(sel_num,sel_num1):
		mtx2[:,i] = np.min(mtx1[:,:,i],axis=1)

	fields = ['max','mean','empirical1.1','empirical2.1']
	for i in range(sel_num1):
		data_1[fields[i]] = mtx2[:,i]

	if output_filename=='':
		output_filename = 'test_%s.1.txt'%(thresh)
		output_filename2 = 'test_%s.2.txt'%(thresh)

	data_1.to_csv(output_filename,index=False,sep='\t')
	vec1 = np.asarray(vec1)
	np.savetxt(output_filename2,vec1,fmt='%d',delimiter='\t')

	return data_1, vec1

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
		# vec3_pre_1 = []
		# for t_value1 in vec3_pre:
		# 	t_value2 = []
		# 	for t_list1 in t_value1:
		# 		t_value2.append(np.sort(list(set(chrom_vec)-set(t_list1))))
		# 	vec3_pre_1.append(t_value2)

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
	# chrom_vec = range(20,22)
	# chrom_vec = [1,9,10]
	# chrom_vec = range(1,6)
	num1 = len(id_vec)

	print("processes")
	start = time.time()
	# processes = [mp.Process(target=self._compute_posteriors_graph_test, args=(len_vec, X, region_id,self.posteriors_test,self.posteriors_test1,self.queue)) for region_id in range(0,num_region)]
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

	# train_sel_list_ori = np.loadtxt('serial_encode1.txt',delimiter='\t')
	kmer_dict1 = kmer_dict(kmer_size)
	f_list, f_mtx, serial = load_seq_2(filename1, kmer_size, kmer_dict1, sel_idx=[])

	return f_list, serial

# load kmer frequency feature
def prep_data_sequence_kmer_chrom_ori(filename1,filename2,kmer_size,chrom_vec=[],filename_prefix='',save_mode=1,pos={}):

	# train_sel_list_ori = np.loadtxt('serial_encode1.txt',delimiter='\t')
	kmer_dict1 = kmer_dict(kmer_size)
	file1 = pd.read_csv(filename1,sep='\t')
	seq1 = np.asarray(file1['seq'])
	serial1 = np.asarray(file1['serial'])

	if len(pos)==0:
		file2 = pd.read_csv(filename2,header=None,sep='\t')
		chrom, start, stop, ref_serial = np.asarray(file2[0]), np.asarray(file2[1]), np.asarray(file2[2]), np.asarray(file2[3])

	else:
		chrom, start, stop, ref_serial = np.asarray(pos['chrom']), np.asarray(pos['start']), np.asarray(pos['stop']), np.asarray(pos['serial'])

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
		t_serial = ref_serial[sel_idx]
		id1 = mapping_Idx(serial1,t_serial)
		b1 = (id1>=0)
		n1,n2 = np.sum(b1), len(t_serial)
		if n1!=n2:
			print('error!',chrom_id,n1,n2)
		sel_idx = id1[b1]
		# f_list, chrom_id = load_seq_2_kmer1(seq1,serial1,kmer_size,kmer_dict1,chrom_id=chrom_id,sel_idx=sel_idx)
		list1.append(sel_idx)

	queue1 = mp.Queue()
	# chrom_vec = range(20,22)
	# chrom_vec = [1,9,10]
	# chrom_vec = range(1,6)

	print("processes")
	start = time.time()
	# processes = [mp.Process(target=self._compute_posteriors_graph_test, args=(len_vec, X, region_id,self.posteriors_test,self.posteriors_test1,self.queue)) for region_id in range(0,num_region)]
	processes = [mp.Process(target=load_seq_2_kmer1, 
				args=(seq1, serial1, kmer_size, kmer_dict1, chrom_vec[i], list1[i], queue1, 
						filename_prefix, save_mode)) for i in range(chrom_num)]

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

	feature_dim = len(kmer_dict1)
	num_region = len(serial1)
	f_list = np.zeros((num_region,feature_dim))

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

	# output_filename = '%s_kmer%d.h5'(filename_prefix,kmer_size)
	# with h5py.File(output_filename,'w') as fid:
	# 	fid.create_dataset("serial", data=ref_serial, compression="gzip")
	# 	fid.create_dataset("vec", data=f_list, compression="gzip")

	return f_list, serial1

# load kmer frequency feature
def prep_data_sequence_kmer_chrom(filename1,filename2,kmer_size,chrom_vec=[],filename_prefix='',save_mode=1,region_size=1):

	# train_sel_list_ori = np.loadtxt('serial_encode1.txt',delimiter='\t')
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

# load kmer frequency feature
def prep_data_sequence_kmer_chrom_1(filename1,filename2,kmer_size,chrom_vec=[],filename_prefix='',save_mode=1):

	# train_sel_list_ori = np.loadtxt('serial_encode1.txt',delimiter='\t')
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
	for chrom_id in chrom_vec:
		sel_idx = np.where(chrom==chrom_id)[0]
		t_serial = ref_serial[sel_idx]
		id1 = mapping_Idx(serial1,t_serial)
		b1 = (id1>=0)
		n1,n2 = np.sum(b1), len(t_serial)
		if n1!=n2:
			print('error!',chrom_id,n1,n2)
		sel_idx = id1[b1]
		# f_list, chrom_id = load_seq_2_kmer1(seq1,serial1,kmer_size,kmer_dict1,chrom_id=chrom_id,sel_idx=sel_idx)
		list1.append(sel_idx)

	queue1 = mp.Queue()
	# chrom_vec = range(20,22)
	# chrom_vec = [1,9,10]
	# chrom_vec = range(1,6)

	print("processes")
	start = time.time()
	# processes = [mp.Process(target=self._compute_posteriors_graph_test, args=(len_vec, X, region_id,self.posteriors_test,self.posteriors_test1,self.queue)) for region_id in range(0,num_region)]
	processes = [mp.Process(target=load_seq_2_kmer1, 
				args=(seq1, serial1, kmer_size, kmer_dict1, chrom_vec[i], list1[i], queue1, 
						filename_prefix, save_mode)) for i in range(chrom_num)]

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

	feature_dim = len(kmer_dict1)
	num_region = len(serial1)
	f_list = np.zeros((num_region,feature_dim))

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
		file_path = 'data1'

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
			# filename1 = '%s/feature_transform_%d_%d.txt'%(file_path,run_id1,method1)
			# filename2 = '%s/feature_transform_%d_%d.txt'%(file_path,run_id2,method1)
			# output_filename = '%s/test_vec2_%d_%d_[%d].%d_%d.txt'%(file_path,run_id1,method1,feature_id1,type_id,type_id1)
			# # output_filename1 = '%s/temp2_%d_%d.%d_%d.2.txt'%(file_path,run_idlist[0][0],run_idlist[0][1],type_id,type_id1)

			# filename1 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id1,method1)
			# filename2 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id2,method1)
			filename1 = '%s/data_3/feature_transform_%d_%d.1.txt'%(file_path,run_id1,method1)
			filename2 = '%s/data_3/feature_transform_%d_%d.1.txt'%(file_path,run_id2,method1)
			output_filename = '%s/test_vec2_%d_%d_[%d].%d_%d.1.txt'%(file_path,run_id1,method1,feature_id1,type_id,type_id1)
			# output_filename1 = '%s/temp2_%d_%d.%d_%d.2.txt'%(file_path,run_idlist[0][0],run_idlist[0][1],type_id,type_id1)
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

# merge files
def run_2_merge(run_idlist,config):
	file_path = './mnt/yy3'

	type_id, type_id1 = config['type_id_1'], config['type_id1_1']
	# feature_id1 = config['feature_id1']

	filename_list, output_filename_list = [], []
	num1 = len(run_idlist)

	for pair1 in run_idlist:
		run_id1, feature_id1, method1 = pair1
		if 'filename_list1' in config:
			print('filename_list1',config['filename_list1'])
			filename_list_1 = config['filename_list1'][run_id1]
			filename1, output_filename = filename_list_1[0], filename_list_1[1]
		else:
			filename1 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id1,method1)
			output_filename = '%s/test_vec2_%d_%d_[%d].%d_%d.1.txt'%(file_path,run_id1,method1,feature_id1,type_id,type_id1)
			# output_filename1 = '%s/temp2_%d_%d.%d_%d.2.txt'%(file_path,run_idlist[0][0],run_idlist[0][1],type_id,type_id1)
		
		output_filename_list.append(output_filename)
		data2 = select_region2_sub(filename1)
		data2.to_csv(output_filename,index=False,sep='\t')
		print(output_filename,data2.shape)

	return output_filename_list

def run_1_sub1(t_list1,filename2,method_id1,feature_id1,filename_vec,thresh_1=0,tol=2,sample_num=1000):
	file_path = './mnt/yy3'

	# if t_list1==[]:
	# 	# run_id = 8100
	# 	# t_list1 = range(8706,8726,4)
	# 	# t_list1 = range(8750,8770,4)
	# 	t_list1 = range(8938,8946,4)
	# 	t_list1 = range(8946,8950,4)
	# 	# t_list1 = range(8100,8108,2)
	region_filename, output_filename, annot1, annot2 = filename_vec

	for run_id in t_list1:
		print(run_id)
		filename1 = '%s/test_vec2_%d_%d_[%d].1_0.1.txt'%(file_path,run_id,method_id1,feature_id1)
		# filename2 = 'early_peaks_H1_Jan.txt'
		# filename2 = 'initiation_zone_HCT_v1.txt'
		# filename2 = 'ERCE.mm10.bed'
		# output_filename = '%s/test_vec2_%d_%d_[%d].1_0.init.txt'%(file_path,run_id,method_id1,feature_id1)
		# output_filename1 = '%s/test_vec2_%d_%d_[%d].1_0.init1.txt'%(file_path,run_id,method_id1,feature_id1)
		# output_filename2 = '%s/test_vec2_%d_%d_[%d].1_0.init2.txt'%(file_path,run_id,method_id1,feature_id1)

		output_filename1 = '%s/test_vec2_%d_%d_[%d].%s.txt'%(file_path,run_id,method_id1,feature_id1,annot1)
		output_filename2 = '%s/test_vec2_%d_%d_[%d].%s.txt'%(file_path,run_id,method_id1,feature_id1,annot2)

		# original tolerance: 1
		# compare_with_regions_random1(filename1,filename2,output_filename,output_filename1,
		# 							output_filename2, tol=tol1, sample_num=sample_num1)

		# region_filename = 'region_test_1.txt'
		# output_filename = 'region_test_2.txt'
		compare_with_regions_random2(filename1,filename2,output_filename,output_filename1,
							output_filename2, region_filename, thresh_1=thresh_1, tol=tol, sample_num=sample_num, type_id=1)

def run_1_1():

	t_list1 = range(8938,8946,4)
	# 	t_list1 = range(8946,8950,4)
	filename2 = 'ERCE.mm10.bed'
	method_id1, feature_id1 = 52, -1
	filename_vec = ['region_test_1.txt','region_test_2.txt','1_0.init1','1_0.init2.2']
	run_1_sub1(t_list1,filename2,method_id1,feature_id1,filename_vec)

	t_list1 = range(8100,8107,2)
	filename2 = 'early_peaks_H1_Jan.txt'
	method_id1, feature_id1 = 52, 5
	run_1_sub1(t_list1,filename2,method_id1,feature_id1,filename_vec)

	t_list1 = range(8108,8115,2)
	filename2 = 'initiation_zone_HCT_v1.txt'
	method_id1, feature_id1 = 52, 5
	run_1_sub1(t_list1,filename2,method_id1,feature_id1,filename_vec)

def run_1_2(t_list1_pre,method1,feature_id1,cell_idtype,run_idlist=[],output_filename='table1.txt',mode=1):

	# t_list1 = list(range(8706,8724,4)) + list(range(8930,8948,4))
	# t_list1_1 = list(range(8707,8724,4)) + list(range(8931,8948,4))

	t_list1, t_list2 = [], []
	# t_list1_pre = list(range(8850,8922,4)) + list(range(8771,8850,4))
	# t_list1_pre = list(range(8100,8127,2)) + list(range(8200,8231,2)) + list(range(8500,8563,2)) \
	# 			 	+ list(range(9100,9131,2)) + list(range(9500,9579,2)) \
	# 			 	+ list(range(9700,9747,2)) + list(range(9927,9938,2))

	# t_list1_pre = list(range(8300,8347,2))
	# t_list1_pre = list(range(10508,10512,4))+list(range(10509,10512,4))
	# t_list1_pre = list(range(8090,8098,2))

	# cell_idtype = 1
	if cell_idtype == 0:
		t_list2_pre = [(i+2) for i in t_list1_pre]
		# t_list2_pre[-1] = t_list1_pre[-1]+2+5000
		# t_list2_pre = [(i+2) for i in t_list1_pre]
		# t_list2_pre[-1] = t_list1_pre[-1]+2+5000
	else:
		t_list2_pre = [(i+1) for i in t_list1_pre]
	
	t_list1 = t_list1+t_list1_pre
	t_list2 = t_list2+t_list2_pre
	# t_list1_pre = list(range(8851,8922,4)) + list(range(8772,8850,4))
	# t_list2_pre = [(i+2) for i in t_list1_pre]
	# t_list2_pre[-1] = t_list1_pre[-1]+2+5000
	# t_list1 = t_list1+t_list1_pre
	# t_list2 = t_list2+t_list2_pre

	# method1 = 52
	t_list3 = [method1]*len(t_list1)
	# t_list3 = []
	# for i in range(8):
	# 	t_list3.extend([0,1,56])
	# for i in range(2):
	# 	t_list3.extend([56])

	# feature_id1 = 5
	# config = {'type_id':1, 'type_id1':0, 'feature_id1':feature_id1}
	config = {'type_id_1':1, 'type_id1_1':0, 'feature_id1':feature_id1}

	run_idlist = list(np.vstack((t_list1,t_list2,t_list3)).T)

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
		data_2.to_csv(output_filename,index=False,sep='\t')
	else:
		data_1.to_csv(output_filename,index=False,sep='\t')

	return True

# merge estimation files
def test_merge_1(run_idlist,output_filename,config,mode=1):

	if 'file_path' in config:
		file_path = config['file_path']
	else:
		file_path = './mnt/yy3'

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

def run_1_3(file_path=''):

	config = {'compare_annotation':['LR','XGBR','DNN-Local','CONCERT'],
				'sel_idx':[2,8,4,7],'barWidth':2}

	celltype_vec = ['GM12878','K562','IMR-90','H1-hESC','H9','HCT116','HEK293','RPE-hTERT','U2OS']
	celltype_vec = ['GM12878','K562','H1-hESC','HCT116','H9','HEK293','IMR-90','RPE-hTERT','U2OS']
	# celltype_vec = ['mm10','mm10']
	vec1 = dict()
	run_idlist = [[8312,8313],[8314,8315],[8316,8317],[8100,8101]]

	cnt2 = 0
	# run_idlist1 = [list(range(8312,8318,2)),list(range(8324,8330,2)),
	# 				list(range(8318,8324,2)),list(range(8330,8336,2))]
	# run_idlist1 = [list(range(8300,8306,2)),list(range(8306,8312,2)),
	# 				list(range(8336,8342,2)),list(range(8342,8348,2))]
	# run_idlist1 = [list(range(8290,8296,2))]

	# mm10
	run_idlist1 = [list(range(10500,10512,4)),list(range(10501,10512,4))]

	# hg38
	run_idlist1 = [list(range(8290,8296,2)),list(range(8300,8306,2)),
					list(range(8312,8318,2)),list(range(8324,8330,2)),
					list(range(8318,8324,2)),list(range(8330,8336,2)),
					list(range(8306,8312,2)),
					list(range(8336,8342,2)),list(range(8342,8348,2))]
	run_idlist2 = [list(range(8090,8098,2)),list(range(8200,8208,2)),
					list(range(8100,8108,2)),list(range(8108,8116,2)),
					list(range(8116,8124,2)),list(range(8124,8132,2)),
					list(range(8208,8216,2)),list(range(8216,8224,2)),
					list(range(8224,8232,2))]

	filename_list_1 = []

	celltype_num = len(celltype_vec)
	for id1 in range(celltype_num):
		cell_id = celltype_vec[id1]
		t1 = run_idlist1[cnt2]
		t2 = run_idlist2[cnt2]
		t_runlist1 = []
		# for run_id1 in t1:
		# 	t_runlist1.append([run_id1,run_id1+2])
		for run_id1 in t1:
			t_runlist1.append([run_id1,run_id1+1])
		# t_runlist1.append([8200+8*cnt2,8201+8*cnt2])
		# t_runlist1.append([8090+8*cnt2,8091+8*cnt2])
		# t_runlist1.append([8706+cnt2,8708+cnt2])
		run_id2 = t2[0]
		t_runlist1.append([run_id2,run_id2+1])
		vec1['cell_id'] = t_runlist1.copy()
		cnt2 += 1

		method_vec = [0,1,56,52]
		feature_idvec = [5,5,5,5]
		# feature_idvec = [-1,-1,-1,-1]
		run_idlist = t_runlist1

		cnt1 = 0
		filename_list = []
		# for run_id1 in [3,5,4,6]:
		for pair1 in run_idlist:
			method1 = method_vec[cnt1]
			feature_id1 = feature_idvec[cnt1]
			# run_id1 = 8100+8*cnt1
			cnt1 += 1
			# pair1 = [run_id1,run_id1+1]
			filename_list1 = []
			for t_run_id in pair1:
				filename_list1.append('%s/test_vec2_%d_%d_[%d]_%s.txt'%(file_path,t_run_id,method1,feature_id1,cell_id))

			filename_list.append(filename_list1)
		filename_list_1.append([cell_id,filename_list])

		print(filename_list)

		config.update({'run_idlist':run_idlist})
		# output_filename = 'fig_test1_%s.%d.pdf'%(cell_id,cnt2+1)
		print(cell_id,run_idlist)
		# plot_3(filename_list,output_filename,config)
	
	output_filename = 'fig_test1_%s.%d.pdf'%(celltype_vec[0],cnt2)
	plot_3_sub1(filename_list_1,output_filename,config)

def table_format(filename1,pre_colnames=[],type_id1=0):

	data1 = pd.read_csv(filename1,sep='\t')
	colnames = list(data1)
	# fields = ['run_id','chrom','value','metrics','train_chrNum']
	# fields = ['run_id','chrom','train_chrNum']

	# run_id, chrom = data1['run_id'], data1['chrom']
	# train_Num = data1['train_chrNum']

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

def test_2(cell,run_idlist):

	file_path2 = './mnt/yy3'
	run_idlist1 = run_idlist
	method1 = 52
	feature_id1 = -1

	for run_id1 in run_idlist1:

		feature_id1 = -1
		filename1 = '%s/test_vec2_%d_%d_[%d].1_0.txt'%(file_path2,run_id1,method1,feature_id1)
		print(cell)
		print(filename1)

		if os.path.exists(filename1)==False:
			feature_id1 = 5
			filename1 = '%s/test_vec2_%d_%d_[%d].1_0.txt'%(file_path2,run_id1,method1,feature_id1)
			if os.path.exists(filename1)==False:
				print('file does not exist',filename1)
				continue

		data1 = pd.read_csv(filename1,sep='\t')
		fields = ['chrom','start','stop','signal']
		data_1 = data1.loc[:,fields]

		fields = ['chrom','start','stop','predicted_signal']
		data_2 = data1.loc[:,fields]

		signal = np.asarray(data_1['signal'],dtype=np.float32)
		predicted_signal = np.asarray(data_2['predicted_signal'],dtype=np.float32)
		min1 = np.min(signal)
		if min1 > -0.1:
			data_1['signal'] = 2*signal-1
			data_2['predicted_signal'] = 2*predicted_signal-1
		else:
			data_1['signal'] = signal
			data_2['predicted_signal'] = predicted_signal

		t_signal1 = data_1['signal']
		t_signal2 = data_2['predicted_signal']
		print(np.min(t_signal1), np.max(t_signal1))
		print(np.min(t_signal2), np.max(t_signal2))

		# fields = ['chrom','start','stop','predicted_attention']
		# data_3 = data1.loc[:,fields]
		# t_attention = np.asarray(data_3['predicted_attention'],dtype=np.float32)

		# s1, s2 = 0, 1
		# s_min, s_max = np.min(t_attention), np.max(t_attention)
		# scaled_attention = s1+(t_attention-s_min)*1.0/(s_max-s_min)*(s2-s1)
		# data_3['predicted_attention'] = scaled_attention

		fields = ['chrom','start','stop','Q2']
		data_3 = data1.loc[:,fields]
		t_score = data_3['Q2']
		print(np.min(t_score),np.max(t_score))
		
		output_filename = '%s_%d'%(cell,run_id1)
		# if (run_id1<=8707) or (run_id1==8930) or (run_id1==8931):
		# 	output_filename1 = '%s/data5/%s_signal.bedGraph'%(file_path2,output_filename)
		# 	data_1.to_csv(output_filename1,header=False,index=False,sep='\t')
		output_filename1 = '%s/data5/%s_signal.bedGraph'%(file_path2,output_filename)
		data_1.to_csv(output_filename1,header=False,index=False,sep='\t')
		output_filename1 = '%s/data5/%s_predicted.bedGraph'%(file_path2,output_filename)
		data_2.to_csv(output_filename1,header=False,index=False,sep='\t')
		output_filename1 = '%s/data5/%s_score.bedGraph'%(file_path2,output_filename)
		data_3.to_csv(output_filename1,header=False,index=False,sep='\t')

# write to bedGrap and BED files
def run_2():
	cell = 'mm10'
	run_idlist = list(range(8706,8722,4))+list(range(8707,8723,4))+[8930,8938,8942]+[8931,8939,8943]
	run_idlist = [8931]
	test_2(cell,run_idlist)

def test_3_1(filename1,filename2,output_filename,sel_idList,annotation_vec,flag=1,type_id5=0):

	data_1 = pd.read_csv(filename1,sep='\t')
	# sel_colums = ['score1','score2','score3','score4']
	sel_colums = ['score3','score4']
	data1 = data_1.loc[:,sel_colums]

	data_2 = pd.read_csv(filename2,sep='\t')
	sel_colums = ['sel3','sel4']
	data2 = data_2.loc[:,sel_colums]

	data1, data2 = np.asarray(data1), np.asarray(data2)
	b = np.where(data1[:,0]>=0)[0]
	data1 = data1[b]

	num1, num2 = data1.shape[0], data2.shape[0]
	print('regions', num1, num2)

	# if num2>1e09:
	# 	id1 = np.arange(0,num2,10)
	# 	data2 = data2[id1]
	# 	num1, num2 = data1.shape[0], data2.shape[0]
	# 	print('regions', num1, num2)

	list1 = []
	for i in range(2):
		vec1 = score_2a_1(data1[:,i],data2[:,i])
		list1.extend(vec1[0])

	for i in range(2):
		vec1 = score_2a_1(data1[:,i],data2[:,i],alternative='two-sided')
		list1.extend(vec1[0])

	print(list1)

	annotation_vec1 = annotation_vec.copy()
	annotation_vec = annotation_vec[0:2]

	thresh = 0.05
	# if flag==1:
	# 	flag = 0
	# 	if np.min(list1)>thresh:
	# 		return list1, flag
	# 	else:
	# 		flag = 1

	if flag<=0:
		return list1, flag

	# label1 = np.hstack((np.ones(num1),np.zeros(num2)))
	# label1 = np.int8(label1)

	params = {
		 'axes.labelsize': 12,
		 'axes.titlesize': 16,
		 'xtick.labelsize':12,
		 'ytick.labelsize':12}
	pylab.rcParams.update(params)

	label1 = [annotation_vec[0]]*num1 + [annotation_vec[1]]*num2
	label1 = np.asarray(label1)

	region_id1 = list(range(1,num1+1)) + list(data_2['region_id'])

	data1 = np.vstack((data1,data2))
	print(data1.shape)
	print(np.max(data1),np.min(data1))

	fields = ['label','region_id','max','mean']
	data_3 = pd.DataFrame(columns=fields)

	data_3['label'] = label1
	data_3['region_id'] = region_id1
	for i in [2,3]:
		data_3[fields[i]] = data1[:,i-2]
	
	# output_filename1 = '%s.h5'%(output_filename)
	# with h5py.File(output_filename1,'a') as fid:
	# 	fid.create_dataset("vec", data=data_3, compression="gzip")
	# vec1 = ['ERCE','Background']
	vec1 = annotation_vec
	fig = plt.figure(figsize=(12,11))

	cnt1, cnt2, vec2 = 1, 0, ['mean','maximum']
	for sel_id in sel_idList:
		print(sel_id)
		
		plt.subplot(2,2,cnt1)
		cnt1 += 1
		plt.title('Estimated importance score I (%s)'%(vec2[cnt2]))
		# sns.violinplot(x='label', y=sel_id, data=data_3)
		sns.boxplot(x='label', y=sel_id, data=data_3, showfliers=False)
		# sns.boxplot(x='label', y=sel_id, data=data_3, showfliers=True)

		# ax.get_xaxis().set_ticks([])
		ax = plt.gca()
		# ax.get_xaxis().set_visible(False)
		ax.xaxis.label.set_visible(False)

		# output_filename1 = '%s_%s_1.png'%(output_filename,sel_id)
		# plt.savefig(output_filename1,dpi=300)

		plt.subplot(2,2,cnt1)
		cnt1 += 1
		plt.title('Estimated importance score II (%s)'%(vec2[cnt2]))
		cnt2 += 1

		ax = plt.gca()
		ax.xaxis.label.set_visible(False)

		for t_label in vec1:
			b1 = np.where(label1==t_label)[0]
			# t_mtx = data_3[data_3['label']==t_label]
			t_mtx = data_3.loc[b1,fields]
			sns.distplot(t_mtx[sel_id], hist = True, kde = True,
						 kde_kws = {'shade':True, 'linewidth': 3},
						 label = t_label)

		# output_filename2 = '%s_%s_2.png'%(output_filename,sel_id)
		# plt.savefig(output_filename2,dpi=300)

	file_path = './mnt/yy3/data3/fig2'
	cell_id = annotation_vec1[2]
	# output_filename1 = '%s/%s_%s_%s.mapping.1.png'%(file_path,output_filename,sel_id,cell_id)
	if type_id5==0:
		output_filename1 = '%s/%s_%s_%s.1.pdf'%(file_path,output_filename,sel_id,cell_id)
	else:
		output_filename1 = '%s/%s_%s_%s.mapping.1.pdf'%(file_path,output_filename,sel_id,cell_id)
	plt.savefig(output_filename1,dpi=300)

	return list1, flag

def run_3_1(run_idList,cell_id,mode=1,mode2=1,output_filename1='table2_1.1.txt',
				thresh_1=0,tol=2,sample_num=200,thresh=0.05,type_id5=0,annot_1='1'):

	file_path = './mnt/yy3'
	# cell_id = 'mm10'

	# method1 = 52
	# run_idList = list(range(8706,8726,4))+list(range(8707,8726,4))\
	# 			+list(range(8930,8950,4))+list(range(8931,8950,4))\
	# 			+list(range(8850,8922,4)) + list(range(8771,8850,4))\
	# 			+list(range(8851,8922,4)) + list(range(8772,8850,4))

	# run_idList = list(range(8706,8726,4))+list(range(8707,8726,4))\
	# 			+list(range(8930,8950,4))+list(range(8931,8950,4))\
	# 			+list(range(8850,8922,4)) + list(range(8771,8850,4))\
	# 			+list(range(8851,8922,4)) + list(range(8772,8850,4))

	# t_list1_pre = list(range(8850,8922,4)) + list(range(8771,8850,4))
	# t_list1_pre = list(range(8100,8127,2)) + list(range(8200,8231,2)) + list(range(8500,8563,2)) \
	# 			 	+ list(range(9100,9131,2)) + list(range(9500,9579,2)) \
	# 			 	+ list(range(9700,9747,2)) + list(range(9927,9938,2))

	# t_list1 = list(range(8706,8724,4)) + list(range(8930,8948,4))
	# t_list1_1 = list(range(8707,8724,4)) + list(range(8931,8948,4))
	# run_idList = [8715,8939]
	list2 = []
	# run_idList = [8772]
	list3 = []
	
	if cell_id=='mm10':

		# filename_list = ['ERCE.mm10.bed']
		filename_list = ['ERCE.mm10.ori.filter.2.copy1.bed']
		
	elif cell_id=='H1-hESC':

		if type_id5==0:
			# initiation zone
			filename_list = ['H1_allIZ_coor.sorted.txt']
		elif type_id5==1:
			# ERCE mapped
			file_path2 = './mnt/yy3/data6_7'
			filename_list_pre = ['mm10_hg38_ERCE.0.5.remap.bed','mm10_hg38_ERCE.0.4.remap.bed']
			# filename_list_pre = filename_list_pre[1:2]
			filename_list_pre = filename_list_pre[0:1]
			filename_list = ['%s/%s'%(file_path2,filename) for filename in filename_list_pre]
		elif type_id5==2:	
			# motif
			filename_list = ['NANOG_2_ENCFF794GVQ.bed',
								'E2F6_2_ENCFF001UBD.bed',
								'RAD21_2_ENCFF255FRL.bed',
								'POLR2A_2_ENCFF422HDN.bed',
								'POLR2Apho_2_ENCFF418QVJ.bed',
								'EP300_2_ENCFF834UVX.bed',
								'EGR1_2_ENCFF477ANT.bed',
								'TAF1_2_ENCFF243PSJ.bed',
								'RXRA_2_ENCFF430SIE.bed',
								'ATF3_2_ENCFF487GLV.bed',
								'CTCF_2_ENCFF821AQO.bed']
			# filename_list = ['NANOG_2_ENCFF794GVQ.bed']
			# filename_list = [
			# 				'CTCF_2_ENCFF821AQO.bed',
			# 				'E2F6_2_ENCFF001UBD.bed',
			# 				'RAD21_2_ENCFF255FRL.bed',
			# 				'POLR2A_2_ENCFF422HDN.bed',
			# 				'POLR2Apho_2_ENCFF418QVJ.bed',
			# 				'EP300_2_ENCFF834UVX.bed',
			# 				'EGR1_2_ENCFF477ANT.bed',
			# 				'TAF1_2_ENCFF243PSJ.bed',
			# 				'RXRA_2_ENCFF430SIE.bed',
			# 				'ATF3_2_ENCFF487GLV.bed',
			# 				]
		elif type_id5==3:
			# candidate CRE
			# filename_list = ['ENCFF169PCK_ENCFF176SFX_ENCFF984WLE_ENCFF620LDT.7group_H1-hESC.bed',
			# 					'GRCh38-ccREs.bed']
			filename_list_pre = ['ENCFF169PCK_ENCFF176SFX_ENCFF984WLE_ENCFF620LDT.7group_H1-hESC.bed']
			file_path2 = './mnt/yy3/data6_5'
			filename_list = ['%s/%s'%(file_path2,filename) for filename in filename_list_pre]
		else:

			# filename_list_pre = ['test1.250.ERCE.1.txt']
			file_path2 = './mnt/yy3/data6_7'
			file_path2_1 = './mnt/yy3/data6_8'
			filename_list_pre = ['%s/test1.250.ERCE.1.txt'%(file_path2),'%s/test1.250.merge.bed'%(file_path2_1)]
			# filename_list = ['%s/%s'%(file_path2,filename) for filename in filename_list_pre]

			filename_list_pre = ['%s/test2_SOX2_cobinding.250.bed'%(file_path2_1)]
			filename_list_pre = ['%s/test1.SOX2.250.ERCE.1.remap0.5.txt'%(file_path2)]
			filename_list_pre = ['%s/test1.binding.250.ERCE.1.remap0.5.merge.bed'%(file_path2)]
			filename_list = filename_list_pre
			print(filename_list)

	else:

		if type_id5==0:
			# initiation zone
			filename_list = ['HCT116_allIZ_coor.sorted.txt']
		elif type_id5==1:
			# ERCE mapped
			file_path2 = './mnt/yy3/data6_7'
			filename_list_pre = ['mm10_hg38_ERCE.0.5.remap.bed','mm10_hg38_ERCE.0.4.remap.bed']
			# filename_list_pre = filename_list_pre[1:2]
			filename_list = ['%s/%s'%(file_path2,filename) for filename in filename_list_pre]
		elif type_id5==2:	

			# filename_list = ['CTCF_2_ENCFF171SNH.bed',
			# 				'POLR2A_2_ENCFF910KOG.bed',
			# 				'POLR2Apho_2_ENCFF910KOG.bed',
			# 				'ZFX_2_ENCFF215SIC.bed']
			# filename_list = ['POLR2A_2_ENCFF910KOG.bed',
			# 				'POLR2Apho_2_ENCFF910KOG.bed',
			# 				'ZFX_2_ENCFF215SIC.bed',
			# 				'CTCF_2_ENCFF171SNH.bed']
			filename_list = [
							'ZFX_2_ENCFF215SIC.bed',
							'CTCF_2_ENCFF171SNH.bed',
							'POLR2A_2_ENCFF910KOG.bed',
							'POLR2Apho_2_ENCFF910KOG.bed']
		else:
			# filename_list = ['ENCFF169PCK_ENCFF176SFX_ENCFF984WLE_ENCFF620LDT.7group_HCT116.bed',
			# 					'GRCh38-ccREs.bed']
			filename_list_pre = ['ENCFF169PCK_ENCFF176SFX_ENCFF984WLE_ENCFF620LDT.7group_HCT116.bed']
			file_path2 = './mnt/yy3/data6_5'
			filename_list = ['%s/%s'%(file_path2,filename) for filename in filename_list_pre]

	cnt1 = 0
	for region_filename2 in filename_list:
		# t_id1 = region_filename2.find('_2_')
		# feature_name = region_filename2[0:t_id1]

		cnt1 += 1
		# feature_name = 'ccRE%d'%(cnt1)
		# region_filename2 = './mnt/yy3/data6_5/%s'%(region_filename2)

	# flag = True
	# if flag==True:
		for run_id in run_idList:
			run_id1, type_id2, method1 = run_id[0], run_id[1], run_id[2]
			print('type_id2',type_id2)
			type_id, type_id1 = 1,0
			feature_id1 = -1
			if cell_id=='mm10':
				# filename2 = 'ERCE.mm10.bed'
				filename2 = region_filename2
				if (run_id1>=22000) and (runid1<60000):
					pair1 = [run_id1,run_id1+1,method1]
				elif (run_id1>8846) and (run_id1<8850):
					pair1 = [run_id1,run_id1+5002,method1]
				else:
					pair1 = [run_id1,run_id1+2,method1]

			else:
				feature_id1 = 5
				pair1 = [run_id1,run_id1+1,method1]
				filename2 = region_filename2
				# if  cell_id=='H1-hESC':
				# 	# filename2 = 'early_peaks_H1_Jan.txt'
				# 	# filename2 = 'early_peaks_H1_Jan.txt'
				# 	if type_id5==0:
				# 		filename2 = 'H1_allIZ_coor.sorted.txt'
				# 	elif type_id5==1:
				# 		# filename2 = 'ERCE.mm10.filter.mapping.0.25.3.200.bed'
				# 		# filename2 = 'ERCE.mm10.ori.filter.mapping.0.5.3.200.bed'
				# 		filename2 = '%s/mm10_hg38_ERCE.0.5.remap.bed'%(file_path2)
				# 	else:
				# 		filename2 = region_filename2

				# elif cell_id=='HCT116':
				# 	# filename2 = 'initiation_zone_HCT_v1.txt'
				# 	# filename2 = 'early_peaks_H1_Jan.txt'
				# 	if type_id5==0:
				# 		filename2 = 'HCT116_allIZ_coor.sorted.txt'
				# 	elif type_id5==1:
				# 		# filename2 = 'ERCE.mm10.filter.mapping.0.25.4.200.bed'
				# 		filename2 = 'ERCE.mm10.ori.filter.mapping.0.5.4.200.bed'
				# 	else:
				# 		filename2 = region_filename2
				# else:
				# 	break

			file_path1 = './mnt/yy3/data3/fig2'
			# output_filename = '%s/fig_test1_%d_max_1.png'%(file_path1,run_id1)
			output_filename = '%s/fig_test1_%d_max_1.%d.png'%(file_path1,run_id1,type_id5)
			if os.path.exists(output_filename):
				t_flag = -1
			else:
				t_flag = 1
			# t_flag = 1

			# annot1 = '%d_%d.tol%d.init1_1'%(type_id,type_id1,tol)
			# annot2 = '%d_%d.tol%d.init2.2_1'%(type_id,type_id1,tol)
			# annot_vec1 = ['3','mapping2','motif','CRE','','mapping2_1']
			# annot_vec1 = ['3','mapping2','motif','CRE','ERCE','mapping2_1']
			# annot_vec1 = ['3','mapping3','motif','CRE','ERCE','mapping2_1'] #previous
			annot_vec1 = ['3','ERCE_mapping3_1','motif','CRE','ERCE','mapping2_1',
							'ERCE_mapping3_2','binding_1','binding_2','binding_3']
			annot_type_id5 = annot_vec1[type_id5]
			annot1 = '%d_%d.tol%d.init1_%s.thresh%s'%(type_id,type_id1,tol,annot_type_id5,str(thresh_1))
			annot2 = '%d_%d.tol%d.init2.2_%s.thresh%s'%(type_id,type_id1,tol,annot_type_id5,str(thresh_1))
			filename_1_ori = '%s/test_vec2_%d_%d_[%d].%d_%d.1.txt'%(file_path,pair1[0],method1,feature_id1,type_id,type_id1)
			# filename_1 = '%s/test_vec2_%d_%d_[%d].%d_%d.init1.txt'%(file_path,pair1[0],method1,feature_id1,type_id,type_id1)
			# filename_2 = '%s/test_vec2_%d_%d_[%d].%d_%d.init2.txt'%(file_path,pair1[0],method1,feature_id1,type_id,type_id1)
			filename_1 = '%s/test_vec2_%d_%d_[%d].%s.txt'%(file_path,pair1[0],method1,feature_id1,annot1)
			filename_2 = '%s/test_vec2_%d_%d_[%d].%s.txt'%(file_path,pair1[0],method1,feature_id1,annot2)

			config = {'type_id':1, 'type_id1':0, 'feature_id1':feature_id1}
			if os.path.exists(filename_1_ori)==False:
				# merge files
				print(pair1, 'merging...')
				filename_list, output_filename_list, chrom_numList = run_1_merge([pair1],config)
				# output_filename = '%s/test_vec2_%d_%d_[%d].%d_%d.txt'%(file_path,pair1[0],method1,feature_id1,type_id,type_id1)
				# output_filename_1 = output_filename_list[0]
			if (os.path.exists(filename_2)==False) or (mode2==1):
				tol1 = tol
				sample_num1 = sample_num
				# region_filename = 'region_test1_%d.%d.mapping.txt'%(tol1,type_id2)
				# output_filename = 'region_test2_%d.%d.mapping.txt'%(tol1,type_id2)
				region_filename = 'region_test1_%d.%d.%s.txt'%(tol1,type_id2,annot_type_id5)
				output_filename = 'region_test2_%d.%d.%s.txt'%(tol1,type_id2,annot_type_id5)
				filename_vec = [region_filename,output_filename,annot1,annot2]
				run_1_sub1([run_id1],filename2,method1,feature_id1,filename_vec,thresh_1=thresh_1,tol=tol1,sample_num=sample_num1)

			# previous
			# vec1_1 = run_3_2(filename_1, filename_2, thresh=thresh)
			# print(vec1_1)

			# t1 = [run_id1]+vec1_1
			# print(run_id1,vec1_1)
			# list3.append(t1)

			ratio = 0.5
			# output_filename = 'fig_test2_%d_tol%d'%(pair1[0],tol)
			# output_filename = 'fig_test2_%d_tol%d_%s'%(pair1[0],tol,ratio)

			feature_name = cnt1
			output_filename = 'fig_test2_%d_tol%d_%s_%s'%(pair1[0],tol,feature_name,annot_1)
			sel_idList = ['mean','max']
			if cell_id=='mm10':
				annotation_vec = ['ERCE','Background',cell_id]
			else:
				if type_id5==0:
					annotation_vec = ['Initiation zone','Background',cell_id]
				elif (type_id5==1) or (type_id5==5):
					annotation_vec = ['ERCE(mapped)','Background',cell_id]
				elif type_id5==2:
					annotation_vec = ['Peak region','Background',cell_id]
				else:
					annotation_vec = ['CRE','Background',cell_id]

			list1, flag = test_3_1(filename_1,filename_2,output_filename,sel_idList,annotation_vec,t_flag,type_id5)
			list2.append([run_id1,flag]+list1)

	vec1 = ['wilcoxon_pvalue','mannwhitney_pvalue','ks_pvalue']
	vec2 = ['max','mean']
	fields = ['run_id','label']
	for l1 in range(2):
		for t1 in vec1[1:]:
			fields.append('%s_%s.1'%(t1,vec2[l1]))

	for l1 in range(2):
		for t1 in vec1[1:]:
			fields.append('%s_%s.2'%(t1,vec2[l1]))

	data_1 = pd.DataFrame(columns=fields)
	list2 = np.asarray(list2)
	print(list2)
	data_1['run_id'] = np.int64(list2[:,0])
	data_1['label'] = np.int64(list2[:,1])
	num1 = len(fields)
	for i in range(2,num1):
		data_1[fields[i]] = list2[:,i]

	if (os.path.exists(output_filename1)==True) and (mode==1):
		data_pre = pd.read_csv(output_filename1,sep='\t')
		data_2 = pd.concat([data_pre,data_1], axis=0, join='outer', ignore_index=True, 
			keys=None, levels=None, names=None, verify_integrity=False, copy=True)
		data_2.to_csv(output_filename1,index=False,sep='\t')
	else:
		data_1.to_csv(output_filename1,index=False,sep='\t')

	# return list3
	return True

def run_3_2(filename1, filename2, thresh=0.05):
	
	# file of regions
	data1 = pd.read_csv(filename1,sep='\t')
	score1_1, score2_1 = data1['score1'], data1['score2']
	score1, score2 = data1['score3'], data1['score4']
	colnames = list(data1)
	chrom1, start1, stop1 = data1[colnames[0]], data1[colnames[1]], data1[colnames[2]]
	chrom_vec = np.unique(chrom1)
	num1 = len(score1)

	# region_id	start	stop	sel1	sel2	sel3	sel4
	data2 = pd.read_csv(filename2,sep='\t')
	region_id = np.asarray(data2['region_id'])
	sample_score1, sample_score2 = np.asarray(data2['sel3']), np.asarray(data2['sel4'])

	score1, score2 = np.asarray(score1,dtype=np.float32), np.asarray(score2,dtype=np.float32)
	sample_score1, sample_score2 = np.asarray(sample_score1,dtype=np.float32), np.asarray(sample_score2,dtype=np.float32)

	empirical1, empirical2 = -np.ones(num1), -np.ones(num1)
	len_vec = []

	for t_chrom in chrom_vec:
		b1 = np.where(chrom1==t_chrom)[0]
		num2 = len(b1)
		# if (t_chrom=='chrX') or (t_chrom=='chrY') or (t_chrom=='chrM')
		# 	print('chromosome not estimated', t_chrom)
		# 	continue

		len1 = 0
		for l in range(0,num2):
			id1 = b1[l]
			t_region_id = id1+1
			b2 = np.where(region_id==t_region_id)[0]
			sample_num = len(b2)
			if sample_num==0:
				continue
			# if l%100==0:
			# 	print(t_region_id,chrom1[id1],start1[id1],stop1[id1],len(b2))
			t_score1, t_score2 = score1[id1], score2[id1]
			len1 += stop1[id1]-start1[id1]
			t_sample_score1, t_sample_score2 = sample_score1[b2], sample_score2[b2]
			b_1 = np.where(t_sample_score1>=t_score1-1e-05)[0]
			b_2 = np.where(t_sample_score2>=t_score2-1e-05)[0]
			ratio1 = len(b_1)/sample_num
			ratio2 = len(b_2)/sample_num
			empirical1[id1], empirical2[id1] = ratio1, ratio2
			# print(id1,score1_1[id1],score2_1[id1],t_score1,t_score2,ratio1,ratio2,len(b_1),len(b_2),
			# 		np.max(t_sample_score1),np.max(t_sample_score2),
			# 		np.min(t_sample_score1),np.min(t_sample_score2))

		len_vec.append(len1)

	print(len_vec)
	data1['empirical1'] = empirical1
	data1['empirical2'] = empirical2
	id1 = np.where(empirical1!=-1)[0]
	b1 = np.where(empirical1[id1]<thresh)[0]
	b2 = np.where(empirical2[id1]<thresh)[0]
	b1, b2 = id1[b1], id1[b2]
	b_1 = np.intersect1d(b1,b2)
	b_2 = np.union1d(b1,b2)
	n1_1, n1_2 = len(b1), len(b2)
	n1, n2 = len(b_1), len(b_2)
	num1 = len(id1)
	if num1>0:
		vec1 = [n1_1,n1_2,n1,n2,n1_1/num1,n1_2/num1,n1/num1,n2/num1]
	else:
		vec1 = [n1_1,n1_2,n1,n2,-1,-1,-1,-1]
	# print(vec1)

	colnames = list(data1)
	data_1 = data1.loc[b_2,colnames]

	data1.to_csv(filename1, index=False, sep='\t')
	id2 = filename1.find('txt')
	filename2 = filename1[0:id2]+'1_1.txt'
	data1.to_csv(filename1, index=False, sep='\t')
	data_1.to_csv(filename2, index=False, sep='\t')

	return vec1

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

# input: estimated attention, type_id: training, validation, or test data
# output: ranking of attention
def select_region2_sub(filename,filename_centromere=''):

	data1 = pd.read_csv(filename,sep='\t')
	colnames = list(data1)

	if filename_centromere!='':
		chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
		id1 = select_idx_centromere(chrom,start,stop,filename_centromere)
		print('select_idx_centromere', len(chrom), len(id1), len(id1)/len(chrom))
		data1 = data1.loc[id1,:]

	# chrom	start	stop	serial	signal	predicted_signal	predicted_attention
	chrom, start, serial = data1['chrom'], data1['start'], data1['serial']
	chrom, start, serial = np.asarray(chrom), np.asarray(start), np.asarray(serial)
	predicted_attention = data1['predicted_attention']
	predicted_attention = np.asarray(predicted_attention)
	id_fold = np.asarray(data1['id_fold'])
	id_fold_vec = np.unique(id_fold)
	rank1 = np.zeros(len(predicted_attention),dtype=np.float32)

	flag1 = 0
	if 'predicted_attention1' in colnames:
		flag1 = 1
		predicted_attention1 = np.asarray(data1['predicted_attention1'])
		rank_1 = rank1.copy()

	for id1 in id_fold_vec:
		b1 = (id_fold==id1)
		t_attention = predicted_attention[b1]
		t_ranking = stats.rankdata(t_attention,'average')/len(t_attention)
		rank1[b1] = t_ranking

		if flag1==1:
			t_attention1 = predicted_attention1[b1]
			t_ranking1 = stats.rankdata(t_attention1,'average')/len(t_attention1)
			rank_1[b1] = t_ranking1

	# data1['Q1'] = rank1[:,0]	# rank across all the included chromosomes
	data1['Q2'] = rank1	# rank by each fold
	data1['typeId'] = data1['id_fold']

	if flag1==1:
		# data1['Q1_1'] = rank_1[:,0]	# rank across all the included chromosomes
		data1['Q2_1'] = rank_1 # rank by each fold
		t1 = np.hstack((rank1[:,np.newaxis],rank_1[:,np.newaxis]))
		# data1['Q1(2)'] = np.max(t1[:,[0,2]],axis=1)
		data1['Q2(2)'] = np.max(t1,axis=1)

	return data1

# input: estimated attention, type_id: training, validation, or test data
# output: ranking of attention
# filename_centromere: centromere file
def select_region3_sub(filename,data1=[],filename_centromere=''):

	if len(data1)==0:
		data1 = pd.read_csv(filename,sep='\t')
	colnames = list(data1)

	if filename_centromere!='':
		chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
		id1 = select_idx_centromere(chrom,start,stop,filename_centromere)
		print('select_idx_centromere', len(chrom), len(id1), len(id1)/len(chrom))
		data1 = data1.loc[id1,:]
		# data1.reset_index(drop=True,inplace=True)

	# chrom	start	stop	serial	signal	predicted_signal	predicted_attention
	chrom, start, serial = data1['chrom'], data1['start'], data1['serial']
	chrom, start, serial = np.asarray(chrom), np.asarray(start), np.asarray(serial)
	predicted_attention = data1['predicted_attention']
	predicted_attention = np.asarray(predicted_attention)

	ranking = stats.rankdata(predicted_attention,'average')/len(predicted_attention)
	rank1 = np.zeros((len(predicted_attention),2))
	rank1[:,0] = ranking
	rank1_1 = []

	flag1 = 0
	if 'predicted_attention1' in colnames:
		flag1 = 1
		predicted_attention1 = np.asarray(data1['predicted_attention1'])

		ranking1 = stats.rankdata(predicted_attention1,'average')/len(predicted_attention1)
		rank1_1 = np.zeros((len(predicted_attention1),2))
		rank1_1[:,0] = ranking1

	chrom_vec = np.unique(chrom)
	for t_chrom in chrom_vec:
		b1 = np.where(chrom==t_chrom)[0]
		t_attention = predicted_attention[b1]
		t_ranking = stats.rankdata(t_attention,'average')/len(t_attention)
		rank1[b1,1] = t_ranking

		if flag1==1:
			t_attention1 = predicted_attention1[b1]
			t_ranking1 = stats.rankdata(t_attention1,'average')/len(t_attention1)
			rank1_1[b1,1] = t_ranking1

	return chrom_vec,rank1,rank1_1


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

# sample regions randomly to compare with elements
# input: chrom1, start1, stop1, attention1 in genomic loci file
def compare_with_regions_sub2(chrom1,start1,stop1,attention1,
						sample_num,position,region_list,bin_size,tol=2):
	
	chrom, start, stop = position
	region_len = stop-start
	tol1 = int(region_len/bin_size)+2

	start_pos_list = []
	for t_region in region_list:
		# region_size = len(t_region)
		region_size = t_region[1] - t_region[0]+1
		start_pos = t_region[0]+np.random.permutation(region_size-tol1)
		start_pos_list.extend(start_pos)

	sel_num = attention1.shape[1]
	vec2 = -np.ones((sample_num,2+2*sel_num))
	start_pos_list = np.asarray(start_pos_list)
	np.random.shuffle(start_pos_list)

	# start_pos1 = start_pos[0:sample_num]
	start_pos1 = start1[start_pos_list] # coordinate
	pos1 = np.vstack((start_pos1,start_pos1+region_len)).T
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

	# vec2 = vec2[vec2[:,0]>=0]

	return vec2

# sample regions randomly to compare with elements
# input: chrom1, start1, stop1, attention1 in genomic loci file
def compare_with_regions_sub3(chrom1,start1,stop1,attention1,
						sample_num,region_len_tol,region_list,bin_size,tol=2,
						thresh_vec=[0.9,0.95,0.975,0.99]):
	
	# chrom, start, stop = position
	# region_len = stop-start
	# tol1 = int(region_len/bin_size)+2
	tol1 = int(region_len_tol/bin_size)+2

	start_pos_list = []
	# print(chrom1,region_list)
	for t_region in region_list:
		# region_size = len(t_region)
		region_size = t_region[1] - t_region[0]+1
		if region_size<=tol1:
			continue
		start_pos = t_region[0]+np.random.permutation(region_size-tol1)
		start_pos_list.extend(start_pos)

	sel_num = attention1.shape[1]
	vec2 = -np.ones((sample_num,2+2*sel_num+len(thresh_vec)))
	start_pos_list = np.asarray(start_pos_list)
	np.random.shuffle(start_pos_list)

	# start_pos1 = start_pos[0:sample_num]
	# print(start_pos_list)
	start_pos1 = start1[start_pos_list] # coordinate
	pos1 = np.vstack((start_pos1,start_pos1+region_len_tol)).T
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
			sel_attention = attention1[b1_ori,l]
			t_vec2.extend([np.max(sel_attention),np.mean(sel_attention)])
		
		# frequency of point estimation
		t_vec3 = []
		sel_attention = attention1[b1_ori,sel_num-1]
		for thresh1 in thresh_vec:
			b1 = np.where(sel_attention>thresh1)[0]
			t_vec3.append(len(b1)/len1)

		vec2[cnt1] = [t_start2,t_stop2]+t_vec2+t_vec3

		cnt1 += 1
		if cnt1>=sample_num:
			break

	# vec2 = vec2[vec2[:,0]>=0]
	return vec2

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
	# serial = np.asarray(data2[colnames[3]])
	# signal = np.asarray(data2[colnames[4]])
	
	chrom_vec = ['chr%d'%(i) for i in range(1,chrom_num+1)]
	id1, id2 = dict(), dict()

	# serial1, num_vec = -np.ones(sample_num1,dtype=np.int64), np.zeros(sample_num1,dtype=np.int64)
	# signal1 = np.zeros(sample_num1,dtype=np.float32)
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
				# serial1[id_1] = serial[b2[idx[0]]]
				# num_vec[id_1] = len(idx)
				# signal1[id_1] = signal[b2[idx[0]]]
				id_2 = b2[idx]
				overlapping = 0
				for t_id in id_2:
					t_start2, t_stop2 = start2[t_id], stop2[t_id]
					overlapping += np.min([t_stop1-t_start2,t_stop2-t_start2,t_stop2-t_start1,t_stop1-t_start1])
				label[id_1] = overlapping

	colnames = list(data1)
	# data1['serial'] = serial1
	# data1['num'] = num_vec
	# data1['signal'] = signal1
	data1['label'] = label
	# data1 = data1.loc[:,colnames[0:3]+[(colnames[-2])]+['serial','num','signal']]

	data1.to_csv(output_filename,index=False,header=False,sep='\t')

	return True

def compare_with_regions_motif(filename1,filename2,ref_serial=[],motif_data=[],motif_name=[],thresh_vec=[0.05,0.05],species_id='hg38'):

	motif_data1, motif_filename = compare_with_regions_motif_sub2(species_id,filename2,motif_data)
	ref_serial = np.asarray(motif_data1['serial'])
	motif_data = np.asarray(motif_data1.loc[:,motif_filename])

	data1 = pd.read_csv(filename1,sep='\t')
	print(list(data1))
	chrom, start, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['serial'])

	# print(ref_serial)
	# print(serial)
	np.savetxt('ref_serial_test.txt',ref_serial,fmt='%d',delimiter='\t')
	id1 = mapping_Idx(ref_serial,serial)
	b1 = np.where(id1>=0)[0]
	if len(b1)!=len(serial):
		print('error!', b1, len(serial))

	id1 = id1[b1]
	motif_data1 = motif_data[id1]

	colnames1 = list(data1)
	data1 = data1.loc[b1,colnames1]
	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	attention1, attention1_chrom = np.asarray(data1['predicted_attention']), np.asarray(data1['Q2'])
	# column_list = ['2_0.5','2_0.9','1_0.5','1_0.9']
	column_list = ['2_0.9','2_0.95','1_0.9','1_0.95']
	thresh_vec1 = [0.99,0.975,0.95]
	region_num = data1.shape[0]

	print(motif_data1.shape,data1.shape)
	
	dict1 = dict()
	for t_column in column_list:
		peak1 = np.asarray(data1[t_column])
		label = peak1<0
		region_list = (chrom,start,label)
		vec1, vec2 = compare_with_regions_motif_sub1(region_list,motif_name,motif_data1,thresh_vec)
		dict1[t_column] = [vec1,vec2]

	for thresh in thresh_vec1:
		label = attention1_chrom>thresh
		region_list = (chrom,start,label)
		vec1, vec2 = compare_with_regions_motif_sub1(region_list,motif_name,motif_data1,thresh_vec)
		dict1['thresh_%s'%(thresh)] = [vec1,vec2]

	return ref_serial, motif_data, motif_name, dict1

def compare_with_regions_motif_sub1(region_list,motif_name,motif_data,thresh_vec=[0.05,0.05]):

	chrom, start, label = region_list
	region_num = len(chrom)

	motif_num = len(motif_name)
	test_num = 2
	vec1 = np.zeros((motif_num,test_num*2))

	thresh, thresh_fdr = thresh_vec
	b1 = np.where(label==1)[0]
	b2 = np.where(label==0)[0]
	num1, num2 = len(b1), len(b2)
	print(num1, num2, num1/region_num)

	for i in range(motif_num):
		data1, data2 = motif_data[b1,i], motif_data[b2,i]
		id1 = np.random.permutation(num2)
		# data1 = data2[id1[0:num1]]
		# data2 = data2[id1[num1:]]
		# data2 = data2[id1[0:num1]]
		# print(len(data1), len(data2))
		# if i>10:
		# 	break

		t1, t2 = np.median(data1), np.median(data2)
		value1, value2 = score_2a_1(data1, data2, alternative='two-sided')
		# value1, value2 = score_2a_1(data1, data2, alternative='greater')
		mannwhitneyu_pvalue,ks_pvalue = value1[0], value1[1]
		thresh1 = 0.5
		if (mannwhitneyu_pvalue<thresh*0.1) and (ks_pvalue<thresh*0.1) and (np.abs(t1-t2)>thresh1):
			print(motif_name[i], mannwhitneyu_pvalue, ks_pvalue,
					np.median(data1),np.median(data2))
		vec1[i,0:test_num] = value1

	# p-value correction with Benjamini-Hochberg correction procedure
	list1, list2 = [], []
	for i in range(test_num):
		try:
			vec2 = multipletests(vec1[:,i],alpha=thresh_fdr,method='fdr_bh')
			vec1[:,i+test_num] = vec2[1]
			b1 = np.where(vec1[:,i]<thresh)[0]
			b2 = np.where(vec1[:,i+test_num]<thresh_fdr)[0]
			if i==0:
				id1, id2 = b1, b2
			else:
				id1, id2 = np.intersect1d(id1,b1), np.intersect1d(id2,b2)
			print(len(b1),len(b2),len(id1),len(id2))
			# print(motif_name[id2])
			list1.append(b1)
			list2.append(b2)
		except:
			list1.append([])
			list2.append([])
			id1, id2 = [], []

	return vec1, (list1, list2, id1, id2)

def compare_with_regions_motif_sub2(species_id,motif_filename,motif_data_ori=[],region_unit_size=1000):

	if len(motif_data_ori)==0:
		motif_data_ori = pd.read_csv(motif_filename,sep='\t')
	
	colnames = list(motif_data_ori)
	# print(colnames)
	motif_name = np.asarray(colnames[3:])
	chrom2, start2, stop2 = np.asarray(motif_data_ori[colnames[0]]), np.asarray(motif_data_ori[colnames[1]]), np.asarray(motif_data_ori[colnames[2]])

	if species_id=='hg38':
		filename_1 = '/work/magroup/yy3/data2/genome/hg38.chrom.sizes'
		chrom_num = 22
	else:
		filename_1 = '/work/magroup/yy3/data2/genome/mm10/mm10.chrom.sizes'
		chrom_num = 19
			
	ref_serial, start_vec = generate_serial_start(filename_1,chrom2,start2,stop2,chrom_num=chrom_num,type_id=1)
	# motif_data['serial'] = ref_serial

	b1 = np.where(ref_serial>=0)[0]
	ref_serial = ref_serial[b1]
	mtx2_ori = np.asarray(motif_data_ori.loc[b1,motif_name],dtype=np.float32)
	# print(np.max(mtx2_ori),np.min(mtx2_ori),np.median(mtx2_ori))

	region_len = stop2-start2
	region_len1 = region_len/region_unit_size
	region_len1 = region_len1[b1]

	b2 = np.where(region_len!=np.median(region_len))[0]
	print(np.max(region_len),np.min(region_len),len(b1))
	if len(b2)>chrom_num:
		print('error!',len(b1),chrom2[b2])

	print('motif',len(motif_name),mtx2_ori.shape)
	region_num, motif_num = mtx2_ori.shape[0], mtx2_ori.shape[1]
	motif_data = np.asarray(mtx2_ori/np.outer(region_len1,np.ones(motif_num)),dtype=np.float32)
	# print(np.max(motif_data),np.min(motif_data),np.median(motif_data))

	# motif_data1 = motif_data_ori.copy()
	print(len(b1))
	fields = list(motif_name)
	motif_data1 = pd.DataFrame(columns=fields,data=motif_data)
	motif_data1['chrom'], motif_data1['start'], motif_data1['stop'] = chrom2[b1], start2[b1], stop2[b1]
	print(len(ref_serial),len(motif_name))
	motif_data1['serial'] = ref_serial
	print(motif_data1.shape)

	# motif_data1.to_csv('test_hg38_motif.1.txt',index=False,sep='\t')
	# print(motif_data1.shape)

	return motif_data1, motif_name

def compare_with_regions_motif_test(run_idList,filename_list,filename_list2,species_id,thresh_vec=[0.05,0.05]):

	init_vec = {0:[],1:[],2:[]}
	motif_data_dict, motif_name_dict,ref_serial_dict = init_vec, init_vec.copy(), init_vec.copy()
	
	dict2 = dict()
	thresh = 0.05
	vec2 = []
	fields = ['run_id']
	cnt1 = 0
	for (run_id, filename1) in zip(run_idList,filename_list):
		run_id1, type_id1 = run_id[0], run_id[1]
		print(run_id1,type_id1)
		id_1 = type_id1-1
		filename2 = filename_list2[id_1]
		ref_serial, motif_data, motif_name, dict1 = compare_with_regions_motif(filename1,filename2,
										ref_serial=ref_serial_dict[id_1],motif_data=motif_data_dict[id_1],motif_name=motif_name_dict[id_1],
										thresh_vec=[0.05,0.05],species_id=species_id)
		if len(ref_serial_dict[id_1])==0:
			ref_serial_dict[id_1], motif_data_dict[id_1], motif_name_dict[id_1] = ref_serial, motif_data, motif_name

		dict2[run_id1] = dict1
		key_values = list(dict1.keys())
		vec2_1 = []
		for key1 in key_values:
			value1, value2 = dict1[key1]
			list1, list2, id1, id2 = value2

			if cnt1==0:
				motif_data = motif_data_dict[id_1]
				motif_num = motif_data.shape[1]
				cnt_vec = np.zeros((motif_num,2))
				cnt_vec1 = np.zeros_like(value1)
				fields.extend([key1+'_mannwhitneyu',key1+'_ks',key1+'_2',key1+'_mannwhitneyu_fdr',key1+'_ks_fdr',key1+'_2_fdr'])
			
			cnt_vec1 += -np.log(value1)
			for b1 in list1:
				cnt_vec[b1,0] += 1
				vec2_1.append(len(b1))
			for b1 in list2:
				cnt_vec[b1,1] += 1
				vec2_1.append(len(b1))

			vec2_1.extend([len(id1),len(id2)])

		vec2.append(vec2_1)

		cnt1+=1

	run_idList = np.asarray(run_idList)
	t1 = run_idList[:,0]
	data2 = np.int64(np.hstack((t1[:,np.newaxis],np.asarray(vec2))))
	data_2 = pd.DataFrame(columns=fields,data=data2)
	output_filename = 'test_motif_%d.1.txt'%(type_id1)
	data_2.to_csv(output_filename,index=False,sep='\t')

	id1 = np.argsort(-cnt_vec[:,1])
	motif_name1 = motif_name[id1]
	cnt_vec = cnt_vec[id1]
	fields = ['motif_name','cnt1','cnt2_fdr']
	data3 = pd.DataFrame(columns=fields)
	data3['motif_name'] = motif_name1
	data3['cnt1'] = cnt_vec[:,0]
	data3['cnt2_fdr'] = cnt_vec[:,1]
	output_filename = 'test_motif_%d.2.txt'%(type_id1)
	data3.to_csv(output_filename,index=False,sep='\t')

	id1 = np.argsort(-cnt_vec1[:,2])
	motif_name1 = motif_name[id1]
	cnt_vec1 = cnt_vec1[id1]
	fields = ['motif_name','logP1','logP2','logP1_fdr','logP2_fdr']
	data3 = pd.DataFrame(columns=fields)
	data3['motif_name'] = motif_name1
	for i in range(1,5):
		data3[fields[i]] = cnt_vec1[:,i-1]
	output_filename = 'test_motif_%d.3.txt'%(type_id1)
	data3.to_csv(output_filename,index=False,sep='\t')

	np.save('test_motif_%d.npy'%(type_id1),dict2,allow_pickle=True)

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

	file_path = './mnt/yy3/data3/fig2'
	cell_id = annotation_vec[3]
	flag = annotation_vec[-1]
	output_filename1 = '%s/%s_%s_%d.png'%(file_path,output_filename1,cell_id,flag)
	plt.savefig(output_filename1,dpi=300)

	return True

def plot_sub2(filename1):

	params = {
		 'axes.labelsize': 12,
		 'axes.titlesize': 16,
		 'xtick.labelsize':12,
		 'ytick.labelsize':12}
	pylab.rcParams.update(params)

	fig = plt.figure(figsize=(26,12))

	for i in range(2):

		filename1 = 'test_3_1845_26736_35_3.pred.init1.txt'
		data1 = pd.read_csv(filename1,sep='\t')
		y_label = np.asarray(data1['label'])
		y_pred = np.asarray(data1['predicted_label'])
		y_prob = np.asarray(data1['predicted_prob'])
		fpr, tpr, thresh1 = roc_curve(y_label, y_prob)
		prec, recall, thresh1 = precision_recall_curve(y_label, y_prob)

		vec1 = score_function(y_label, y_pred, y_prob)
		accuracy, auc, aupr, precision, recall, F1 = vec1
		print(vec1)

		plt.subplot(1,2,1)
		print(fpr,tpr)
		print(fpr.shape,tpr.shape)
		plt.plot(fpr,tpr,color='orange',linewidth=2,label='Flanking: 45')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('AUROC: IZ prediction in H1-hESC')

		plt.subplot(1,2,2)
		plt.plot(recall,prec,color='orange',linewidth=2,label='Flanking: 45')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('AUPR: IZ prediction in H1-hESC')

	output_filename1 = 'test1.png'
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

def local_peak_preliminary(x1):

	x = 1

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

def compare_with_regions_motif1_2(motif_data,motif_name,est_data,label,output_filename,
									sel_idvec=[2,1,1,1],sel_column='Q2.adj',thresh_vec=[0.05,0.1]):

	chrom, start, stop, serial = np.asarray(est_data['chrom']), np.asarray(est_data['start']), np.asarray(est_data['stop']), np.asarray(est_data['serial'])
	# attention_1 = np.asarray(est_data['predicted_attention'])
	attention1 = np.asarray(est_data[sel_column])

	config = {'feature_name_list':motif_name,'tol':0}
	value = np.asarray(motif_data.loc[:,motif_name])
	data2, data_2, feature_name_list = compare_with_regions_single(chrom,start,stop,serial,label,value,attention1,
																		thresh_vec=thresh_vec,config=config)
	
	output_filename1 = '%s_fc.txt'%(output_filename)
	output_filename2 = '%s_pvalue.txt'%(output_filename)
	data2.to_csv(output_filename1,index=False,sep='\t')
	data_2.to_csv(output_filename2,index=False,sep='\t')

	thresh1, thresh2 = 1, 0.05
	mtx1 = np.asarray(data2.loc[:,feature_name_list])
	mtx2 = np.asarray(data_2.loc[:,feature_name_list])
	b1 = (mtx1>thresh1)
	# b2 = (mtx2<thresh2)
	# b_2 = (mtx2>1-thresh2)
	region_num1 = mtx1.shape[0]
	t1 = np.sum(b1,axis=0)/region_num1
	# t2, t_2 = np.sum(b2,axis=0)/region_num1, np.sum(b_2,axis=0)/region_num1
	id1 = np.argsort(-t1)
	feature_name_list1 = feature_name_list[id1]
	list1 = [mtx1[:,id1], mtx2[:,id1]]
	print(list1[0][0:10])
	print(feature_name_list1[0:10])

	fields = ['name']
	str1 = ['max','min','median','mean']
	column_list = []
	for i in range(2):
		t_column = []
		for t1 in str1:
			t_column.append('%s.%d'%(t1,i+1))
		column_list.append(t_column)
		fields.extend(t_column)

	data3 = pd.DataFrame(columns=fields)
	data3['name'] = feature_name_list1
	for i in range(2):
		t_mtx = list1[i]
		data3.loc[:,column_list[i]] = np.column_stack((np.max(t_mtx,axis=0),np.min(t_mtx,axis=0),np.median(t_mtx,axis=0),np.mean(t_mtx,axis=0)))

	output_filename3 = '%s_stats.txt'%(output_filename)
	data3.to_csv(output_filename3,index=False,sep='\t')

	return data2, data_2, data3

def compare_with_regions_motif_test1(run_idList,filename_list,filename_list2,species_id,celltype_id,thresh_vec=[0.05,0.05],config={}):

	init_vec = {0:[],1:[],2:[]}
	motif_data_dict, motif_name_dict,ref_serial_dict = init_vec, init_vec.copy(), init_vec.copy()
	
	dict2 = dict()
	thresh = 0.05
	vec2 = []
	fields = ['run_id']
	cnt1 = 0
	filename_list2 = ['129S1_SvImJ.motif.count.txt','CAST_EiJ.motif.count.txt']
	file_path2 = '/work/magroup/yy3/data1/replication_timing3'
	# filename2 = '%s/hg38.motif.txt'%(file_path2)
	# filename_list2.append(filename2)
	filename_list2.append('hg38.motif.count.txt')

	if species_id=='mm10':
		filename2 = filename_list2[celltype_id-1]
		# regionlist_filename = 'mm10_gap.txt'
		regionlist_filename = ""
	else:
		filename2 = filename_list2[2]
		regionlist_filename = 'hg38.centromere.bed'

	motif_data_ori, motif_name = compare_with_regions_motif_sub2(species_id,filename2)
	# motif_data_ori = pd.read_csv('test_hg38_motif.1.txt',sep='\t')
	colnames = list(motif_data_ori)
	print(colnames[0:10],colnames[-10:])
	# motif_name = colnames[0:-4]

	ref_serial = np.asarray(motif_data_ori['serial'])
	print('motif_data',motif_data_ori.shape,len(motif_name),len(ref_serial))
	print(motif_name[0:10])
	# return -1

	type_id5 = config['type_id5']
	if celltype_id<=2:
		region_filename1 = 'ERCE.mm10.bed'
	elif celltype_id==3:
		if type_id5==0:
			region_filename1 = 'H1_allIZ_coor.sorted.txt'
		else:
			# region_filename1 = 'ERCE.mm10.filter.mapping.0.25.3.200.bed'
			region_filename1 = 'ERCE.mm10.ori.filter.mapping.0.5.3.200.bed'
	else:
		if type_id5==0:
			region_filename1 = 'HCT116_allIZ_coor.sorted.txt'
		else:
			# region_filename1 = 'ERCE.mm10.filter.mapping.0.25.4.200.bed'
			region_filename1 = 'ERCE.mm10.ori.filter.mapping.0.5.4.200.bed'
		
	region_data = pd.read_csv(region_filename1,header=None,sep='\t')
	region_chrom, region_start, region_stop = np.asarray(region_data[0]), np.asarray(region_data[1]), np.asarray(region_data[2])
	init_zone = (region_chrom,region_start,region_stop)
	config['init_zone'] = init_zone
	flanking = 30
	if not ('flanking1' in config):
		config['flanking1'] = flanking
	else:
		flanking = config['flanking1']

	for (run_id, filename1) in zip(run_idList,filename_list):
		run_id1, celltype_id = run_id[0], run_id[1]
		print(run_id1,celltype_id)
		config['celltype_id'] = celltype_id
		# thresh_vec=[0.05,0.05]

		# regionlist_filename = 'hg38.centromere.bed'
		est_data, sel_id1 = adjust_attention(filename1,regionlist_filename,thresh1=100)
		est_data = est_data.loc[sel_id1,:]
		est_data.reset_index(drop=True,inplace=True)

		print(est_data.shape,len(sel_id1))

		# dict1: regions selected: local peaks of scores, init zone, signal peak, signal prediction accuracy
		sel_idvec = [0,1,0,0]
		dict1 = compare_with_regions_motif1_1(est_data,sel_idvec=sel_idvec,sel_column='Q2.adj',thresh1=0.95,config=config)

		chrom, serial = np.asarray(est_data['chrom']), np.asarray(est_data['serial'])
		print(motif_data_ori.shape,est_data.shape)
		id1 = mapping_Idx(ref_serial,serial)
		b1 = np.where(id1>=0)[0]
		id1 = id1[b1]
		motif_data = motif_data_ori.loc[id1,:]
		est_data = est_data.loc[b1,:]
		print(motif_data.shape,est_data.shape)

		label = compare_with_regions_motif1_sub1(motif_data,motif_name,est_data,dict1,
										sel_idvec=sel_idvec,sel_column='Q2.adj',thresh1=0.9)

		print(len(label),np.sum(label>0),np.sum(label==0),np.sum(label<0))

		output_filename = '%d_%d_motif'%(run_id1,celltype_id)
		data2, data_2, data3 = compare_with_regions_motif1_2(motif_data,motif_name,est_data,label,output_filename,
									sel_idvec=sel_idvec,sel_column='Q2.adj',thresh_vec=[0.05,0.1])

		n1, t1 = len(sel_idvec), 0
		for i1 in range(n1):
			t1 += sel_idvec[i1]*(2**(n1-i1-1))
		output_filename1 = 'test_motif_%d'%(t1)
		if (sel_idvec[1]>0) or (sel_idvec[2]>0):
			output_filename1 = 'test_motif_%d_%d'%(t1,flanking)
		config['output_filename'] = output_filename1
		compare_with_regions_distribute_test(motif_data,motif_name,est_data,label,thresh_vec=[0.05,0.1],config=config)

	return True

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

# adjust attention in mapping regions
def adjust_attention(filename1,regionlist_filename,thresh1=100):
	
	data1 = pd.read_csv(filename1,sep='\t')
	print(list(data1))
	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	attention1 = np.asarray(data1['predicted_attention'])
	sample_num = len(chrom)
	attention2 = -np.ones((sample_num,2),dtype=np.float32)

	# regionlist_filename = 'hg38.centromere.bed'
	if regionlist_filename!="":
		serial_list1, centromere_serial = select_region(chrom,start,stop,serial,regionlist_filename)
		id1 = mapping_Idx(serial,serial_list1)
		id1_centromere = mapping_Idx(serial,centromere_serial)
		b1 = np.where(id1>=0)[0]
		id1 = id1[b1]
		if len(id1)!=len(serial_list1):
			print('error!', len(id1), len(serial_list1))
			return -1

		chrom, start, stop, serial = chrom[id1], start[id1], stop[id1], serial[id1]
		attention1 = attention1[id1]
		print(len(id1),len(id1_centromere),len(attention1))
	else:
		id1 = np.arange(sample_num)
		id1_centromere = []

	seq_list = generate_sequences_chrom(chrom,serial,gap_tol=5, region_list=[])
	len1 = seq_list[:,1]-seq_list[:,0]+1
	b1 = np.where(len1>thresh1)[0]
	seq_list = id1[seq_list[b1]]
	id_vec1 = np.zeros(sample_num,dtype=bool)
	for t_seq in seq_list:
		pos1, pos2 = t_seq[0], t_seq[1]+1
		id_vec1[pos1:pos2] = 1

	select_id = np.where(id_vec1>0)[0]
	#t_chrom, t_attention1 = chrom[select_id], attention1[select_id]

	data2 = data1.loc[select_id,:]
	chrom_vec, rank1, rank1_1 = select_region3_sub('temp1.txt',data2)
	attention2[select_id] = rank1
	attention2[id1_centromere] = -2

	data1['Q1.adj'] = attention2[:,0]
	data1['Q2.adj'] = attention2[:,1]

	data1.to_csv('temp2.txt',index=False,sep='\t')
	# data2 = data1.loc[select_id,:]
	print(len(select_id),sample_num)

	return data1, select_id

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

# predicton on deleted regions
def mouse_region_test_1(filename_list1,filename2,region_list,region_filename='',sample_num=2000,
						thresh_vec=[0.9,0.95,0.975,0.99,0.995],thresh=0.05,thresh_fdr=0.05,
						sel_idx=[]):
	
	data2 = []
	dict1 = dict()
	for vec1 in filename_list1:
		filename1, celltype_id = vec1
		score1, id1, score_ori1, data2 = mouse_region_test_2(filename1,filename2,celltype_id,region_filename='',sample_num=2000,
							thresh_vec=thresh_vec,thresh=thresh,thresh_fdr=thresh_fdr,
							sel_idx=sel_idx,region_data=data2)

		# mouse_region_test_2_1(run_idList,filename_list,filename2,
		# 						region_filename='',sample_num=2000,thresh_vec=[0.9,0.95,0.975,0.99,0.995])
		dict1[celltype_id] = [score1,id1]

	region_list = np.asarray(region_list)
	region_chrom, region_start, region_stop = region_list[:,0], region_list[:,1], region_list[:,2]
	chrom_vec = np.unique(region_num)

	filename_1 = filename_list1[0][0]
	pred1 = pd.read_csv(filename_1,sep='\t')
	t_chrom1, t_start1, t_stop1, t_serial1 = np.asarray(pred1['chrom']), np.asarray(pred1['start']), np.asarray(pred1['stop']), np.asarray(pred1['serial'])
	t_signal1, t_pred_signal1 = np.asarray(pred1['signal']), np.asarray(pred1['predicted_signal'])

	filename_2 = filename_list1[1][0]
	pred2 = pd.read_csv(filename_2,sep='\t')
	t_chrom2, t_start2, t_stop2, t_serial2 = np.asarray(pred2['chrom']), np.asarray(pred2['start']), np.asarray(pred2['stop']), np.asarray(pred2['serial'])
	t_signal2, t_pred_signal2 = np.asarray(pred2['signal']), np.asarray(pred2['predicted_signal'])

	bin_size = t_stop1[1]-t_start1[1]
	tol = bin_size*5
	chrom_num_region = len(chrom_vec)
	local_score_vec = np.zeros((chrom_num_region,5))
	thresh1, thresh2 = -0.1, 0.1

	for i in range(chrom_num_region):
		t_region_chrom = chrom_vec[i]
		b1 = np.where(region_chrom==t_region_chrom)[0]
		t_region_start, t_region_stop = region_start[b1], region_stop[b1]
		start1, stop1 = t_region_start[0]-tol, t_region_stop[-1]+tol

		id_1 = np.where((t_chrom1==t_region_chrom)&(t_start1<stop1)&(t_stop1>start1))[0]
		signal1, pred_signal1 = t_signal1[id_1], t_pred_signal1[id_1]
		serial1 = t_serial1[id_1]
		local_score1 = score_2a(signal1, pred_signal1)

		id_2 = np.where((t_chrom2==t_region_chrom)&(t_start2<stop1)&(t_stop2>start1))[0]
		signal2, pred_signal2 = t_signal1[id_2], t_pred_signal1[id_2]
		serial2 = t_serial2[id_2]
		local_score2 = score_2a(signal2, pred_signal2)

		local_score_vec[i,0] = (np.mean(pred_signal2)>thresh2)&(np.mean(pred_signal1)<thresh1)

		serial_1 = np.intersect1d(serial1,serial2)
		b_1 = mapping_Idx(serial1,serial_1)
		b_2 = mapping_Idx(serial2,serial_1)
		difference1 = signal2[b_2]-signal1[b_1]
		difference2 = pred_signal2[b_2]-pred_signal1[b_1]
		id_3 = np.argmax(difference1)
		id_5 = np.argmax(difference2)
		l1 = np.where((pred_signal2[b_2]>thresh2)&(pred_signal1[b_1]<thresh1))[0]
		# local_score_vec[i,2] = difference2[id_5]
		local_score_vec[i,1] = len(l1)/len(serial_1)
		local_score_vec[i,2] = np.mean(difference2)/np.mean(difference1)
		t1, t2 = b_1[id_5], b_2[id_5] 
		local_score_vec[i,3:5] = [pred_signal1[t1], pred_signal2[t2]]

	region_num1 = len(region_list)
	chrom2, start2, stop2 = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop'])

	score1, id1 = dict1[2]
	id2, id2_fdr = id1
	local_score_vec2 = np.zeros((region_num1,2))
	for i in range(region_num1):
		t_region = region_list[i]
		t_region_chrom, t_region_start, t_region_stop = t_region
		b1 = np.where((chrom2==t_region_num)&(start2<t_region_stop)&(stop2>t_region_start))[0]
		local_score_vec2[i,0] = np.sum(id2[b1])>0
		local_score_vec2[i,1] = np.sum(id2_fdr[b1])>0

	return local_score_vec, local_score_vec2

# prediction on ERCE regions
def mouse_region_test_2(filename1,filename2,celltype_id,region_filename='',sample_num=2000,
						thresh_vec=[0.9,0.95,0.975,0.99,0.995],thresh=0.05,thresh_fdr=0.05,
						sel_idx=[],region_data=[]):

	score2, score2_fdr, score1, data2 = compare_with_regions_random3(filename1,filename2,celltype_id,region_filename=region_filename,
										tol=2,sample_num=sample_num,thresh_vec=thresh_vec,region_data=region_data)

	b1 = np.where(score2[:,0]>=0)[0]
	region_num1 = len(b1)
	t_score2 = score2[b1,2:]
	t_score2_fdr = score2_fdr[b1,2:]

	id1 = (t_score2<thresh)
	id2 = (t_score2_fdr<thresh_fdr)

	if sel_idx==[]:
		sel_idx = list(range(t_score2.shape[1]))

	s1 = np.sum(id1[:,sel_idx],axis=1)
	s2 = np.sum(id2[:,sel_idx],axis=1)

	ratio1 = np.sum(s1>0)/region_num1
	ratio2 = np.sum(s2>0)/region_num1

	return (ratio1, ratio2), (b1[id1], b1[id2]), (score2, score2_fdr, score1), data2

# prediction on ERCE regions
def mouse_region_test_3(run_id1,filename1,filename2,thresh_vec=[0.5,0.9]):

	data1, vec1 = mouse_region_test_2_2([run_id1],[filename1],filename2,tol=2,thresh_vec=[0.9, 0.95],save_mode=0)

	region_num = data1.shape[0]
	thresh_num = len(thresh_vec)
	cnt_vec = np.zeros((region_num,thresh_num),dtype=np.int8)
	for i in range(thresh_num):
		cnt_vec[:,i] = data1['thresh%s'%(thresh_vec[i])]
	
	cnt_vec1 = (cnt_vec>0)
	ratio = np.sum(cnt_vec1,axis=0)/region_num

	return ratio

# prediction on ERCE rgions
def mouse_region_test_2_1_sub(run_idList,data2,mtx1,mtx,type_id1,annot1,thresh):

	t_mtx = mtx[:,:,2:]
	mask = (t_mtx<thresh)&(t_mtx>=0)

	t1 = np.sum(mask,axis=2)
	# b1 = np.where(t_mtx[:,:,0]>=0)[0]
	# region_num1 = len(b1)
	region_num1 = data2.shape[0]
	test_num = len(run_idList)

	print(region_num1,mask.shape)

	t_mtx1 = (t1>0)
	data_1 = np.sum(t_mtx1,axis=1)	# the number of tests that select the region
	data_2 = np.sum(t_mtx1,axis=0)	# regions selected by each test
	data_3 = np.sum(t1,axis=1)

	ratio2 = data_2/region_num1
	ratio1 = data_1/test_num

	id1 = np.argsort(-ratio1)
	id2 = np.argsort(-ratio2)
	ratio1_sorted, ratio2_sorted = ratio1[id1], ratio2[id2]
	print(ratio1_sorted, ratio2_sorted)

	run_idList = np.asarray(run_idList)

	fields = ['chrom','start','stop']+list(run_idList[:,0])
	fields1 = ['chrom','start','stop']+list(run_idList[:,0])+['pvalue_%d'%(i) for i in run_idList[:,0]]

	# data2_1 = data2.copy()
	data2_1 = pd.DataFrame(columns=fields)
	# print(data2.shape)
	# print(list(data2))
	data2_1['chrom'], data2_1['start'], data2_1['stop'] = data2[0], data2[1], data2[2]

	data2_2 = pd.DataFrame(columns=fields1)
	data2_2['chrom'], data2_2['start'], data2_2['stop'] = data2[0], data2[1], data2[2]

	num1 = len(fields)
	for i in range(num1-3):
		data2_1[fields[i+3]] = t1[:,i]
	data2_1['num'] = data_1
	# data2_1['max'] = score1

	t_mtx = mtx1[:,:,2]
	t_mtx_score = np.min(mtx[:,:,2:],axis=-1)
	t_mtx_1 = np.hstack((t_mtx,t_mtx_score))

	num1 = len(fields1)
	for i in range(num1-3):
		data2_2[fields1[i+3]] = t_mtx_1[:,i]

	data2_1['max'] = np.max(t_mtx,axis=1)
	data2_1['pvalue'] = np.min(t_mtx_score,axis=1)
	
	# np.savetxt('test1_%d.1.%d.txt'%(type_id1,type_id2),data_1,fmt='%d',delimiter='\t')
	# np.savetxt('test1_%d.2.%s.txt'%(type_id1,annot1),data_2,fmt='%d',delimiter='\t')

	id1 = np.argsort(-data_2)
	run_idList = np.asarray(run_idList)
	run_idList1 = run_idList[id1]
	t_mtx_score1 = np.mean(t_mtx_score,axis=0)
	t_mtx_score2 = np.median(t_mtx_score,axis=0)

	fields = ['run_id','celltype','method','num_region','min_pvalue(mean)','min_pvalue(median)']
	data2_3 = pd.DataFrame(columns=fields)
	data2_3['run_id'], data2_3['celltype'], data2_3['method'] = run_idList1[:,0], run_idList1[:,1], run_idList1[:,2]
	data2_3['num_region'], data2_3['min_pvalue(mean)'], data2_3['min_pvalue(median)'] = data_2[id1], t_mtx_score1[id1], t_mtx_score2[id1]
	
	data2_3.to_csv('test1_%d.3.%s.txt'%(type_id1,annot1),index=False,sep='\t')
	data2_1.to_csv('test1_%d.1.%s.txt'%(type_id1,annot1),index=False,sep='\t')
	data2_2.to_csv('test1_%d.2.%s.txt'%(type_id1,annot1),index=False,sep='\t')

	dict1 = {'mtx':mtx,'mtx1':mtx1,'data2':data2}
	filename = 'test1_%d.%s.npy'%(type_id1,annot1)
	np.save(filename,dict1,allow_pickle=True)

	temp1 = np.load(filename,allow_pickle=True)
	temp1 = temp1[()]
	print(temp1.keys())
	print(temp1['mtx1'].shape)

	return True

# prediction on ERCE regions
def mouse_region_test_2_1(run_idList,filename_list,filename2,
							region_filename='',
							sample_num=2000,thresh_vec=[0.9,0.95,0.975,0.99,0.995],
							type_id2=1,
							tol=2,
							thresh_fdr=0.1,
							quantile_vec=['Q1','Q2']):

	thresh = 0.05
	test_num = len(run_idList)
	cnt1 = 0

	# find chromosomes with estimation
	data1 = pd.read_csv(filename_list[0],sep='\t')
	chrom1 = np.asarray(data1['chrom'])
	data2 = pd.read_csv(filename2,header=None,sep='\t')
	colnames2 = list(data2)
	chrom2 = np.asarray(data2[colnames2[0]])
	# serial1 = find_serial(chrom2,chrom_num=len(np.unique(chrom1)))
	# print(data2.shape,len(serial1))
	# data2 = data2.loc[serial1,:]
	run_idList = np.asarray(run_idList)

	for (run_id, filename1) in zip(run_idList, filename_list):
		run_id1, type_id1, method1 = run_id[0], run_id[1], run_id[2]
		quantile_vec1 = quantile_vec
		# if method1 in [53,55,57]:
		# 	quantile_vec1 = ['Q1_1','Q1_2']

		score2, score2_fdr, score1, data2 = compare_with_regions_random3(filename1,filename2,type_id1,region_filename='',
										tol=tol,sample_num=sample_num,type_id=1,
										thresh_vec=thresh_vec,thresh_fdr=thresh_fdr,region_data=data2,
										quantile_vec1 = quantile_vec1)
		t_score2 = score2[:,2:]
		print(run_id,type_id1,np.median(t_score2))
		region_num = data2.shape[0]
		if cnt1==0:
			sel_num = score2.shape[1]
			# mask = np.zeros((region_num,test_num,sel_num),dtype=np.int8)
			mtx = -np.ones((region_num,test_num,sel_num),dtype=np.float32)
			mtx_fdr = mtx.copy()
			mtx1 = -np.ones((region_num,test_num,sel_num),dtype=np.float32)

		print(mtx.shape,score2.shape)
		mtx[:,cnt1,:] = score2
		mtx_fdr[:,cnt1,:] = score2_fdr

		mtx1[:,cnt1,:] = score1[:,0:sel_num]
		cnt1 += 1

	# annot1 = 'pvalue_%d_tol%d_1'%(sample_num,tol)
	annot1 = 'pvalue_%d_tol%d_3_2'%(sample_num,tol)
	mouse_region_test_2_1_sub(run_idList,data2,mtx1,mtx,type_id1,annot1,thresh=0.05)

	# annot1 = 'fdr_%d_tol%d_1'%(sample_num,tol)
	annot1 = 'fdr_%d_tol%d_3_2'%(sample_num,tol)
	mouse_region_test_2_1_sub(run_idList,data2,mtx1,mtx_fdr,type_id1,annot1,thresh=thresh_fdr)

	return True

# prediction on ERCE regions
def mouse_region_test_2_2(run_idList,filename_list,filename2,tol=2,thresh_vec=[0.9, 0.95],save_mode=1):

	list1 = []

	# find chromosomes with estimation
	data1 = pd.read_csv(filename_list[0],sep='\t')
	chrom1 = np.asarray(data1['chrom'])
	data2 = pd.read_csv(filename2,header=None,sep='\t')
	colnames2 = list(data2)
	# chrom2 = np.asarray(data2[colnames2[0]])
	# serial1 = find_serial(chrom2,chrom_num=len(np.unique(chrom1)))
	# data2 = data2.loc[serial1,:]
	chrom2, start2, stop2 = np.asarray(data2[colnames2[0]]), np.asarray(data2[colnames2[1]]), np.asarray(data2[colnames2[2]])

	for (run_id, filename1) in zip(run_idList, filename_list):
		run_id1, type_id1 = run_id[0], run_id[1]

		dict2, data1 = mouse_region_test_5(filename1,thresh_vec=thresh_vec,save_mode=1)

		chrom1, start1, stop1 = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop'])
		attention1 = np.asarray(data1['predicted_attention'])
		colnames1 = list(data1)
		colnames_1 = []
		for thresh in thresh_vec:
			colnames_1.extend(['1_%s'%(thresh),'2_%s'%(thresh)])

		peak_region = np.asarray(data1.loc[:,colnames_1])
		thresh_num = len(thresh_vec)
		bin_size = stop1[1]-start1[1]

		region_num = data2.shape[0]
		num_ptype = 2
		mtx1 = np.zeros((region_num,thresh_num*num_ptype*2),dtype=np.int8)
		print(mtx1.shape)
		
		chrom_vec = np.unique(chrom1)
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

			t_chrom1, t_start1, t_stop1, t_attention1 = np.asarray(chrom1[b1]), np.asarray(start1[b1]), np.asarray(stop1[b1]), np.asarray(attention1[b1])
			t_chrom_region, t_start_region, t_stop_region = np.asarray(chrom2[b2]), np.asarray(start2[b2]), np.asarray(stop2[b2])
		
			for l in range(t_num2):

				t_chrom2, t_start2, t_stop2 = t_chrom_region[l], t_start_region[l], t_stop_region[l]

				tol1 = tol
				t_start_2 = max(0,t_start2-tol1*bin_size)
				t_stop_2 = min(t_stop1[-1],t_stop2+tol1*bin_size)
				len1 = (t_stop_2-t_start_2)/bin_size

				b1_ori = np.where((t_start1<t_stop_2)&(t_stop1>t_start_2))[0]
				if len(b1_ori)==0:
					continue

				i1 = b2[l]
				vec1 = []
				b1_ori = b1[b1_ori]
				t_peak_region = peak_region[b1_ori]
				for i in range(thresh_num):
					for c1 in [2*i, 2*i+1]:
						b_1 = np.where(t_peak_region[:,c1]<0)[0]
						b_2 = np.where(t_peak_region[:,c1]!=0)[0]
						temp1 = np.abs(t_peak_region[b_1,c1])
						temp2 = np.abs(t_peak_region[b_2,c1])
						# vec1.extend([len(b_1),len(b_2)])
						vec1.extend([len(np.unique(temp1)),len(np.unique(temp2))])
				mtx1[i1] = vec1

		list1.append([run_id1,mtx1])

	fields = ['chrom','start','stop']	
	num1, num2 = len(run_idList), len(list1)
	vec1 = np.zeros((region_num,thresh_num),dtype=np.int8)
  
	for i in range(num1):
		run_id1 = run_idList[i][0]
		vec2 = []
		for thresh in thresh_vec:
			vec2.extend(['%d.1_%s_1'%(run_id1,thresh),'%d.1_%s_2'%(run_id1,thresh),'%d.2_%s_1'%(run_id1,thresh),'%d.2_%s_2'%(run_id1,thresh)])
		fields.extend(vec2)

	data2_1 = pd.DataFrame(columns=fields)
	data2_1['chrom'], data2_1['start'], data2_1['stop'] = chrom2, start2, stop2 

	for i in range(num1):
		run_id1, mtx1 = list1[i]
		vec2 = []
		for thresh in thresh_vec:
			vec2.extend(['%d.1_%s_1'%(run_id1,thresh),'%d.1_%s_2'%(run_id1,thresh),'%d.2_%s_1'%(run_id1,thresh),'%d.2_%s_2'%(run_id1,thresh)])
		num3 = len(vec2)
		for t1 in range(num3):
			data2_1[vec2[t1]] = mtx1[:,t1]
		
		sel_num1 = 4
		for t2 in range(thresh_num):
			s1 = np.sum(mtx1[:,t2*sel_num1:(t2+1)*sel_num1],axis=1)
			temp1[:,t2] += (s1>0)

	for i in range(thresh_num):
		data2_1['thresh%s'%(thresh_vec[i])] = vec1[:,i]
	
	if save_mode==1:
		# data2_1.to_csv('test1_%d.5.txt'%(type_id1),index=False,sep='\t')
		data2_1.to_csv('test1_%d.5.1.txt'%(type_id1),index=False,sep='\t')

	return data2_1, vec1

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

def mouse_region_test_5(filename1,thresh_vec=[0.5,0.9],save_mode=0):

	data1 = pd.read_csv(filename1,sep='\t')
	chrom, start, stop, serial = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop']), np.asarray(data1['serial'])
	attention1, attention1_chrom = np.asarray(data1['predicted_attention']), np.asarray(data1['Q2'])

	chrom_vec = np.unique(chrom)
	dict1, dict2 = dict(), dict()

	colnames = list(data1)
	flag = 0
	for thresh in thresh_vec:
		if not (('1_%s'%(thresh) in colnames) and ('2_%s'%(thresh) in colnames)):
			flag = 1

	if flag==0:
		return dict2, data1

	for thresh in thresh_vec:
		list1, list2 = [], []
		dict1[thresh] = [list1,list2]

	width_thresh = 5
	cnt1 = 0
	print(chrom_vec)
	for chrom_id in chrom_vec:
		b1 = np.where(chrom==chrom_id)[0]
		x = attention1[b1]
		x1 = attention1_chrom[b1]

		t_serial = serial[b1]
		# s1, s2 = np.max(x), np.min(x)
		print(chrom_id,len(x),np.max(x),np.min(x),np.median(x))

		peaks, c1 = find_peaks(x,width=(1,10),plateau_size=(1,10))
		width1 = np.arange(1,11)
		peaks_cwt = find_peaks_cwt(x, width1)

		if len(peaks)>0:
			dict1 = peak_region_search(x,x1,peaks,b1,width_thresh,thresh_vec,dict1,type_id2=0)

		if len(peaks_cwt)>0:
			dict1 = peak_region_search(x,x1,peaks_cwt,b1,width_thresh,thresh_vec,dict1,type_id2=1)

		cnt1+=1

	sample_num = len(chrom)
	thresh_num = len(thresh_vec)
	label = np.zeros((sample_num,thresh_num,2),dtype=np.int64)
	print(dict1.keys())

	for l in range(thresh_num):
		thresh = thresh_vec[l]
		list1, list2 = dict1[thresh]

		list1 = np.asarray(list1)
		list2 = np.asarray(list2)
		print(len(list1),len(list2))

		n1, n2 = list1.shape[0], list2.shape[0]
		serial1, serial2 = list1[:,0], list2[:,0]
		id1, id2 = np.argsort(serial1), np.argsort(serial2)
		list1, list2 = list1[id1], list2[id2]

		len1 = list1[:,2]-list1[:,1]
		len2 = list2[:,2]-list2[:,1]
		print(n1,n2,list1.shape,list2.shape,np.max(len1),np.min(len1),max(len2),np.min(len2))

		id_1,id_2 = list1[:,0], list2[:,0]
		data_1 = data1.loc[id_1,colnames]
		data_1['start_1'], data_1['stop_1'] = start[list1[:,1]], stop[list1[:,2]]
		data_2 = data1.loc[id_2,colnames]
		data_2['start_1'], data_2['stop_1'] = start[list2[:,1]], stop[list2[:,2]]
		dict2[thresh] = [data_1,data_2]

		for i in range(n1):
			s1, s2 = list1[i,1:3]
			label[s1:(s2+1),l,0] = i+1
		label[id_1,l,0] = -np.arange(1,n1+1)

		for i in range(n2):
			s1, s2 = list2[i,1:3]
			label[s1:(s2+1),l,1] = i+1
		label[id_2,l,1] = -np.arange(1,n2+1)

		data1['1_%s'%(thresh)] = label[:,l,0]
		data1['2_%s'%(thresh)] = label[:,l,1]

		# np.savetxt('%s_thresh1_%s.txt'%(filename1,thresh),list1,fmt='%d',delimiter='\t')
		# np.savetxt('%s_thresh2_%s.txt'%(filename1,thresh),list2,fmt='%d',delimiter='\t')

	# np.savetxt('test_label.1.txt',label[:,:,0],fmt='%d',delimiter='\t')
	# np.savetxt('test_label.2.txt',label[:,:,1],fmt='%d',delimiter='\t')
	if save_mode==1:
		data1.to_csv(filename1,index=False,sep='\t')

	return dict2, data1

# prediction evaluation on human cell lines
def prediction_test_3():

	x = 3

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

# sample regions randomly to compare with elements
def compare_with_regions_random2(filename1,filename2,output_filename,output_filename1,
				output_filename2, region_filename, thresh_1 = 0, tol=1, sample_num=200, type_id=1, label_name='label'):
	
	# config1 = {'thresh1':0.5}
	config1 = {'thresh1':thresh_1}
	data1, data2 = compare_with_regions_pre(filename1,filename2,output_filename, tol,label_name, 
												save_mode=0,region_data=[],select_id=1,config=config1)

	print(config1['thresh1'],data1.shape,data2.shape)
	print(np.median(data1['signal']))
	# print(np.median(data1['signal']),np.min(data2['min']),np.min(data2['max']))

	print('signal')

	region_list = []
	# if os.path.exists(region_filename)==True:
	# 	region_list1 = pd.read_csv(region_filename,sep='\t')
	# else:
	# 	label_name = 'label'
	# 	region_list1 = query_region2(data1,label_name=label_name)
	# 	region_list1.to_csv(region_filename,index=False,sep='\t')

	label_name = 'label'
	region_list1 = query_region2(data1,label_name=label_name)
	region_list1.to_csv(region_filename,index=False,sep='\t')

	region_chrom, pair1 = np.asarray(region_list1['chrom']), np.asarray(region_list1[['pos1','pos2']])

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
	vec1, vec1_1 = [], []

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

			# print(t_stop,t_start)
			region_len_ori = t_stop-t_start
			region_len_ori1 = (region_len_ori)/bin_size
			# tol1 = min(int(region_len_ori1*0.25),tol)
			if region_len_ori<10:
				tol1 = 2
			else:
				tol1 = tol
			t_start = max(0,t_start-tol1*bin_size)
			t_stop = min(t_stop1[-1],t_stop+tol1*bin_size)

			b1_ori = np.where((t_start1<t_stop)&(t_stop1>t_start))[0]
			if len(b1_ori)==0:
				continue

			b1_ori = b2[b1_ori]
			# s1 = max(0,b1_ori[0]-tol)
			# s2 = min(chrom_size1,b1_ori[0]+tol+1)
			# b1 = list(range(s1,s2))
			# b1 = np.where((chrom1==t_chrom)&(start1>=t_start)&(stop1<=t_stop))[0]
			label1[b1_ori] = 1+i1
		
			for l1 in range(sel_num):
				id2 = 2*l1
				score1[i1,id2:(id2+2)] = [np.max(attention1[b1_ori,l1]),np.mean(attention1[b1_ori,l1])]

			# randomly sample regions
			region_len = t_stop-t_start
			t_chrom_size = len(b2)
			# sample_num = 200
			position = (t_chrom, t_start, t_stop)
			b_1 = np.where(region_chrom==t_chrom)[0]
			region_list = pair1[b_1]
			vec2 = compare_with_regions_sub2(t_chrom1,t_start1,t_stop1,t_attention1,
									sample_num,position,region_list,bin_size,tol)
			# vec2 = compare_with_regions_sub2(t_chrom1,t_start1,t_stop1,t_attention1,
			# 							sample_num,region_len,t_chrom_size,bin_size,tol)
				
			vec3 = []
			# print(vec2.shape)
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
			vec1_1.extend([t_chrom]*sample_num1)

			# if i%100==0:
			# 	print(i,score1[i],len(vec2))
			if i1%1000==0:
				print(i1,score1[i1],len(vec2),vec2.shape)
			# if l>10:
			# 	break

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
	fields = ['chrom','region_id','start','stop']
	data_1 = pd.DataFrame(columns=fields)
	data_1[fields[0]] = vec1_1
	for i in range(0,3):
		data_1[fields[i+1]] = np.int64(vec1[:,i])
	for i in range(3,num1):
		data_1['sel%d'%(i-2)] = vec1[:,i]
	data_1.to_csv(output_filename2,index=False,sep='\t')

	return True

def compare_with_regions_signal(filename1,filename2):

	data1 = pd.read_csv(filename1,sep='\t')
	colnames1 = list(data1)
	chrom1, start1, stop1 = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop'])
	serial1, signal1 = np.asarray(data1['serial']), np.asarray(data1['signal'])
	bin_size = stop1[1]-start1[1]

	data2 = pd.read_csv(filename2,header=None,sep='\t')
	colnames2 = list(data2)
	col1, col2, col3 = colnames2[0], colnames2[1], colnames2[2]
	chrom2, start2, stop2 = np.asarray(data2[col1]), np.asarray(data2[col2]), np.asarray(data2[col3])
	fields = ['length','num','max','min','mean','median']

	region_num = len(chrom2)
	value1 = -np.ones((region_num,6),dtype=np.float32)
	for i in range(region_num):
		t_chrom2, t_start2, t_stop2 = chrom2[i], start2[i], stop2[i]
		b = np.where((chrom1==t_chrom2)&(start1<t_stop2)&(stop1>t_start2))[0]
		t_signal = signal1[b]
		region_len = (t_stop2-t_start2)/bin_size
		value1[i,0:2] = [region_len,len(b)]
		if len(b)>0:
			value1[i,2:] = [np.max(t_signal),np.min(t_signal),np.mean(t_signal),np.median(t_signal)]
		if i%1000==0:
			print(t_chrom2,t_start2,t_stop2,value1[i])

	num1 = len(fields)
	for i in range(num1):
		data2[fields[i]] = value1[:,i]

	data2.to_csv(filename2,index=False,header=False,sep='\t')

	return True

def compare_with_regions_local(filename1,filename1_1,filename2,region_filename,run_idList,output_filename,tol=2,thresh=5):

	data1 = pd.read_csv(filename1,sep='\t')
	colnames1 = list(data1)
	chrom, start, stop = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop'])

	data1_1 = pd.read_csv(filename1_1,sep='\t')

	fields = ['chrom','start','stop','length','num','max','min','mean','median']
	data2 = pd.read_csv(filename2,sep='\t',names=fields)
	colnames2 = list(data2)
	col1, col2, col3 = colnames2[0], colnames2[1], colnames2[2]
	chrom2, start2, stop2 = np.asarray(data2[col1]), np.asarray(data2[col2]), np.asarray(data2[col3])

	sel_column_vec = ['predicted_attention','Q2']
	sel_id = 1
	sel_column = sel_column_vec[sel_id]
	# chrom	start	stop	serial	signal	label	label_tol2
	region_list = pd.read_csv(region_filename,sep='\t')
	region_chrom, region_start, region_stop, region_serial = np.asarray(region_list['chrom']),np.asarray(region_list['start']),np.asarray(region_list['stop']), np.asarray(region_list['serial'])
	region_label = np.asarray(region_list['label_tol%d'%(tol)])
	region_label_vec = np.unique(region_label)
	region_label_vec = region_label_vec[region_label_vec>0]
	sample_num = len(region_label)
	bin_size = region_stop[1]-region_start[1]
	tol1 = tol*bin_size

	sel_column_vec2 = ['1_0.9','1_0.95','2_0.9','2_0.95']
	num1 = len(sel_column_vec2)
	data_2 = data2.copy()
	region_num = data_2.shape[0]
	id_vec1 = dict()
	# mask1 = np.zeros((region_num,num1+2),dtype=np.int32)
	sel_num = num1+2
	base1 = 2**num1
	for t_runid in run_idList:
		run_id1, type_id2, method1 = t_runid
		if type_id2>2:
			feature_id1 = 5
		else:
			feature_id1 = -1
		filename1 = './mnt/yy3/test_vec2_%d_%d_[%d].1_0.1.txt'%(run_id1,method1,feature_id1)
		if os.path.exists(filename1):
			data3 = pd.read_csv(filename1,sep='\t')
			chrom1, start1, stop1, serial1 = np.asarray(data3['chrom']), np.asarray(data3['start']), np.asarray(data3['stop']), np.asarray(data3['serial'])
			if np.sum(chrom1!=region_chrom)>0 or np.sum(start1!=region_start)>0:
				print('error! compare with regions local',len(chrom1),len(region_chrom))
				return

			predicted_attention1 = np.asarray(data3[sel_column])
			peak_vec1 = np.asarray(data3.loc[:,sel_column_vec2])
			print(peak_vec1.shape)
			id_vec1[run_id1] = dict()
			for i in range(num1):
				id2 = np.where(peak_vec1[:,i]<0)[0]
				print(run_id1,len(id2),len(id2)/sample_num)
				id_vec1[run_id1][sel_column_vec2[i]] = id2

			# mask = mask1.copy()
			mask = np.zeros((region_num,num1+2),dtype=np.int32)
			max_value = np.zeros(region_num,dtype=np.float32)
			mask[:,-2] = data1[str(run_id1)]
			mask[:,-1] = data1_1[str(run_id1)]
			print(region_label_vec)
			for label1 in region_label_vec:
				# print(label1)
				b1 = np.where(region_label==label1)[0]
				t_chrom, t_start, t_stop = chrom1[b1],start1[b1],stop1[b1]
				t_peak_vec = peak_vec1[b1]
				t_chrom1 = t_chrom[0]
				pos1, pos2 = t_start[0], t_stop[-1]
				id1 = label1-1
				# b_1 = ((t_chrom==chrom2[id1])&(t_stop>start2[id1])&(t_start<stop2[id1]))
				b_1 = ((chrom2[id1]==t_chrom1)&(start2[id1]<pos2+tol1)&(stop2[id1]>pos1-tol1))

				# n1, n2 = np.sum(b_1), len(b1)
				if b_1==False:
					print('error!',label1,chrom2[id1],start2[id1],stop2[id1])
					b2 = np.where((chrom2==t_chrom)&(start2<pos2)&(stop2>pos1))[0]
					id1 = b2[0]
				
				max_value[id1] = np.max(predicted_attention1[b1])
				for i in range(num1):
					b = np.where(t_peak_vec[:,i]<0)[0]
					if len(b)>0:
						mask[id1,i] = len(b)
					else:
						t_column = sel_column_vec2[i]
						id2 = id_vec1[run_id1][t_column]
						id_2 = np.where(region_chrom==t_chrom1)[0]
						id2 = np.intersect1d(id2,id_2)
						# print(t_column,len(id2))
						t_region_serial = region_serial[id2]
						distance1 = np.hstack((np.abs(serial1[b1[0]]-t_region_serial),np.abs(serial1[b1[-1]]-t_region_serial)))
						idx = np.hstack((id2,id2))
						t1 = np.argsort(distance1)
						t_distance = distance1[t1[0]]
						mask[id1,i] = -t_distance

				if id1%1000==0:
					print(label1,chrom2[id1],start2[id1],stop2[id1],max_value[id1],mask[id1])

			t_label = np.zeros(region_num,dtype=np.int32)
			mask2 = mask[:,0:-2]
			t1 = (mask2>0)
			cnt1 = np.sum(t1,axis=1)
			t2 = (mask2<0)
			distance = np.zeros((region_num,num1),dtype=np.int32)
			for i in range(num1):
				t_id1 = t2[:,i]
				distance[t_id1,i] = -mask2[t_id1,i]

			vec1 = np.zeros(region_num,dtype=np.int32)
			for i in range(num1):
				b2 = (distance[:,i]<thresh)&(distance[:,i]>0)
				t_label[b2] = base1+i+2

				vec1 += (2**i)*t1[:,i]

			b2 = (vec1>0)
			t_label[b2] = vec1[b2]
			
			for i in range(num1,sel_num):
				b1 = (mask[:,i]>0)
				t_label[b1] = base1+i-num1

			id1, id_1 = np.where(t_label==base1+1)[0], np.where(t_label>0)[0]
			id2, id3 = np.where(t_label[id_1]<base1)[0], np.where(t_label>base1+1)[0]
			id2 = id_1[id2]
			id_2 = np.where(t_label<=0)[0]
			print(run_id1,len(id1),len(id2),len(id3),len(id_2),cnt1)

			fields = ['%d_max'%(run_id1),'%d_1_0.9'%(run_id1),'%d_2_0.9'%(run_id1),'%d_1_0.95'%(run_id1),'%d_2_0.95'%(run_id1)]
			data_2['%d_max'%(run_id1)] = max_value
			for i in range(num1):
				t_col = '%d_%s'%(run_id1,sel_column_vec2[i])
				data_2[t_col] = mask[:,i]
			data_2['%d_pvalue'%(run_id1)] = mask[:,-2]
			data_2['%d_fdr'%(run_id1)] = mask[:,-1]
			data_2['%d_label'%(run_id1)] = t_label

	data_2.to_csv('%s_tol%d.txt'%(output_filename,tol),index=False,sep='\t')
	np.save('%s_tol%d.npy'%(output_filename,tol),id_vec1,allow_pickle=True)

	return True

def compare_with_regions_local_sub1(filename2,region_filename,run_idList,output_filename,tol=2,thresh=5,thresh_vec=[],type_id_1=0):

	# data1 = pd.read_csv(filename1,sep='\t')
	# colnames1 = list(data1)
	# chrom, start, stop = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop'])

	fields = ['chrom','start','stop','length','num','max','min','mean','median']
	data2 = pd.read_csv(filename2,sep='\t',names=fields)
	colnames2 = list(data2)
	col1, col2, col3 = colnames2[0], colnames2[1], colnames2[2]
	chrom2, start2, stop2 = np.asarray(data2[col1]), np.asarray(data2[col2]), np.asarray(data2[col3])

	sel_column_vec = ['predicted_attention','Q2']
	sel_id = 1
	sel_column = sel_column_vec[sel_id]
	# chrom	start	stop	serial	signal	label	label_tol2
	region_list = pd.read_csv(region_filename,sep='\t')
	region_chrom, region_start, region_stop, region_serial = np.asarray(region_list['chrom']),np.asarray(region_list['start']),np.asarray(region_list['stop']), np.asarray(region_list['serial'])
	if tol>0:
		region_label = np.asarray(region_list['label_tol%d'%(tol)])
	else:
		region_label = np.asarray(region_list['label'])

	region_signal = np.asarray(region_list['signal'])
	print(np.max(region_signal))
	if len(thresh_vec)==0:
		if type_id_1==0:
			thresh_vec2 = np.array([0.6,0.7,0.75,0.8,0.9])
			if np.median(region_signal)> 0.25:
				thresh_vec = list(thresh_vec2)
			else:
				thresh_vec = list(2*thresh_vec2-1)
		else:
			thresh_vec = [0.9,0.95,0.975,0.98,0.985,0.99,0.9925,0.995,0.9975,0.999,0.9995,0.9999]

	region_label_vec = np.unique(region_label)
	region_label_vec = region_label_vec[region_label_vec>0]
	sample_num = len(region_label)
	bin_size = region_stop[1]-region_start[1]
	tol1 = tol*bin_size

	sel_column_vec2 = ['1_0.9','1_0.95','2_0.9','2_0.95']
	num1 = len(sel_column_vec2)
	data_2 = data2.copy()
	region_num = data_2.shape[0]
	id_vec1 = dict()
	# mask1 = np.zeros((region_num,num1+2),dtype=np.int32)
	sel_num = num1+2
	base1 = 2**num1

	ratio_vec = []
	for t_runid in run_idList:
		run_id1, type_id2, method1 = t_runid
		if type_id2>2:
			feature_id1 = 5
		else:
			feature_id1 = -1
		filename1 = './mnt/yy3/test_vec2_%d_%d_[%d].1_0.1.txt'%(run_id1,method1,feature_id1)
		if os.path.exists(filename1):
			data3 = pd.read_csv(filename1,sep='\t')
			chrom1, start1, stop1, serial1 = np.asarray(data3['chrom']), np.asarray(data3['start']), np.asarray(data3['stop']), np.asarray(data3['serial'])
			if np.sum(chrom1!=region_chrom)>0 or np.sum(start1!=region_start)>0:
				print('error! compare with regions local',len(chrom1),len(region_chrom))
				return

			predicted_attention1 = np.asarray(data3[sel_column])

			peak_vec1 = np.asarray(data3.loc[:,sel_column_vec2])
			print(peak_vec1.shape)
			id_vec1[run_id1] = dict()
			for i in range(num1):
				id2 = np.where(peak_vec1[:,i]<0)[0]
				print(run_id1,len(id2),len(id2)/sample_num)
				id_vec1[run_id1][sel_column_vec2[i]] = id2

			id1 = np.where(region_label>0)[0]
			region_num1 = len(id1)	# number of regions
			sample_num1 = len(chrom1)	
			# len1 = (stop1[id1]-start1[id1])/bin_size
			# len2 = (stop1-start1)/bin_size
			expected_ratio = region_num1/sample_num1
			# expected_ratio = np.sum(len1)/np.sum(len2)

			if type_id_1==1:
				for thresh_1 in thresh_vec:
					id2 = np.where(predicted_attention1>thresh_1)[0]
					print(len(id2)/sample_num1)
					id3 = np.intersect1d(id1,id2) # peaks in init_zones

					n1, n2 = len(id2), len(id3)
					t_region_label = region_label[id3]
					t_region_num = len(np.unique(t_region_label))
					ratio_vec.append([run_id1,0,thresh_1,t_region_num,n2,n1,n2/(n1+1e-12),expected_ratio])

			else:
				for i in range(num1):
					id2 = np.where(peak_vec1[:,i]<0)[0]
					id3 = np.intersect1d(id1,id2) # peaks in init_zones
					n1, n2 = len(id2), len(id3)
					t_region_label = region_label[id3]
					t_region_num = len(np.unique(t_region_label))
					ratio_vec.append([run_id1,i,0,t_region_num,n2,n1,n2/(n1+1e-12),expected_ratio])

					for thresh_1 in thresh_vec:
						early_id = np.where(region_signal>thresh_1)[0]
						early_id1 = np.where(region_label[early_id]>0)[0]
						# len1 = (stop1[early_id1]-start1[early_id1])/bin_size
						# len2 = (stop1[early_id]-start1[early_id])/bin_size
						# early_expected_ratio = num.sum(len1)/np.sum(len2)
						early_expected_ratio = len(early_id1)/len(early_id)

						early_id2 = np.where(peak_vec1[early_id,i]<0)[0]
						early_id3 = np.intersect1d(early_id1,early_id2)
						early_id2, early_id3 = early_id[early_id2], early_id[early_id3]
						t_region_label = region_label[early_id3]
						t_region_num = len(np.unique(t_region_label))
						n1, n2 = len(early_id2), len(early_id3)
						ratio_vec.append([run_id1,i,thresh_1,t_region_num,n2,n1,n2/(n1+1e-12),early_expected_ratio])

			print(run_id1,ratio_vec[-1])

	ratio_vec = np.asarray(ratio_vec)
	fields = ['run_id','peak_type','thresh','region_num','init_zone_peak','peak_num','ratio','expected_ratio']
	data_3 = pd.DataFrame(columns=fields)
	for i in [0,1,3,4,5]:
		data_3[fields[i]] = np.int64(ratio_vec[:,i])
	
	for i in [2,6,7]:
		data_3[fields[i]] = ratio_vec[:,i]

	data_3.to_csv('%s_tol%d_%s.txt'%(output_filename,tol,str(thresh_1)),index=False,sep='\t')

	return True

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

def compare_with_regions_local_sub2(filename_list1,run_idList,annot_list1,output_filename,type_id1=1):

	params = {
		 'axes.labelsize': 15,
		 'axes.titlesize': 16,
		 'xtick.labelsize':15,
		 'ytick.labelsize':15}
	pylab.rcParams.update(params)

	fig = plt.figure(figsize=(22,12))

	# color_vec = ['steelblue','orange','mediumseagreen','wheat','olive','maroon','khaki','green','blue',
	# 				'#abd9e9','#d7191c','#fdae61','#ffffbf','#abdda4','#2b8']
	# color_vec = ['steelblue','orange','olive','maroon']
	color_vec = ['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac']
	color_vec = ['#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4']

	bars_vec = [['p-value<0.05','FDR<0.1','Max.>0.975','Peak I','Peak II','Near peaks','p-value+peaks','FDR+peaks','p-value+peaks+adj.'],
			['p-value<0.05','FDR<0.1','Max.>0.975','Peak I','Peak II','Near peaks','p-value+peaks','FDR+peaks','p-value+peaks+adj.']]

	cnt = len(run_idList)
	for i in range(cnt):

		run_id = run_idList[i]
		annot1 = annot_list1[i]
		print(run_id)
		
		ratio_vec1, data1, id_vec = compare_with_regions_load_1(filename_list1[i],run_id)

		plt.subplot(1,cnt,i+1)
		
		num2 = 1
		num1 = ratio_vec1.shape[1]
		barWidth = 0.8
		c1 = 0.2
		t_pos1 = np.arange(0,num1*(num2*barWidth+c1),num2*barWidth+c1)

		bars = bars_vec[type_id1]
		# t_pos1 = np.arange(0,num1*(num2*barWidth+c1),num2*barWidth+c1)
		# t_pos1 = np.arange(len(bars))
		# for i in range(0,num2):
		# 	t_pos2 = [x + barWidth*i for x in t_pos1]
		#	plt.bar(t_pos2, ratio_vec1[type_id1], color=color_vec, width=barWidth, 
		#				edgecolor='white')
		print(len(t_pos1),len(ratio_vec1[type_id1]))
		plt.bar(t_pos1, ratio_vec1[type_id1], color=color_vec, width=barWidth, 
						edgecolor='white')

		# height = [3, 12, 5, 18, 45]
		# bars = ('A', 'B', 'C', 'D', 'E')
		# y_pos = np.arange(len(bars))
		# plt.bar(y_pos, height, color = color_vec[0:5])
		# plt.xticks(y_pos, bars)

		# plt.xticks([r + 1.5*barWidth for r in t_pos1], bars)
		plt.xticks(t_pos1, bars, rotation=30)
		plt.ylabel('Percentage of Initiation Zones')
		plt.title(annot1)
	
	plt.savefig(output_filename,dpi=300)

	return True

def compare_with_regions_local_sub2_1(filename_list1,run_idList,annot_list1,output_filename,type_id1=1):

	params = {
		 'axes.labelsize': 15,
		 'axes.titlesize': 16,
		 'xtick.labelsize':15,
		 'ytick.labelsize':15}
	pylab.rcParams.update(params)

	fig = plt.figure(figsize=(22,16))

	# color_vec = ['steelblue','orange','mediumseagreen','wheat','olive','maroon','khaki','green','blue',
	# 				'#abd9e9','#d7191c','#fdae61','#ffffbf','#abdda4','#2b8']
	# color_vec = ['steelblue','orange','olive','maroon']
	color_vec = ['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac']
	color_vec = ['#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4']

	bars_vec = [['p-value<0.05','FDR<0.1','Max.>0.975','Peak I','Peak II','Near peaks','p-value+peaks','FDR+peaks','p-value+peaks+adj.'],
			['p-value<0.05','FDR<0.1','Max.>0.975','Peak I','Peak II','Near peaks','p-value+peaks','FDR+peaks','p-value+peaks+adj.']]

	cnt = len(run_idList)
	for i in range(cnt):

		run_id = run_idList[i]
		print(run_id)
		
		filename1 = filename_list1[i]
		data1 = pd.read_csv(filename1,sep='\t')

		chrom = np.asarray(data1['chrom'])
		signal = np.asarray(data1['max'])

		chrom_vec = []
		chrom_num = 22
		t_filename1 = ''
		vec1 = np.zeros((chrom_num,3))
		for i1 in range(0,chrom_num):
			chrom_id = 'chr%d'%(i1+1)
			chrom_vec.append(chrom_id)
			id1 = np.where(chrom==chrom_id)[0]
			data_1 = data1.loc[id1,:]

			ratio_vec1, data_1, id_vec = compare_with_regions_load_1(t_filename1,run_id,data1=data_1)

			vec1[i1] = ratio_vec1[1,[0,1,6]]
			print(chrom_id,vec1[i1])

		id2 = [0,1,6]
		barWidth = 0.5
		annot_vec = ['I: p-value<0.05','II: FDR<0.1','III: p-value+peaks']

		for l in range(vec1.shape[1]):
			annot1 = annot_list1[l]
			plt.subplot(3,1,l+1)
			t_pos1 = np.arange(chrom_num)
			plt.bar(t_pos1, vec1[:,l], color=color_vec[id2[l]], width=barWidth, 
						edgecolor='white')

			# height = [3, 12, 5, 18, 45]
			# bars = ('A', 'B', 'C', 'D', 'E')
			# y_pos = np.arange(len(bars))
			# plt.bar(y_pos, height, color = color_vec[0:5])
			# plt.xticks(y_pos, bars)

			# plt.xticks([r + 1.5*barWidth for r in t_pos1], bars)
			
			plt.ylabel('Percentage of Initiation Zones')
			if l==2:
				plt.xticks(t_pos1, chrom_vec, rotation=30)
			else:
				plt.xticks([], [])
			plt.title(annot1)
	
	plt.savefig(output_filename,dpi=300)

	return True

def compare_with_regions_local_sub2_2(filename_list1,run_idList,annot_list1,output_filename,type_id1=1):

	params = {
		 'axes.labelsize': 15,
		 'axes.titlesize': 16,
		 'xtick.labelsize':15,
		 'ytick.labelsize':15}
	pylab.rcParams.update(params)

	fig = plt.figure(figsize=(22,16))

	# color_vec = ['steelblue','orange','mediumseagreen','wheat','olive','maroon','khaki','green','blue',
	# 				'#abd9e9','#d7191c','#fdae61','#ffffbf','#abdda4','#2b8']
	# color_vec = ['steelblue','orange','olive','maroon']
	color_vec = ['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac']
	color_vec = ['#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4']

	bars_vec = [['p-value<0.05','FDR<0.1','Max.>0.975','Peak I','Peak II','Near peaks','p-value+peaks','FDR+peaks','p-value+peaks+adj.'],
			['p-value<0.05','FDR<0.1','Max.>0.975','Peak I','Peak II','Near peaks','p-value+peaks','FDR+peaks','p-value+peaks+adj.']]

	cnt = len(run_idList)
	for i in range(cnt):

		run_id = run_idList[i]
		print(run_id)
		
		filename1 = filename_list1[i]
		data1 = pd.read_csv(filename1,sep='\t')
		print(data1.shape)

		chrom = np.asarray(data1['chrom'])
		max_signal = np.asarray(data1['max'])
		id1_ori = np.where(max_signal!=-1)[0]
		# data1 = data1.loc[id1_ori,:]
		print(data1.shape)
		max_signal = max_signal[id1_ori]
		v1 = np.quantile(max_signal,[0,0.25,0.5,0.75,1])
		v1[-1] = v1[-1]+0.01

		vec1 = []
		num1 = len(v1)-1
		t_filename1 = ''
		for l in range(num1):
			id1 = np.where((max_signal>=v1[l])&(max_signal<v1[l+1]))[0]
			print(v1[l],v1[l+1],len(id1),id1)
			data_1 = data1.loc[id1,:]

			ratio_vec1, data_1, id_vec = compare_with_regions_load_1(t_filename1,run_id,data1=data_1)

			vec1.append(ratio_vec1[1,[0,1,6]])
			print(v1[l])

		id2 = [0,1,6]
		barWidth = 0.5
		# annot_vec = ['RT signal: 0-0.25','RT signal: 0.25-0.5','RT signal: 0.5-0.75','RT Signal: 0.75-1']
		annot_vec = ['0-0.25','0.25-0.5','0.5-0.75','0.75-1']

		vec1 = np.asarray(vec1)

		for l in range(vec1.shape[1]):
			annot1 = annot_list1[l]
			plt.subplot(1,3,l+1)
			t_pos1 = np.arange(num1)
			plt.bar(t_pos1, vec1[:,l], color=color_vec[id2[l]], width=barWidth, 
						edgecolor='white')

			# height = [3, 12, 5, 18, 45]
			# bars = ('A', 'B', 'C', 'D', 'E')
			# y_pos = np.arange(len(bars))
			# plt.bar(y_pos, height, color = color_vec[0:5])
			# plt.xticks(y_pos, bars)

			# plt.xticks([r + 1.5*barWidth for r in t_pos1], bars)
			
			plt.ylabel('Percentage of Initiation Zones')
			plt.xticks(t_pos1, annot_vec, rotation=30)
			plt.title(annot1)
	
	plt.savefig(output_filename,dpi=300)

	return True

def compare_with_regions_local_sub2_3(filename1,run_idList,annot_list1,output_filename,type_id1=1):

	params = {
		 'axes.labelsize': 15,
		 'axes.titlesize': 16,
		 'xtick.labelsize':15,
		 'ytick.labelsize':15}
	pylab.rcParams.update(params)

	# color_vec = ['steelblue','orange','mediumseagreen','wheat','olive','maroon','khaki','green','blue',
	# 				'#abd9e9','#d7191c','#fdae61','#ffffbf','#abdda4','#2b8']
	# color_vec = ['steelblue','orange','olive','maroon']
	color_vec = ['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac']
	color_vec = ['#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4']

	bars_vec = [['p-value<0.05','FDR<0.1','Max.>0.975','Peak I','Peak II','Near peaks','p-value+peaks','FDR+peaks','p-value+peaks+adj.'],
			['p-value<0.05','FDR<0.1','Max.>0.975','Peak I','Peak II','Near peaks','p-value+peaks','FDR+peaks','p-value+peaks+adj.']]

	cnt = len(run_idList)
	dict1 = dict()
	dict2 = dict()
	num1 = 5
	sel_idx = [0,1,6]
	for i in range(num1):
		dict2[i] = {0:[],1:[],6:[]}

	for i in range(cnt):

		run_id = run_idList[i]
		print(run_id)
		
		data1 = pd.read_csv(filename1,sep='\t')
		print(data1.shape)

		if i==0:
			chrom = np.asarray(data1['chrom'])
			max_signal = np.asarray(data1['max'])
			id1_ori = np.where(max_signal!=-1)[0]
			# data1 = data1.loc[id1_ori,:]
			print(data1.shape)
			sample_num = len(id1_ori)
			max_signal = max_signal[id1_ori]
			v1 = np.quantile(max_signal,[0,0.25,0.5,0.75,1])
			v1[-1] = v1[-1]+0.01

		num1 = len(v1)-1
		dict1[run_id] = dict()
		t_filename1 = ''
		ratio_vec1, data_1, id_vec = compare_with_regions_load_1(t_filename1,run_id,data1=data1)
		dict1[run_id][num1] = id_vec
		print(ratio_vec1)

		for l in sel_idx:
			t_idvec = dict2[num1][l]
			print(run_id,l,len(t_idvec))
			t_idvec = np.union1d(t_idvec,id_vec[l])
			dict2[num1].update({l:t_idvec})
			print(run_id,l,len(t_idvec))

		cnt_vec1 = np.zeros(num1+1)
		for i1 in range(num1):
			id1 = np.where((max_signal>=v1[i1])&(max_signal<v1[i1+1]))[0]
			print(v1[i1],v1[i1+1],len(id1))
			data_1 = data1.loc[id1,:]

			ratio_vec1, data_1, id_vec = compare_with_regions_load_1(t_filename1,run_id,data1=data_1)
			dict1[run_id][i] = id_vec
			cnt_vec1[i1] = data_1.shape[0]

			for l in sel_idx:
				t_idvec = dict2[i1][l]
				print(run_id,i1,l,len(t_idvec))
				t_idvec = np.union1d(t_idvec,id_vec[l])
				dict2[i1].update({l:t_idvec})
				print(run_id,i1,l,len(t_idvec))

		cnt_vec1[-1] = sample_num
	
	vec1 = np.zeros((num1+1,3))
	for i in range(num1+1):
		for i1 in range(3):
			t_idvec = dict2[i][sel_idx[i1]]
			vec1[i,i1] = len(t_idvec)/cnt_vec1[i]

	print(vec1)
	fig = plt.figure(figsize=(22,16))
	barWidth = 0.5
	# annot_vec = ['RT signal: 0-0.25','RT signal: 0.25-0.5','RT signal: 0.5-0.75','RT Signal: 0.75-1']
	annot_vec = ['0-0.25','0.25-0.5','0.5-0.75','0.75-1','0-1']

	for i in range(vec1.shape[1]):

		annot1 = annot_list1[i]
		plt.subplot(1,3,i+1)
		t_pos1 = np.arange(num1+1)
		plt.bar(t_pos1, vec1[:,i], color=color_vec[sel_idx[i]], width=barWidth, 
						edgecolor='white')
			
		plt.ylabel('Percentage of Initiation Zones')
		plt.xticks(t_pos1, annot_vec, rotation=30)
		plt.title(annot1)
	
	plt.savefig(output_filename,dpi=300)

	return True

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
		# print(encoding.shape)
		# print(encoding[0].shape)
		# list1.append(np.asarray(encoding,dtype=np.int8))
		# list2.append(serial1)
		list1[i] = encoding
		list2[i] = [serial1,n1]
		if i%10000==0:
			print(i,serial1)

	# b1 = np.where(list2>=0)[0]
	# list2 = list2[b1]
	# list1 = list1[b1]
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
	# signal_vec[id1] = aver_value
	# serial_vec[id1] = 1
	# start_idx = id3[-1]+1

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
		# if start_idx>m_idx:
		# 	break
		# id3 = np.where((stop_vec[start_idx:]<=t_stop)&(start_vec[start_idx:]>=t_start))[0]
		# # print("start_idx",t_start,t_stop,start_idx,len(id3))
		# # id3 = np.where((stop_vec<=t_stop)&(start_vec>=t_start))[0]
		# if len(id3)>0:
		# 	id3 = id3 + start_idx
		# 	start_idx = id3[-1] + 1
		# 	# print(count,t_start,t_stop,t_stop-t_start,id3[0],id3[-1],start_vec[id3[0]],stop_vec[id3[-1]],len(id3),len(id3)*200)
		# 	if count%100==0:
		# 		# print(count,t_start,t_stop,len(id3),start_idx,start_vec[id3[0]],stop_vec[id3[-1]])
		# 		print(count,t_start,t_stop,t_stop-t_start,id3[0],id3[-1],start_vec[id3[0]],stop_vec[id3[-1]],len(id3),len(id3)*bin_size)
		# 	# if count>50:
		# 	# 	break
		# 	list1.extend(id3)
		# 	count += 1
		# else:
		# 	count2 += 1

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
	# for i in range(0,num2):
	# 	# print(id1)
	# 	id1 = list2[i]
	# 	t_start, t_stop = start_vec[id1], stop_vec[id1]
	# 	id3 = np.where((start1<t_stop)&(stop1>t_start))[0] # find the overlapping regions
	# 	count1 += 1
	# 	if len(id3)>0:
	# 		t_start1, t_stop1 = start1[id3], stop1[id3]
	# 		flag1 = True
	# 		start_id1 = i + 1
	# 		break

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
	
def merge_data_kmer(filename1, filename2):

	path1 = '/volume01/yy3/seq_data/dl/replication_timing'
	filename_1 = '%s/signal/hg19.human.E_L.EL.log.min0.1.6k.match2ori.bed'%(path1)
	filename1 = '%s/estimate_rt/estimate_rt_hg19.txt'%(path1)
	filename2 = '%s/estimate_rt/estimate_rt_hg19.2.txt'%(path1)

	ref_table = pd.read_csv(filename_1,header=None,sep='\t')
	table1 = pd.read_csv(filename1,sep='\t')
	table2 = pd.read_csv(filename2,sep='\t')

	filename2 = '%s/training2_kmer_hg19.2.npy'%(path1)
	data2 = np.load(filename2)
	
	ref_chrom, ref_serial = np.asarray(ref_table[0]), np.asarray(ref_table[3])
	colnames1 = list(table1)
	chrom1, serial1 = np.asarray(table1[colnames1[0]]), np.asarray(table1[colnames1[3]])
	colnames2 = list(table2)
	chrom2, serial2 = np.asarray(table2[colnames2[0]]), np.asarray(table2[colnames2[3]])

	for chrom_id in range(1,23):
		filename1 = '%s/training_mtx/training2_kmer_%s.npy'%(path1,chrom_id)
		data1 = np.load(filename1)
		feature_dim = data1.shape[1]
		chrom_id1 = 'chr%s'%(chrom_id)
		b_1 = np.where(ref_chrom==chrom_id1)[0]
		mtx = np.zeros((len(b_1),feature_dim))
		t_ref_serial = ref_serial[b_1]
		b1 = np.where(chrom1==chrom_id1)[0]
		t_serial1 = serial1[b1]
		id1 = mapping_Idx(t_ref_serial,t_serial1)

		b2 = np.where(chrom2==chrom_id1)[0]
		t_serial2 = serial2[b2]
		id2 = mapping_Idx(t_ref_serial,t_serial2)
		if (len(id1)+len(id2))!=len(b_1):
			print('error! %d %d %d'%(len(id1),len(id2),len(b_1)))

		mtx[id1] = data1
		mtx[id2] = data2[b2]

		output_filename = '%s/training_mtx/training2_kmer_%s.1.npy'%(path1,chrom_id)
		np.save(output_filename,mtx,allow_pickle=True)

	return True

def merge_data_phyloP(filename1, filename2):

	path1 = '/volume01/yy3/seq_data/dl/replication_timing'
	filename_1 = '%s/signal/hg19.human.E_L.EL.log.min0.1.6k.match2ori.bed'%(path1)
	filename1 = '%s/estimate_rt/estimate_rt_hg19.txt'%(path1)
	filename2 = '%s/estimate_rt/estimate_rt_hg19.2.txt'%(path1)
	
	ref_table = pd.read_csv(filename_1,header=None,sep='\t')
	table1 = pd.read_csv(filename1,sep='\t')
	table2 = pd.read_csv(filename2,sep='\t')

	filename2 = '%s/training2_kmer_hg19.2.npy'%(path1)
	data2 = np.load(filename2)
	
	ref_chrom, ref_serial = ref_table[0], ref_table[3]
	colnames1 = list(table1)
	chrom1, serial1 = table1[colnames1[0]], table1[colnames1[3]]
	colnames2 = list(table2)
	chrom2, serial2 = table2[colnames2[0]], table2[colnames2[3]]

	for chrom_id in range(1,23):
		chrom_id1 = 'chr%s'%(chrom_id)
		filename1 = '%s/phyloP_chr%s.txt'%(chrom_id)
		data1 = pd.read_csv(filename1,sep='\t') 
		colnames1 = list(data1)
		t_serial1 = data1[colnames1[0]]

		filename2 = '%s/phyloP_chr%s.2.txt'%(chrom_id)
		data2 = pd.read_csv(filename1,sep='\t')
		colnames2 = list(data2)
		t_serial2 = data2[colnames2[0]]

		b_1 = np.where(ref_chrom==chrom_id1)[0]
		t_ref_serial = ref_serial[b_1]

		# data_1[colnames1[0]] = t_ref_serial
		id1 = mapping_Idx(t_ref_serial,t_serial1)
		data1 = np.asarray(data1)

		id2 = mapping_Idx(t_ref_serial,t_serial2)
		data2 = np.asarray(data2)
		n1, n2 = len(t_ref_serial), data1.shape[1]
		mtx = np.zeros((n1,n2))
		mtx[id1] = data1
		mtx[id2] = data2

		data_1 = pd.DataFrame(columns=colnames1,data=data_1)

		output_filename = '%s/phyloP_chr%s.1.txt'%(path1,chrom_id)
		data_1.to_csv(output_filename,index=False,sep='\t')

	return True

def merge_data_gc(filename1, filename2):

	path1 = '/volume01/yy3/seq_data/dl/replication_timing'
	filename_1 = '%s/signal/hg19.human.E_L.EL.log.min0.1.6k.match2ori.bed'%(path1)
	filename1 = '%s/estimate_rt/estimate_rt_hg19.txt'%(path1)
	filename2 = '%s/estimate_rt/estimate_rt_hg19.2.txt'%(path1)
	
	ref_table = pd.read_csv(filename_1,header=None,sep='\t')
	table1 = pd.read_csv(filename1,sep='\t')
	table2 = pd.read_csv(filename2,sep='\t')

	filename1 = '%s/training2_gc_hg19.txt'%(path1)
	data1 = pd.read_csv(filename2,header=None,sep='\t')
	data1 = np.asarray(data1)

	filename2 = '%s/training2_gc_hg19.2.txt'%(path1)
	data2 = pd.read_csv(filename2,header=None,sep='\t')
	data2 = np.asarray(data2)
	
	ref_chrom, ref_serial = ref_table[0], ref_table[3]
	colnames1 = list(table1)
	chrom1, serial1 = table1[colnames1[0]], table1[colnames1[3]]
	colnames2 = list(table2)
	chrom2, serial2 = table2[colnames2[0]], table2[colnames2[3]]

	n1, n2 = len(ref_serial), data1.shape[1]
	mtx = np.zeros((n1,n2))

	id1 = mapping_Idx(ref_serial,serial1)
	id2 = mapping_Idx(ref_serial,serial2)

	mtx[id1] = data1
	mtx[id2] = data2

	filename1 = '%s/training2_gc_hg19.1.txt'%(path1)
	fields = ['gc','gc_N','gc_skew']
	data_1 = pd.DataFrame(columns=fields,data=mtx)
	data_1.to_csv(filename1,header=False,index=False,sep='\t')

	return True

def get_model(input_shape):
	input1 = Input(shape = (input_shape,4))
	conv = Conv1D(filters = conv_1[0][0], kernel_size = conv_1[0][1],activation = "linear")(input1)
	conv = BatchNormalization()(conv)
	conv = Activation("relu")(conv)
	# conv = MaxPooling1D(stride = conv_1[0][2])(conv)
	
	conv = Conv1D(filters = conv_1[1][0], kernel_size = conv_1[1][1],activation = "linear")(conv)
	conv = BatchNormalization()(conv)
	conv = Activation("relu")(conv)

	conv = Conv1D(filters = conv_1[2][0], kernel_size = conv_1[2][1], activation="linear")(conv)
	conv = BatchNormalization()(conv)
	conv = Activation("relu")(conv)

	# conv = Flatten()(conv)

	input2 = Input(shape= (23,4))
	conv1 = input2

	if NUM_CONV_LAYER_2>2:
		conv1 = Conv1D(filters = conv_2[0][0], kernel_size = conv_2[0][1],activation = "linear")(conv1)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation("relu")(conv1)

	conv1 =  Conv1D(filters = conv_2[1][0], kernel_size = conv_2[1][1],activation = "linear")(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation("relu")(conv1)

	conv1 = Conv1D(filters = conv_2[2][0], kernel_size = conv_2[2][1], activation="linear")(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation("relu")(conv1)

	# conv1 = Flatten()(conv1)

	# conv = Concatenate()([conv,conv1])
	# conv = Dropout(0.5)(conv)

	print(conv.shape,conv1.shape)
	concat_layer_output = Concatenate(axis=1)([conv,conv1])
	print(concat_layer_output.shape)

	# units = 16
	# gru = tf.keras.layers.GRU(units, 
 #                              return_sequences=True, 
 #                              return_state=True, 
 #                              recurrent_initializer='glorot_uniform')

	n_kernels = lstm_num_kernels
	output_dim = lstm_output_dim
	biLSTM_layer1 = Bidirectional(LSTM(input_dim = n_kernels,
									output_dim = output_dim,
									return_sequences = True))

	x1 = biLSTM_layer1(concat_layer_output)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization()(x1)
	conv = Flatten()(x1)

	dense1 = Dense(fc1_output_dim,activation='relu')(conv)

	dense_layer_output = Dropout(0.5)(dense1)

	if NUM_DENSE_LAYER==2:
		dense2 = Dense(fc2_output_dim,activation='relu')(dense_layer_output)
		dense_layer_output = Dropout(0.5)(dense2)

	# output = Dense(1,activation= 'sigmoid')(dense1)
	output = Dense(1,activation= 'sigmoid')(dense_layer_output)

	model = Model(input = [input1,input2],output = output)
	adam = Adam(lr = learning_rate)
	model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])

	model.summary()
	return model

def get_model1(input_shape):
	input1 = Input(shape = (input_shape,4))
	conv = Conv1D(filters = conv_1[0][0], kernel_size = conv_1[0][1],activation = "linear")(input1)
	conv = BatchNormalization()(conv)
	conv = Activation("relu")(conv)
	conv = MaxPooling1D(strides = conv_1[0][2])(conv)
	
	conv = Conv1D(filters = conv_1[1][0], kernel_size = conv_1[1][1],activation = "linear")(conv)
	conv = BatchNormalization()(conv)
	conv = Activation("relu")(conv)

	conv = Conv1D(filters = conv_1[2][0], kernel_size = conv_1[2][1], activation="linear")(conv)
	conv = BatchNormalization()(conv)
	conv = Activation("relu")(conv)

	# conv = Flatten()(conv)

	input2 = Input(shape= (23,4))
	conv1 = input2

	if NUM_CONV_LAYER_2>2:
		conv1 = Conv1D(filters = conv_2[0][0], kernel_size = conv_2[0][1],activation = "linear")(conv1)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation("relu")(conv1)

	conv1 = Conv1D(filters = conv_2[2][0], kernel_size = conv_2[2][1], activation="linear")(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation("relu")(conv1)

	kernel_size1 = conv_1[2][1]
	kernel_size2 = conv_2[2][1]

	n_kernels = kernel_size1
	output_dim = lstm_output_dim
	biLSTM_layer1 = Bidirectional(LSTM(input_dim = n_kernels,
									output_dim = output_dim,
									return_sequences = True))

	n_kernels = kernel_size2
	output_dim = lstm_output_dim
	biLSTM_layer2 = Bidirectional(LSTM(input_dim = n_kernels,
									output_dim = output_dim,
									return_sequences = True))

	x1 = biLSTM_layer1(conv)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization()(x1)

	x2 = biLSTM_layer2(conv1)
	# x1 = BatchNormalization()(x1)
	x2 = LayerNormalization()(x2)

	# conv1 = Flatten()(conv1)

	# conv = Concatenate()([conv,conv1])
	# conv = Dropout(0.5)(conv)

	print(conv.shape,conv1.shape)
	print(x1.shape,x2.shape)
	concat_layer_output = Concatenate(axis=1)([x1,x2])
	print(concat_layer_output.shape)

	# units = 16
	# gru = tf.keras.layers.GRU(units, 
 #                              return_sequences=True, 
 #                              return_state=True, 
 #                              recurrent_initializer='glorot_uniform')

	concat1 = Flatten()(concat_layer_output)

	dense1 = Dense(fc1_output_dim,activation='relu')(concat1)

	dense_layer_output = Dropout(0.5)(dense1)

	if NUM_DENSE_LAYER==2:
		dense2 = Dense(fc2_output_dim,activation='relu')(dense_layer_output)
		dense_layer_output = Dropout(0.5)(dense2)

	# output = Dense(1,activation= 'sigmoid')(dense1)
	output = Dense(1,activation= 'sigmoid')(dense_layer_output)

	model = Model(input = [input1,input2],output = output)
	adam = Adam(lr = learning_rate)
	model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])

	model.summary()
	return model

def get_model2(input_shape):
	input1 = Input(shape = (input_shape,4))
	conv = Conv1D(filters = conv_1[0][0], kernel_size = conv_1[0][1], activation = "linear")(input1)
	conv = BatchNormalization()(conv)
	conv = Activation("relu")(conv)
	conv = MaxPooling1D(pool_size = conv_1[0][2], strides = conv_1[0][2])(conv)
	
	conv = Conv1D(filters = conv_1[1][0], kernel_size = conv_1[1][1], activation = "linear")(conv)
	conv = BatchNormalization()(conv)
	conv = Activation("relu")(conv)
	conv = MaxPooling1D(pool_size = conv_1[1][2], strides = conv_1[1][2])(conv)

	conv = Conv1D(filters = conv_1[2][0], kernel_size = conv_1[2][1], activation = "linear")(conv)
	conv = BatchNormalization()(conv)
	conv = Activation("relu")(conv)

	# kernel_size1 = conv_1[2][1]
	# kernel_size2 = conv_2[2][1]

	n_kernels = conv_1[2][0]
	output_dim = lstm_output_dim
	biLSTM_layer1 = Bidirectional(LSTM(input_dim = n_kernels,
									output_dim = output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1))

	# n_kernels = conv_2[2][0]
	# output_dim = lstm_output_dim
	# biLSTM_layer2 = Bidirectional(LSTM(input_dim = n_kernels,
 #                                    output_dim = output_dim,
 #                                    return_sequences = True,
 #                                    recurrent_dropout = 0.1))

	x1 = biLSTM_layer1(conv)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization()(x1)
	x1 = Flatten()(x1)
	# dense1 = Dense(fc1_output_dim,activation='relu')(x1)
	dense1 = Dense(fc1_output_dim)(x1)
	dense1 = BatchNormalization()(dense1)
	dense1 = Activation("relu")(dense1)

	dense_layer_output = Dropout(0.5)(dense1)

	# if NUM_DENSE_LAYER==2:
	# 	dense2 = Dense(fc2_output_dim,activation='relu')(dense_layer_output)
	# 	dense_layer_output = Dropout(0.5)(dense2)

	# output = Dense(1,activation= 'sigmoid')(dense1)
	# output = Dense(1,activation= 'sigmoid')(dense_layer_output)
	output = Dense(1)(dense_layer_output)
	output = BatchNormalization()(output)
	output = Activation("sigmoid")(output)
	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	model.compile(adam,loss = 'mean_absolute_percentage_error')
	# model.compile(adam,loss = 'kullback_leibler_divergence')

	model.summary()
	return model

def get_model2a(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (input_shape,feature_dim))

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1))

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization()(x1)
	x1 = Flatten()(x1)

	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim)(x1)
		dense1 = BatchNormalization()(dense1)
		dense1 = Activation("relu")(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
	else:
		dense_layer_output = x1

	output = Dense(1)(dense_layer_output)
	output = BatchNormalization()(output)
	output = Activation("sigmoid")(output)
	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()
	return model

def get_model2a_1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (input_shape,feature_dim))

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1))

	biLSTM_layer2 = Bidirectional(LSTM(input_shape=(None, output_dim),
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1))

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization()(x1)

	x2 = biLSTM_layer2(x1)
	x2 = LayerNormalization()(x2)
	input2 = Flatten()(x2)

	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim)(input2)
		dense1 = BatchNormalization()(dense1)
		dense1 = Activation("relu")(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
	else:
		dense_layer_output = input2

	output = Dense(1)(dense_layer_output)
	output = BatchNormalization()(output)
	output = Activation("sigmoid")(output)
	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()
	return model

def get_model2a_2(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (input_shape,feature_dim))

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1))

	biLSTM_layer2 = Bidirectional(LSTM(input_shape=(None, output_dim),
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1))

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization()(x1)

	x2 = biLSTM_layer2(x1)
	x2 = LayerNormalization()(x2)

	t1 = Flatten()(x1)
	t2 = Flatten()(x2)

	input2 = Average()([t1,t2])

	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim)(input2)
		dense1 = BatchNormalization()(dense1)
		dense1 = Activation("relu")(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
	else:
		dense_layer_output = input2

	output = Dense(1)(dense_layer_output)
	output = BatchNormalization()(output)
	output = Activation("sigmoid")(output)
	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()
	return model

def test1():

	model = keras.models.Sequential()
	model.add(keras.layers.Embedding(input_dim=10000,
								output_dim=300,
								mask_zero=True))
	model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
													   return_sequences=True)))
	model.add(SeqSelfAttention(attention_activation='sigmoid'))
	model.add(keras.layers.Dense(units=5))
	model.compile(
		optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['categorical_accuracy'],
		)
	model.summary()

	return model

def test2():

	inputs = keras.layers.Input(shape=(None,))
	embd = keras.layers.Embedding(input_dim=32,
							  output_dim=16,
							  mask_zero=True)(inputs)
	lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=16,
													return_sequences=True))(embd)
	att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
					   kernel_regularizer=keras.regularizers.l2(1e-4),
					   bias_regularizer=keras.regularizers.l1(1e-4),
					   attention_regularizer_weight=1e-4,
					   name='Attention')(lstm)
	dense = keras.layers.Dense(units=5, name='Dense')(att)
	model = keras.models.Model(inputs=inputs, outputs=[dense])
	model.compile(
		optimizer='adam',
		loss={'Dense': 'sparse_categorical_crossentropy'},
		metrics={'Dense': 'categorical_accuracy'},
	)

	return model

def test3():
	_input = Input(shape=[max_length], dtype='int32')

	# get the embedding layer
	embedded = Embedding(
		input_dim=vocab_size,
		output_dim=embedding_size,
		input_length=max_length,
		trainable=False,
		mask_zero=False
		)(_input)

	activations = LSTM(units, return_sequences=True)(embedded)

	# compute importance for each step
	attention = Dense(1, activation='tanh')(activations)
	attention = Flatten()(attention)
	attention = Activation('softmax')(attention)
	attention = RepeatVector(units)(attention)
	attention = Permute([2, 1])(attention)

	sent_representation = merge([activations, attention], mode='mul')
	sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(units,))(sent_representation)

	probabilities = Dense(3, activation='softmax')(sent_representation)

	model = Model(input = _input, output = probabilities)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()

	return model

# probability
def get_model2a3(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	fc2_output_dim = config['fc2_output_dim']
	prev_context = config['prev_context']
	# input1 = Input(shape = (None,feature_dim))
	# input2 = Input(shape = (None,prev_context))
	input1 = Input(shape = (feature_dim,))
	input2 = Input(shape = (prev_context,1))

	# signal predicted from sequence feature
	dense1 = Dense(fc1_output_dim)(input1)
	dense1 = BatchNormalization()(dense1)
	dense1 = Activation("relu")(dense1)

	dense2 = Dense(1)(dense1)
	dense2 = BatchNormalization()(dense2)
	signal2 = Activation("sigmoid")(dense2)

	# signal predicted from previous loci
	output_dim1 = 5
	biLSTM_layer1 = LSTM(input_shape=(prev_context, 1), 
									units=output_dim1,
									return_sequences = True,
									recurrent_dropout = 0.1)

	signal1 = biLSTM_layer1(input2)
	signal1 = LayerNormalization()(signal1)
	signal1 = Flatten()(signal1)
	signal1 = Dense(1)(signal1)
	signal1 = BatchNormalization()(signal1)
	signal1 = Activation("relu")(signal1)
	
	# if output_dim1>1:
	# 	signal1 = LayerNormalization()(signal1)
	# 	signal1 = Dense(1)(signal1)
	# 	signal1 = BatchNormalization()(signal1)
	# 	signal1 = Activation("relu")(signal1)

	# probability
	dense3 = Dense(fc2_output_dim)(input1)
	dense3 = BatchNormalization()(dense3)
	dense_3 = Activation("relu")(dense3)

	dense_3 = Dense(2)(dense_3)
	# dense_3 = BatchNormalization()(dense_3)
	prob = Activation("softmax")(dense_3)

	t_signal1 = Concatenate(axis=-1)([signal1,signal2])
	output = Dot(axes=-1)([t_signal1,prob]) 

	# biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
	# 								units=output_dim,
	# 								return_sequences = True,
	# 								recurrent_dropout = 0.1))

	# x1 = biLSTM_layer1(input1)
	# # x1 = BatchNormalization()(x1)
	# x1 = LayerNormalization()(x1)
	# # x1 = Flatten()(x1)

	# x1 = SeqSelfAttention(attention_activation='sigmoid')(x1)
	# x1 = Concatenate(axis=-1)([x1,input2])
	# if fc1_output_dim>0:
	# 	dense1 = Dense(fc1_output_dim)(x1)
	# 	dense1 = BatchNormalization()(dense1)
	# 	dense1 = Activation("relu")(dense1)
	# 	dense_layer_output = Dropout(0.5)(dense1)
	# else:
	# 	dense_layer_output = x1

	# # concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	# output = Dense(1)(dense_layer_output)
	# output = BatchNormalization()(output)
	# output = Activation("sigmoid")(output)
	# # output = Activation("softmax")(output)

	model = Model(input = [input1,input2], output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()
	return model

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

def get_model2a1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	# input1 = Input(shape = (input_shape,feature_dim))
	input1 = Input(shape = (None,feature_dim))

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1))

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization()(x1)
	# x1 = Flatten()(x1)

	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim)(x1)
		dense1 = BatchNormalization()(dense1)
		dense1 = Activation("relu")(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
	else:
		dense_layer_output = x1

	output = Dense(1)(dense_layer_output)
	output = BatchNormalization()(output)
	output = Activation("sigmoid")(output)
	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = learning_rate)
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

def get_model2a1_attention_sequential(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	# input1 = Input(shape = (input_shape,feature_dim))

	model = keras.models.Sequential()
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1),input_shape=(None, feature_dim)))
	model.add(LayerNormalization())
	model.add(SeqSelfAttention(attention_activation='sigmoid'))

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

	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')
	adam = Adam(lr = learning_rate)

	model.summary()

	return model

# multi-fraction output
def get_model2a1_attention_3(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	output_dim2 = config['output_dim2']
	input1 = Input(shape = (None,feature_dim))
	lr = config['lr']
	ltype = config['ltype']
	batch_norm = config['batch_norm']
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
	output = Dense(output_dim2,name='dense2')(dense_layer_output)
	if batch_norm>0:
		output = BatchNormalization(name='batchnorm2')(output)
	# output = Activation("sigmoid",name='activation2')(output)
	output = Activation("softmax",name='activation2')(output)
	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = lr)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	# model.compile(adam,loss = 'mean_squared_error')
	# model.compile(adam,loss = 'kullback_leibler_divergence')
	model.compile(adam,loss = ltype)

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

	# units_1 = config['units_1']
	# if units_1>0:
	# 	dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_1')(input1))
	# 	# x2 = K.batch_dot(dense_layer2,biLSTM_layer1)
	# 	dense_layer_2 = TimeDistributed(Dense(1,name='dense_1')(dense_layer_1))
	# else:
	# 	dense_layer_2 = TimeDistributed(Dense(1,name='dense_1')(input1))

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

# method 12 and 17: network1 for estimating weights, network2 for predicting signals
def get_model2a1_attention_1_1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']

	input1 = Input(shape = (n_steps,feature_dim))

	units_1 = config['units1']
	if units_1>0:
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)
		dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_1)
	else:
		dense_layer_2 = dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(input1)
	
	# attention1 = Flatten()(dense_layer_2)
	# attention1 = Activation('softmax',name='attention1')(attention1)
	attention1 = TimeDistributed(Activation("sigmoid",name='activation_1'),name='logits_T')(dense_layer_2)
	attention1 = Flatten()(attention1)

	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)
	layer_1 = Multiply()([dense_layer_output1, attention1])
	# dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation!='':
		x1 = Activation(activation,name='activation1')(x1)
	
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if config['attention2']==1:
		x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention2')(x1)
	output = Dense(1,name='dense2')(x1)
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

# network1 for estimating weights, self-attention and network2 for predicting signals
# method 15: network2 + self-attention
# method 18: network2
def get_model2a1_attention_1_2(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']

	input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)
		dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_1)
	else:
		dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(input1)
	
	# attention1 = Flatten()(dense_layer_2)
	# attention1 = Activation('softmax',name='attention1')(attention1)
	attention1 = TimeDistributed(Activation("sigmoid",name='activation_1'))(dense_layer_2)
	attention1 = Flatten()(attention1)

	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)
	layer_1 = Multiply()([dense_layer_output1, attention1])
	# dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)

	x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(layer_1)

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(x1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation!='':
		x1 = Activation(activation,name='activation1')(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if config['attention2']==1:
		x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(x1)
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

# plot and comparison
def plot_test1():

	# print(self.run_id,self.config['tau'],self.config['n_select'],
	# 			self.config['activation3'],self.config['activation_self'],
	# 			self.config['type_id1'],self.config['ratio1'],
	# 			self.config['regularizer1'],self.config['regularizer2'],
	# 			self.train_chromvec,self.test_chromvec)

	n_col = 1
	n_row = 20
	regularizer1 = 0.001

	fig = plt.figure(figsize=(6, 6))
	grid1 = gridspec.GridSpec(n_row, n_col, wspace=0.2, hspace=0.2)
	annotation = ['WT','WT predicted','Del.-abc','Del.-abc predicted']
	activation = 'ReLU'

	for i in range(0,n_row):

		fig1 = gridspec.GridSpecFromSubplotSpec(1, 2,
					subplot_spec=grid1[i], wspace=0.1, hspace=0.1)
		ax = fig.add_subplot(fig1[0])
		plt.title('Signal: %s, L1: %.2e '%(activation,regularizer1))
		# plt.suptitle('%d, ReLU, l1: %.2e '%(i,regularizer1))
		x1 = np.asarray(range(0,10))
		y1_region = np.random.rand(10)
		y1_predicted_region = np.random.rand(10)
		# y2_region = np.random.rand(10)
		# y2_predicted_region = np.random.rand(10)

		ax.plot(x1,y1_region,color='orange',linewidth=2,linestyle='dashed',label='WT')
		ax.plot(x1,y1_predicted_region,color='green',linewidth=2,label='WT predicted')

		for j in range(3):
			y2_region = np.random.rand(10)
			y2_predicted_region = np.random.rand(10)
			ax.plot(x1,y2_region,color='blue',linewidth=2,linestyle='dashed',label='Del.-abc')
			ax.plot(x1,y2_predicted_region,color='red',linewidth=2,label='Del.-abc predicted')
		ax.legend(annotation,loc='upper left',frameon=False)

		fig.add_subplot(ax)

		ax = fig.add_subplot(fig1[1])
		plt.title('Estimated importance')
		ax.plot(x1,y1_predicted_region,color='red',linewidth=2,label='WT')
		# fig.add_subplot(ax)
		fig.add_subplot(ax)

	plt.savefig('test1.png',dpi=300)

	return True

# plot and comparison
# def plot_1(file_path,method,runid1_list,runid2_list,method_vec,region_list1,region_list2,config_vec):
def plot_1(file_path,runid1_list,region_list1,region_list2,config_vec,output_file_path=''):
	# print(self.run_id,self.config['tau'],self.config['n_select'],
	# 			self.config['activation3'],self.config['activation_self'],
	# 			self.config['type_id1'],self.config['ratio1'],
	# 			self.config['regularizer1'],self.config['regularizer2'],
	# 			self.train_chromvec,self.test_chromvec)

	# regularizer1 = config['regularizer1']

	# fields = ['chrom','start','stop','serial','signal',
	# 				'predicted_signal','predicted_attention']
	
	# num1 = int(np.ceil(len(runid1_list)/2))
	# import matplotlib.pylab as pylab
	# params = {'legend.fontsize': 'x-large',
	# 	 'figure.figsize': (15, 5),
	# 	 'axes.labelsize': 'x-large',
	# 	 'axes.titlesize':'x-large',
	# 	 'xtick.labelsize':'x-large',
	# 	 'ytick.labelsize':'x-large'}
	params = {
		 'axes.labelsize': 8,
		 'axes.titlesize': 8,
		 'xtick.labelsize':8,
		 'ytick.labelsize':8}
	pylab.rcParams.update(params)

	num1 = len(runid1_list)
	n_col = 1
	n_row = num1
	#  regularizer1 = 0.001

	fig = plt.figure(figsize=(10, 6))
	grid1 = gridspec.GridSpec(n_row, n_col, wspace=0.5, hspace=0.3)
	annotation = ['WT','WT predicted','Del.-abc','Del.-abc predicted']
	color_vec1 = ['black','seagreen','olive','blue','red','green']

	region1 = region_list1[0]
	t_region_list = []
	num2 = len(region_list2)
	t_region = [region1[0], region1[1], region_list2[0][1]]
	t_region_list.append(t_region)
	for i in range(0,num2-1):
		t_region = [region_list2[i][0],region_list2[i][2],region_list2[i+1][1]]
		t_region_list.append(t_region)
	
	t_region = [region_list2[num2-1][0],region_list2[num2-1][2],region1[2]]
	t_region_list.append(t_region)
	print(t_region_list)

	filename1_list, filename2_list = [], []
	filename1_list1, filename2_list1 = [], []
	feature_id1 = -1
	for run_id in runid1_list:
		run_id1, run_id2, method1 = run_id
		# filename1 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id1,method1)
		filename1 = '%s/test_vec2_%d_%d_[%d].1_0.1.txt'%(file_path,run_id1,method1,feature_id1)
		filename1_list.append(filename1)
		filename1 = '%s/test_vec2_%d_%d_[%d].1_0.txt'%(file_path,run_id1,method1,feature_id1)
		filename1_list1.append(filename1)

		# filename2 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id2,method1)
		filename2 = '%s/test_vec2_%d_%d_[%d].1_0.1.txt'%(file_path,run_id2,method1,feature_id1)
		filename2_list.append(filename2)
		filename2 = '%s/test_vec2_%d_%d_[%d].1_0.txt'%(file_path,run_id2,method1,feature_id1)
		filename2_list1.append(filename2)

	# for run_id in runid2_list:
	# 	filename2 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id,method)
	# 	filename2_list.append(filename2)

	for i in range(0,num1):
		
		filename1, filename2 = filename1_list[i], filename2_list[i]
		config = config_vec[runid1_list[i][0]]
		run_id = runid1_list[i]

		regularizer1, regularizer2 = 0, 0
		if 'regularizer1' in config:
			regularizer1 = config['regularizer1']
		if 'regularizer2' in config:
			regularizer2 = config['regularizer2']
		
		# activation3 = config['activation3']
		# print(activation3,regularizer1,regularizer2)

		sel_column = ['predicted_attention','Q2']
		sel_id_ori, sel_id_filter = 0, 0
		if 'sel_id' in config:
			sel_id_ori = config['sel_id']

		sel_id = sel_id_ori
		if sel_id_ori==2:
			sel_id=1
			sel_id_filter = 1

		sel_id1 = sel_column[sel_id]

		data1 = pd.read_csv(filename1,sep='\t')
		chrom1, start1, stop1 = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop'])
		y_signal1, y_predicted1 = np.asarray(data1['signal']), np.asarray(data1['predicted_signal'])
		# predicted_attention1 = np.asarray(data1['predicted_attention'])
		# predicted_attention1 = np.asarray(data1['Q2'])
		predicted_attention1 = np.asarray(data1[sel_id1])

		data2 = pd.read_csv(filename2,sep='\t')
		chrom2, start2, stop2 = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop'])
		y_signal2, y_predicted2 = np.asarray(data2['signal']), np.asarray(data2['predicted_signal'])
		predicted_attention2 = np.asarray(data2[sel_id1])

		filename1_1, filename2_1 = filename1_list1[i], filename2_list1[i]

		if os.path.exists(filename1_1)==True and os.path.exists(filename2_1)==True:
			data1_1 = pd.read_csv(filename1_1,sep='\t')
			t_chrom1, t_start1, t_stop1 = np.asarray(data1_1['chrom']), np.asarray(data1_1['start']), np.asarray(data1_1['stop'])
			y_predicted1 = np.asarray(data1_1['predicted_signal'])
			# predicted_attention1 = np.asarray(data1['predicted_attention'])
			if np.min(y_signal1)>-0.1:
				y_signal1 = 2*y_signal1-1
				y_predicted1 = 2*y_predicted1-1

			data2_1 = pd.read_csv(filename2_1,sep='\t')
			t_chrom2, t_start2, t_stop2 = np.asarray(data2_1['chrom']), np.asarray(data2_1['start']), np.asarray(data2_1['stop'])
			y_predicted2 = np.asarray(data2_1['predicted_signal'])
			# predicted_attention2 = np.asarray(data2['predicted_attention'])
		
		if np.min(y_signal2)>-0.1:
			y_signal2 = 2*y_signal2-1
			y_predicted2 = 2*y_predicted2-1

		if np.min(predicted_attention2)<0:
			predicted_attention2 = signal_normalize(predicted_attention2, [0,1])

		if sel_id_filter==1:
			predicted_attention2[predicted_attention2<0.8]=0

		fig1 = gridspec.GridSpecFromSubplotSpec(1, 2,
					subplot_spec=grid1[i], wspace=0.2, hspace=0.1)
		ax = fig.add_subplot(fig1[0])
		# plt.title('Signal: %s, L1: %.2e '%(activation,regularizer1))

		region1 = region_list1[0]
		t_chrom, t_start, t_stop = region1[0], region1[1], region1[2]
		t_chrom = 'chr%s'%(t_chrom)
		b = np.where((chrom2==t_chrom)&(start2<t_stop)&(stop2>t_start))[0]
		print(t_chrom, t_start, t_stop,len(b))
		x2, y2_region, y2_predicted_region = start2[b], y_signal2[b], y_predicted2[b]
		print(x2.shape,y2_region.shape,y2_predicted_region.shape)
		predicted_attention2_region = predicted_attention2[b]

		# x2 = ['%.2fMb'%(pos*1.0/1e06) for pos in x2]
		# print(x2[0:10])

		ax.plot(x2,y2_region,color='orange',linewidth=2,linestyle='dashed',label='WT')
		ax.plot(x2,y2_predicted_region,color='green',linewidth=2,label='WT predicted')

		for t_region in t_region_list:
			t_chrom, t_start, t_stop = t_region[0], t_region[1], t_region[2]
			t_chrom = 'chr%s'%(t_chrom)
			b = np.where((chrom1==t_chrom)&(start1<t_stop)&(stop1>t_start))[0]

			x1, y1_region, y1_predicted_region = start1[b], y_signal1[b], y_predicted1[b]
			ax.plot(x1,y1_region,color='blue',linewidth=2,linestyle='dashed',label='Del.-abc')
			ax.plot(x1,y1_predicted_region,color='red',linewidth=2,label='Del.-abc predicted')
			plt.ylim(-1,1)

		plt.hlines(y=0, xmin=region1[1], xmax=region1[2], color='black', linewidth=1, linestyle='dashed')

		j = 0
		for t_region in region_list2:
			plt.axvline(x=t_region[1], ymin=-1, ymax=1,color=color_vec1[j])
			plt.axvline(x=t_region[2], ymin=-1, ymax=1,color=color_vec1[j])
			j += 1

		if i==0:
			ax.legend(annotation,loc='upper left',frameon=False, labelspacing=0.5, prop={'size': 6})

		# plt.title('%d: %s L1:%.2e L2:%.2e'%(run_id,activation3,regularizer1,regularizer2))
		# plt.title('%s L1:%.2e L2:%.2e'%(activation3,regularizer1,regularizer2))
		plt.title('RT prediction')
		plt.ylabel('RT signal')
		plt.xlabel('Chr16')
		if i<num1-1:
			ax.get_xaxis().set_ticks([])
			ax.xaxis.label.set_visible(False)
			# ax.set_xticklabels([])
		else:
			x_labels = ax.get_xticks()
			print(x_labels[0])
			ax.set_xticklabels(['%.2f'%(x/1e06) for x in x_labels])
			ax.text(0.95, -0.5, '(Mb)',transform=ax.transAxes,fontsize=8)
			# ax.text(1.0, -0.1, 'a',transform=ax.transAxes)
			# ax.text(1.0, -0.1, 'b',transform=ax.transAxes)
			# ax.text(1.0, -0.1, 'c',transform=ax.transAxes)

		fig.add_subplot(ax)

		region2 = region_list1[1]
		t_chrom, t_start, t_stop = region2[0], region2[1], region2[2]
		print(t_chrom, t_start, t_stop)
		t_chrom = 'chr%s'%(t_chrom)
		b = np.where((chrom2==t_chrom)&(start2<t_stop)&(stop2>t_start))[0]
		x2 = start2[b]
		print(len(b))
		predicted_attention2_region = predicted_attention2[b]

		value1, value2, value3 = np.max(predicted_attention2_region),np.min(predicted_attention2_region),np.median(predicted_attention2_region)
		print(value1,value2,value3)

		ax = fig.add_subplot(fig1[1])
		ax.plot(x2,predicted_attention2_region, marker='o', markerfacecolor='red', 
			markersize=3, color='red',linewidth=1,label='WT')

		t_value = 1
		j = 0
		for t_region in region_list2:
			plt.axvline(x=t_region[1], ymin=0, ymax=t_value,color=color_vec1[j])
			plt.axvline(x=t_region[2], ymin=0, ymax=t_value,color=color_vec1[j])
			j += 1
		
		plt.title('Estimated importance score')
		plt.xlabel('Chr16')
		if i<num1-1:
			ax.get_xaxis().set_ticks([])
			ax.xaxis.label.set_visible(False)
		else:
			x_labels = ax.get_xticks()
			ax.set_xticklabels(['%.2f'%(x/1e06) for x in x_labels])
			ax.text(0.95, -0.5, '(Mb)',transform=ax.transAxes,fontsize=8)

		fig.add_subplot(ax)

	if output_file_path == '':
		file_path1 = './mnt/yy3/data3/fig2'
	else:
		file_path1 = output_file_path

	plt.savefig('%s/test%d_%d.%d.png'%(file_path1,runid1_list[0][0],runid1_list[-1][1],sel_id_ori),dpi=300)

	# for i in range(1,num1+1):
	# 	plt.subplot(n_row,n_col,2*i-1)
	# 	plt.title('Signal')
	# 	for t_region in t_region_list:
	# 		x1, y1_region, y1_predicted_region = t_region[0], t_region[1], t_region[2]
	# 		plt.plot(x1,y1_region,color='blue',linewidth=2,linestyle='dashed',label='Del.-abc')
	# 		plt.plot(x1,y1_predicted_region,color='red',,linewidth=2,label='Del.-abc predicted')

	# 	plt.plot(x2,y2_region,color='orange',linewidth=2,linestyle='dashed',label='WT')
	# 	plt.plot(x2,y2_predicted_region,color='green',,linewidth=2,label='WT predicted')

	# 	plt.subplot(n_row,n_col,2*i)
	# 	plt.title('Estimated importance')
	# 	plt.plot(x1,y1_region,color='blue',linewidth=2,label='Del.-abc')
	# 	plt.plot(x1,y1_predicted_region,color='red',,linewidth=2,label='Del.-abc')
	# 	plt.plot(x1,y1_region,color='blue',linewidth=2,label='Del.-abc')
	# 	plt.plot(x1,y1_predicted_region,color='red',,linewidth=2,label='Del.-abc')

	# plt.suptitle('%s, l1: %.2e '%(regularizer1))

	return True

# plot and comparison
# def plot_1(file_path,method,runid1_list,runid2_list,method_vec,region_list1,region_list2,config_vec):
def plot_1_sub1(file_path,runid1_list,region_list1,region_list2,config_vec):
	# print(self.run_id,self.config['tau'],self.config['n_select'],
	# 			self.config['activation3'],self.config['activation_self'],
	# 			self.config['type_id1'],self.config['ratio1'],
	# 			self.config['regularizer1'],self.config['regularizer2'],
	# 			self.train_chromvec,self.test_chromvec)

	# regularizer1 = config['regularizer1']

	# fields = ['chrom','start','stop','serial','signal',
	# 				'predicted_signal','predicted_attention']
	
	# num1 = int(np.ceil(len(runid1_list)/2))
	# import matplotlib.pylab as pylab
	# params = {'legend.fontsize': 'x-large',
	# 	 'figure.figsize': (15, 5),
	# 	 'axes.labelsize': 'x-large',
	# 	 'axes.titlesize':'x-large',
	# 	 'xtick.labelsize':'x-large',
	# 	 'ytick.labelsize':'x-large'}
	params = {
		 'axes.labelsize': 8,
		 'axes.titlesize': 8,
		 'xtick.labelsize':8,
		 'ytick.labelsize':8}
	pylab.rcParams.update(params)

	num1 = len(runid1_list)
	n_col = 1
	n_row = num1
	#  regularizer1 = 0.001

	fig = plt.figure(figsize=(10, 6))
	grid1 = gridspec.GridSpec(n_row, n_col, wspace=0.5, hspace=0.3)
	# annotation = ['WT','WT predicted','Del.-abc','Del.-abc predicted']
	annotation = ['WT(Allele 1)','WT predicted','Allele 2','Allele 2 predicted']
	color_vec1 = ['black','seagreen','olive','blue','red','green']

	region1 = region_list1[0]
	t_region_list = []
	num2 = len(region_list2)
	t_region = [region1[0], region1[1], region_list2[0][1]]
	t_region_list.append(t_region)
	for i in range(0,num2-1):
		t_region = [region_list2[i][0],region_list2[i][2],region_list2[i+1][1]]
		t_region_list.append(t_region)
	
	t_region = [region_list2[num2-1][0],region_list2[num2-1][2],region1[2]]
	t_region_list.append(t_region)
	print(t_region_list)

	filename1_list, filename2_list = [], []
	filename1_list1, filename2_list1 = [], []
	feature_id1 = -1
	for run_id in runid1_list:
		run_id1, run_id2, method1 = run_id
		# filename1 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id1,method1)
		filename1 = '%s/test_vec2_%d_%d_[%d].1_0.1.txt'%(file_path,run_id1,method1,feature_id1)
		filename1_list.append(filename1)
		filename1 = '%s/test_vec2_%d_%d_[%d].1_0.txt'%(file_path,run_id1,method1,feature_id1)
		filename1_list1.append(filename1)

		# filename2 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id2,method1)
		filename2 = '%s/test_vec2_%d_%d_[%d].1_0.1.txt'%(file_path,run_id2,method1,feature_id1)
		filename2_list.append(filename2)
		filename2 = '%s/test_vec2_%d_%d_[%d].1_0.txt'%(file_path,run_id2,method1,feature_id1)
		filename2_list1.append(filename2)

	# for run_id in runid2_list:
	# 	filename2 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id,method)
	# 	filename2_list.append(filename2)

	for i in range(0,num1):
		
		filename1, filename2 = filename1_list[i], filename2_list[i]
		config = config_vec[runid1_list[i][0]]
		run_id = runid1_list[i]

		regularizer1, regularizer2 = 0, 0
		if 'regularizer1' in config:
			regularizer1 = config['regularizer1']
		if 'regularizer2' in config:
			regularizer2 = config['regularizer2']
		
		# activation3 = config['activation3']
		# print(activation3,regularizer1,regularizer2)

		sel_column = ['predicted_attention','Q2']
		sel_id_ori, sel_id_filter = 0, 0
		sel_id = sel_id_ori
		if 'sel_id' in config:
			sel_id_ori = config['sel_id']
			if sel_id_ori==2:
				sel_id=1
				sel_id_filter = 1
		sel_id1 = sel_column[sel_id]

		data1 = pd.read_csv(filename1,sep='\t')
		chrom1, start1, stop1 = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop'])
		y_signal1, y_predicted1 = np.asarray(data1['signal']), np.asarray(data1['predicted_signal'])
		# predicted_attention1 = np.asarray(data1['predicted_attention'])
		# predicted_attention1 = np.asarray(data1['Q2'])
		predicted_attention1 = np.asarray(data1[sel_id1])

		data2 = pd.read_csv(filename2,sep='\t')
		chrom2, start2, stop2 = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop'])
		y_signal2, y_predicted2 = np.asarray(data2['signal']), np.asarray(data2['predicted_signal'])
		predicted_attention2 = np.asarray(data2[sel_id1])

		filename1_1, filename2_1 = filename1_list1[i], filename2_list1[i]

		if os.path.exists(filename1_1)==True and os.path.exists(filename2_1)==True:
			data1_1 = pd.read_csv(filename1_1,sep='\t')
			t_chrom1, t_start1, t_stop1 = np.asarray(data1_1['chrom']), np.asarray(data1_1['start']), np.asarray(data1_1['stop'])
			y_predicted1 = np.asarray(data1_1['predicted_signal'])
			# predicted_attention1 = np.asarray(data1['predicted_attention'])
			if np.min(y_signal1)>-0.1:
				y_signal1 = 2*y_signal1-1
				y_predicted1 = 2*y_predicted1-1

			data2_1 = pd.read_csv(filename2_1,sep='\t')
			t_chrom2, t_start2, t_stop2 = np.asarray(data2_1['chrom']), np.asarray(data2_1['start']), np.asarray(data2_1['stop'])
			y_predicted2 = np.asarray(data2_1['predicted_signal'])
			# predicted_attention2 = np.asarray(data2['predicted_attention'])
		
		if np.min(y_signal2)>-0.1:
			y_signal2 = 2*y_signal2-1
			y_predicted2 = 2*y_predicted2-1

		if np.min(predicted_attention2)<0:
			predicted_attention2 = signal_normalize(predicted_attention2, [0,1])

		if sel_id_filter==1:
			predicted_attention2[predicted_attention2<0.8]=0

		fig1 = gridspec.GridSpecFromSubplotSpec(1, 2,
					subplot_spec=grid1[i], wspace=0.2, hspace=0.1)
		ax = fig.add_subplot(fig1[0])
		# plt.title('Signal: %s, L1: %.2e '%(activation,regularizer1))

		region1 = region_list1[0]
		t_chrom, t_start, t_stop = region1[0], region1[1], region1[2]
		t_chrom = 'chr%s'%(t_chrom)
		b = np.where((chrom2==t_chrom)&(start2<t_stop)&(stop2>t_start))[0]
		print(t_chrom, t_start, t_stop,len(b))
		x2, y2_region, y2_predicted_region = start2[b], y_signal2[b], y_predicted2[b]
		print(x2.shape,y2_region.shape,y2_predicted_region.shape)
		predicted_attention2_region = predicted_attention2[b]

		b = np.where((chrom1==t_chrom)&(start1<t_stop)&(stop1>t_start))[0]
		print(t_chrom, t_start, t_stop,len(b))
		x1, y1_region, y1_predicted_region = start1[b], y_signal1[b], y_predicted1[b]
		print(x2.shape,y2_region.shape,y2_predicted_region.shape)

		# x2 = ['%.2fMb'%(pos*1.0/1e06) for pos in x2]
		# print(x2[0:10])

		ax.plot(x2,y2_region,color='orange',linewidth=2,linestyle='dashed',label='WT(Allele 1)')
		ax.plot(x2,y2_predicted_region,color='green',linewidth=2,label='WT predicted')

		ax.plot(x1,y1_region,color='blue',linewidth=2,linestyle='dashed',label='Allele 2')
		ax.plot(x1,y1_predicted_region,color='red',linewidth=2,label='Allele 2 predicted')

		# for t_region in t_region_list:
		# 	t_chrom, t_start, t_stop = t_region[0], t_region[1], t_region[2]
		# 	t_chrom = 'chr%s'%(t_chrom)
		# 	b = np.where((chrom1==t_chrom)&(start1<t_stop)&(stop1>t_start))[0]

		# 	x1, y1_region, y1_predicted_region = start1[b], y_signal1[b], y_predicted1[b]
		# 	ax.plot(x1,y1_region,color='blue',linewidth=2,linestyle='dashed',label='Del.-abc')
		# 	ax.plot(x1,y1_predicted_region,color='red',linewidth=2,label='Del.-abc predicted')
		# 	plt.ylim(-1,1)

		plt.hlines(y=0, xmin=region1[1], xmax=region1[2], color='black', linewidth=1, linestyle='dashed')

		j = 0
		for t_region in region_list2:
			plt.axvline(x=t_region[1], ymin=-1, ymax=1,color=color_vec1[j])
			plt.axvline(x=t_region[2], ymin=-1, ymax=1,color=color_vec1[j])
			j += 1

		if i==0:
			ax.legend(annotation,loc='upper left',frameon=False, labelspacing=0.5, prop={'size': 6})

		# plt.title('%d: %s L1:%.2e L2:%.2e'%(run_id,activation3,regularizer1,regularizer2))
		# plt.title('%s L1:%.2e L2:%.2e'%(activation3,regularizer1,regularizer2))
		plt.title('RT prediction')
		plt.ylabel('RT signal')
		plt.xlabel('Chr%d'%(region1[0]))
		if i<num1-1:
			ax.get_xaxis().set_ticks([])
			ax.xaxis.label.set_visible(False)
			# ax.set_xticklabels([])
		else:
			x_labels = ax.get_xticks()
			print(x_labels[0])
			ax.set_xticklabels(['%.2f'%(x/1e06) for x in x_labels])
			ax.text(0.95, -0.5, '(Mb)',transform=ax.transAxes,fontsize=8)
			# ax.text(1.0, -0.1, 'a',transform=ax.transAxes)
			# ax.text(1.0, -0.1, 'b',transform=ax.transAxes)
			# ax.text(1.0, -0.1, 'c',transform=ax.transAxes)

		fig.add_subplot(ax)

		region2 = region_list1[1]
		t_chrom, t_start, t_stop = region2[0], region2[1], region2[2]
		print(t_chrom, t_start, t_stop)
		t_chrom = 'chr%s'%(t_chrom)
		b = np.where((chrom2==t_chrom)&(start2<t_stop)&(stop2>t_start))[0]
		x2 = start2[b]
		print(len(b))
		predicted_attention2_region = predicted_attention2[b]

		value1, value2, value3 = np.max(predicted_attention2_region),np.min(predicted_attention2_region),np.median(predicted_attention2_region)
		print(value1,value2,value3)

		ax = fig.add_subplot(fig1[1])
		ax.plot(x2,predicted_attention2_region, marker='o', markerfacecolor='red', 
			markersize=3, color='red',linewidth=1,label='WT')

		t_value = 1
		j = 0
		for t_region in region_list2:
			plt.axvline(x=t_region[1], ymin=0, ymax=t_value,color=color_vec1[j])
			plt.axvline(x=t_region[2], ymin=0, ymax=t_value,color=color_vec1[j])
			j += 1
		
		plt.title('Estimated importance score')
		plt.xlabel('Chr%d'%(region1[0]))
		if i<num1-1:
			ax.get_xaxis().set_ticks([])
			ax.xaxis.label.set_visible(False)
		else:
			x_labels = ax.get_xticks()
			ax.set_xticklabels(['%.2f'%(x/1e06) for x in x_labels])
			ax.text(0.95, -0.5, '(Mb)',transform=ax.transAxes,fontsize=8)

		fig.add_subplot(ax)

	file_path1 = './mnt/yy3/data3/fig2'
	plt.savefig('%s/test%d_%d.%d.chr%d.png'%(file_path1,runid1_list[0][0],runid1_list[-1][1],sel_id_ori,region1[0]),dpi=300)

	# for i in range(1,num1+1):
	# 	plt.subplot(n_row,n_col,2*i-1)
	# 	plt.title('Signal')
	# 	for t_region in t_region_list:
	# 		x1, y1_region, y1_predicted_region = t_region[0], t_region[1], t_region[2]
	# 		plt.plot(x1,y1_region,color='blue',linewidth=2,linestyle='dashed',label='Del.-abc')
	# 		plt.plot(x1,y1_predicted_region,color='red',,linewidth=2,label='Del.-abc predicted')

	# 	plt.plot(x2,y2_region,color='orange',linewidth=2,linestyle='dashed',label='WT')
	# 	plt.plot(x2,y2_predicted_region,color='green',,linewidth=2,label='WT predicted')

	# 	plt.subplot(n_row,n_col,2*i)
	# 	plt.title('Estimated importance')
	# 	plt.plot(x1,y1_region,color='blue',linewidth=2,label='Del.-abc')
	# 	plt.plot(x1,y1_predicted_region,color='red',,linewidth=2,label='Del.-abc')
	# 	plt.plot(x1,y1_region,color='blue',linewidth=2,label='Del.-abc')
	# 	plt.plot(x1,y1_predicted_region,color='red',,linewidth=2,label='Del.-abc')

	# plt.suptitle('%s, l1: %.2e '%(regularizer1))

	return True

# plot and comparison
# def plot_1(file_path,method,runid1_list,runid2_list,method_vec,region_list1,region_list2,config_vec):
def plot_1_sub2(runid1_list,region_list1,region_list2,config_vec,output_file_path='',output_filename_prefix=''):
	
	# num1 = int(np.ceil(len(runid1_list)/2))
	# import matplotlib.pylab as pylab
	# params = {'legend.fontsize': 'x-large',
	# 	 'figure.figsize': (15, 5),
	# 	 'axes.labelsize': 'x-large',
	# 	 'axes.titlesize':'x-large',
	# 	 'xtick.labelsize':'x-large',
	# 	 'ytick.labelsize':'x-large'}
	params = {
		 'axes.labelsize': 8,
		 'axes.titlesize': 8,
		 'xtick.labelsize':8,
		 'ytick.labelsize':8}
	pylab.rcParams.update(params)

	num1 = len(runid1_list)
	n_col = 1
	n_row = int(np.ceil(num1/n_col))
	#  regularizer1 = 0.001

	fig = plt.figure(figsize=(10, 6))
	grid1 = gridspec.GridSpec(n_row, n_col, wspace=0.5, hspace=0.3)
	annotation = ['WT','WT predicted','Del.-abc','Del.-abc predicted']
	color_vec1 = ['black','seagreen','olive','blue','red','green']

	region1 = region_list1[0]
	t_region_list = []
	num2 = len(region_list2)
	t_region = [region1[0], region1[1], region_list2[0][1]]
	t_region_list.append(t_region)
	for i in range(0,num2-1):
		t_region = [region_list2[i][0],region_list2[i][2],region_list2[i+1][1]]
		t_region_list.append(t_region)
	
	t_region = [region_list2[num2-1][0],region_list2[num2-1][2],region1[2]]
	t_region_list.append(t_region)
	print(t_region_list)

	feature_id1 = -1
	feature_id1 = config_vec['feature_id']
	# prediction and attention value file
	filename1_list, filename2_list = config_vec['filename1_list'], config_vec['filename2_list']
	# prediction file
	filename1_list1, filename2_list1 = config_vec['filename1_list1'], config_vec['filename2_list1']
	thresh = config_vec['attention_thresh']
	# for run_id in runid2_list:
	# 	filename2 = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id,method)
	# 	filename2_list.append(filename2)

	# print(filename1_list,filename2_list,num1)
	for i in range(0,num1):
		print(i)
		filename1, filename2 = filename1_list[i], filename2_list[i]
		print(i,filename1,filename2)
		# config = config_vec[runid1_list[i][0]]
		run_id = runid1_list[i]

		# regularizer1, regularizer2 = 0, 0
		# if 'regularizer1' in config:
		# 	regularizer1 = config['regularizer1']
		# if 'regularizer2' in config:
		# 	regularizer2 = config['regularizer2']
		
		# activation3 = config['activation3']
		# print(activation3,regularizer1,regularizer2)

		sel_column = ['predicted_attention','Q2']
		sel_id_ori, sel_id_filter = 0, 0
		if 'sel_id' in config_vec:
			sel_id_ori = config_vec['sel_id']

		sel_id = sel_id_ori
		if sel_id_ori==2:
			sel_id=1
			sel_id_filter = 1

		sel_id1 = sel_column[sel_id]
		data1 = pd.read_csv(filename1,sep='\t')
		chrom1, start1, stop1 = np.asarray(data1['chrom']), np.asarray(data1['start']), np.asarray(data1['stop'])
		y_signal1, y_predicted1 = np.asarray(data1['signal']), np.asarray(data1['predicted_signal'])
		# predicted_attention1 = np.asarray(data1['predicted_attention'])
		# predicted_attention1 = np.asarray(data1['Q2'])
		predicted_attention1 = np.asarray(data1[sel_id1])

		data2 = pd.read_csv(filename2,sep='\t')
		chrom2, start2, stop2 = np.asarray(data2['chrom']), np.asarray(data2['start']), np.asarray(data2['stop'])
		y_signal2, y_predicted2 = np.asarray(data2['signal']), np.asarray(data2['predicted_signal'])
		predicted_attention2 = np.asarray(data2[sel_id1])

		# if prediction and attention are in different files, load the prediction file
		if len(filename1_list1)>0:
			filename1_1, filename2_1 = filename1_list1[i], filename2_list1[i]
			if os.path.exists(filename1_1)==True and os.path.exists(filename2_1)==True:
				data1_1 = pd.read_csv(filename1_1,sep='\t')
				t_chrom1, t_start1, t_stop1 = np.asarray(data1_1['chrom']), np.asarray(data1_1['start']), np.asarray(data1_1['stop'])
				y_predicted1 = np.asarray(data1_1['predicted_signal'])
				# predicted_attention1 = np.asarray(data1['predicted_attention'])

				data2_1 = pd.read_csv(filename2_1,sep='\t')
				t_chrom2, t_start2, t_stop2 = np.asarray(data2_1['chrom']), np.asarray(data2_1['start']), np.asarray(data2_1['stop'])
				y_predicted2 = np.asarray(data2_1['predicted_signal'])
				# predicted_attention2 = np.asarray(data2['predicted_attention'])
		
		if np.min(y_signal1)>-0.1:
			y_signal1 = 2*y_signal1-1
			y_predicted1 = 2*y_predicted1-1

		if np.min(y_signal2)>-0.1:
			y_signal2 = 2*y_signal2-1
			y_predicted2 = 2*y_predicted2-1

		if np.min(predicted_attention2)<0:
			predicted_attention2 = signal_normalize(predicted_attention2, [0,1])

		if sel_id_filter==1:
			predicted_attention2[predicted_attention2<thresh]=0

		fig1 = gridspec.GridSpecFromSubplotSpec(1, 2,
					subplot_spec=grid1[i], wspace=0.2, hspace=0.1)
		ax = fig.add_subplot(fig1[0])
		# plt.title('Signal: %s, L1: %.2e '%(activation,regularizer1))
		# plt.title(run_id)

		region1 = region_list1[0]
		t_chrom, t_start, t_stop = region1[0], region1[1], region1[2]
		t_chrom = 'chr%s'%(t_chrom)
		b = np.where((chrom2==t_chrom)&(start2<t_stop)&(stop2>t_start))[0]
		print(t_chrom, t_start, t_stop,len(b))
		x2, y2_region, y2_predicted_region = start2[b], y_signal2[b], y_predicted2[b]
		print(x2.shape,y2_region.shape,y2_predicted_region.shape)
		predicted_attention2_region = predicted_attention2[b]

		# x2 = ['%.2fMb'%(pos*1.0/1e06) for pos in x2]
		# print(x2[0:10])

		ax.plot(x2,y2_region,color='orange',linewidth=2,linestyle='dashed',label='WT')
		ax.plot(x2,y2_predicted_region,color='green',linewidth=2,label='WT predicted')

		for t_region in t_region_list:
			t_chrom_1, t_start, t_stop = t_region[0], t_region[1], t_region[2]
			t_chrom = 'chr%s'%(t_chrom_1)
			b = np.where((chrom1==t_chrom)&(start1<t_stop)&(stop1>t_start))[0]

			x1, y1_region, y1_predicted_region = start1[b], y_signal1[b], y_predicted1[b]
			ax.plot(x1,y1_region,color='blue',linewidth=2,linestyle='dashed',label='Del.-abc')
			ax.plot(x1,y1_predicted_region,color='red',linewidth=2,label='Del.-abc predicted')
			plt.ylim(-1,1)

		plt.hlines(y=0, xmin=region1[1], xmax=region1[2], color='black', linewidth=1, linestyle='dashed')

		j = 0
		for t_region in region_list2:
			plt.axvline(x=t_region[1], ymin=-1, ymax=1,color=color_vec1[j])
			plt.axvline(x=t_region[2], ymin=-1, ymax=1,color=color_vec1[j])
			j += 1

		if i==0:
			ax.legend(annotation,loc='upper left',frameon=False, labelspacing=0.5, prop={'size': 6})

		# plt.title('%d: %s L1:%.2e L2:%.2e'%(run_id,activation3,regularizer1,regularizer2))
		# plt.title('%s L1:%.2e L2:%.2e'%(activation3,regularizer1,regularizer2))
		# plt.title('RT prediction')
		plt.ylabel('RT signal')
		xlabel = 'Chr%d'%(t_chrom_1)
		plt.xlabel(xlabel)

		if i<num1-1:
			ax.get_xaxis().set_ticks([])
			ax.xaxis.label.set_visible(False)
			# ax.set_xticklabels([])
		else:
			x_labels = ax.get_xticks()
			print(x_labels[0])
			ax.set_xticklabels(['%.2f'%(x/1e06) for x in x_labels])
			ax.text(0.95, -0.5, '(Mb)',transform=ax.transAxes,fontsize=8)
			# ax.text(1.0, -0.1, 'a',transform=ax.transAxes)
			# ax.text(1.0, -0.1, 'b',transform=ax.transAxes)
			# ax.text(1.0, -0.1, 'c',transform=ax.transAxes)

		fig.add_subplot(ax)

		region2 = region_list1[1]
		t_chrom, t_start, t_stop = region2[0], region2[1], region2[2]
		print(t_chrom, t_start, t_stop)
		t_chrom = 'chr%s'%(t_chrom)
		b = np.where((chrom2==t_chrom)&(start2<t_stop)&(stop2>t_start))[0]
		x2 = start2[b]
		print(len(b))
		predicted_attention2_region = predicted_attention2[b]

		value1, value2, value3 = np.max(predicted_attention2_region),np.min(predicted_attention2_region),np.median(predicted_attention2_region)
		print(value1,value2,value3)

		ax = fig.add_subplot(fig1[1])
		ax.plot(x2,predicted_attention2_region, marker='o', markerfacecolor='red', 
			markersize=3, color='red',linewidth=1,label='WT')

		t_value = 1
		j = 0
		for t_region in region_list2:
			plt.axvline(x=t_region[1], ymin=0, ymax=t_value,color=color_vec1[j])
			plt.axvline(x=t_region[2], ymin=0, ymax=t_value,color=color_vec1[j])
			j += 1
		
		# plt.title('Estimated importance score')
		plt.xlabel(xlabel)
		if i<num1-1:
			ax.get_xaxis().set_ticks([])
			ax.xaxis.label.set_visible(False)
		else:
			x_labels = ax.get_xticks()
			ax.set_xticklabels(['%.2f'%(x/1e06) for x in x_labels])
			ax.text(0.95, -0.5, '(Mb)',transform=ax.transAxes,fontsize=8)

		fig.add_subplot(ax)

	if output_file_path == '':
		file_path1 = './mnt/yy3/data3/fig2'
	else:
		file_path1 = output_file_path
		
	plt.savefig('%s/test%d_%d.%s.pdf'%(file_path1,runid1_list[0][0],runid1_list[-1][1],output_filename_prefix),dpi=300)

	return True

# plot and comparison: histogram
def plot_2(filename1,filename2,output_filename):

	data1 = pd.read_csv(filename1,sep='\t')
	attention = data1['predicted_attention']
	label = data1['label']
	attention, label = np.asarray(attention), np.asarray(label)
	b1 = np.where(label==0)[0]
	b2 = np.where(label>0)[0]
	x1 = attention[b1]
	x2 = attention[b2]

	fig = plt.figure(figsize=(8, 6))
	# plt.style.use('ggplot')
	plt.hist(x1, bins=200, density=True, alpha=0.5, label='Non-ERCE')
	plt.hist(x2, bins=50, density=True, alpha=0.5, label='ERCE')
	plt.show()
	t1 = [skew(x1),np.median(x1),np.mean(x1),np.max(x1),np.min(x1)]
	t2 = [skew(x2),np.median(x2),np.mean(x2),np.max(x2),np.min(x2)]
	print('1',t1)
	print('2',t2)
	plt.legend(loc='upper right')
	plt.show()

	difference = 0
	if (t1[1]<=t2[1]) and (t1[2]<t2[2]):
		difference = 1

	if difference>0:
		plt.savefig(output_filename,dpi=300)
	else:
		print(difference)

	return difference, t1, t2

# plot and comparison: average performance
def plot_3(filename_list,output_filename,config):

	params = {
		 'axes.labelsize': 12,
		 'axes.titlesize': 16,
		 'xtick.labelsize':12,
		 'ytick.labelsize':12}
	pylab.rcParams.update(params)

	title_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal'} # Bottom vertical alignment for more space
	axis_font = {'fontname':'Arial', 'size':'16'}

	compare_annotation = config['compare_annotation']
	sel_idx = config['sel_idx']
	run_idlist = config['run_idlist']
	barWidth = config['barWidth']

	fig = plt.figure(figsize=(10,8))
	# sel_idx = [2,4,7]	# Pearsonr, explained variance, r2 score

	for l in range(1):
		# plt.subplot(2,3,l+1)
		# bars = ['NMI', 'AMI', 'ARI', 'Precision', 'Recall', 'F1']

		# bars = ['NMI', 'AMI', 'ARI']
		# vec1 = ['LR','XGBR','Single DNN','CONCERT']
		# bars = ['Pearson correlation','Explained variance','R2 score','Spearman correlation']
		bars = ['Pearson correlation','Spearman correlation','Explained variance','R2 score']
		# if sel_idx==[2,4]:
		# 	bars = bars[0:2]
		# bars = np.asarray(bars)
		# bars = list(bars[sel_idx])

		vec1 = compare_annotation
		y_pos = np.arange(len(bars))
		num1 = len(bars)	# number of evaluation metrics
		num2 = len(vec1)	# number of methods

		mean_value = np.zeros((num2,num1))
		std_vec = np.zeros((num2,num1))

		# run_idlist = [1,2,3]
		num3 = len(filename_list)
		for id2 in range(num3):
			filename_list1 = filename_list[id2]
			list1, chrom_idvec1, chrom_idvec2 = [], [], []
			for filename1 in filename_list1:
				data_1 = pd.read_csv(filename1,header=None,sep='\t')
				data_1 = np.asarray(data_1)
				list1.append(data_1)
				chrom_id = np.int64(data_1[:,0])
				b1 = np.where(chrom_id>0)[0]
				chrom_idvec1.extend(chrom_id[b1])
				chrom_idvec2.append(chrom_id[b1])

			chrom_idvec1 = np.sort(np.unique(chrom_idvec1))
			chrom_num = len(chrom_idvec1)
			print('chrom num',chrom_num,chrom_idvec2)

			num_train = len(chrom_idvec2)
			mtx1 = -np.ones((chrom_num,len(sel_idx),num_train))
			flag_vec1 = np.zeros((chrom_num,num_train))
			mtx2 = -np.ones((chrom_num,len(sel_idx)))
			flag_vec2 = np.zeros((chrom_num))
			for i in range(num_train):
				data_1 = list1[i]
				value1 = data_1[1:-1,sel_idx]
				id1 = mapping_Idx(chrom_idvec1,chrom_idvec2[i])
				b2 = np.where(id1<0)[0]
				if len(b2)>0:
					print('error!',chrom_idvec2[i][b2])
					return
				flag_vec1[id1,i] = 1
				mtx1[id1,:,i] = value1
				b1 = np.where(flag_vec2[id1]<=0)[0]
				flag_vec2[id1[b1]] = 1
				mtx2[id1[b1]] = value1[b1]

			if np.sum(flag_vec2)!=chrom_num:
				print('error!', np.where(flag_vec2<=0)[0])

			mean_value[id2] = np.mean(mtx2,axis=0)
			std_vec[id2] = np.std(mtx2,axis=0)/np.sqrt(chrom_num)

		# set width of bar
		# barWidth = 2	
		# Set position of bar on X axis
		print(mtx2)
		print(mean_value)
		c1 = barWidth+1
		t_pos1 = np.arange(0,num1*(num2*barWidth+c1),num2*barWidth+c1)
		# color_vec = ['#7f6d5f','#557f2d','#2d7f5e']
		if 'color_vec' in config:
			color_vec = config['color_vec']
		else:
			if num2>4:
				color_vec = ['steelblue','orange','mediumseagreen','wheat','olive','maroon','khaki','green','blue',
							 '#abd9e9','#d7191c','#fdae61','#ffffbf','#abdda4','#2b8']
			else:
				color_vec = ['steelblue','orange','olive','maroon']

		for i in range(0,num2):
			t_pos2 = [x + barWidth*i for x in t_pos1]
			plt.bar(t_pos2, mean_value[i], color=color_vec[i], width=barWidth, 
					edgecolor='white', 
					label=vec1[i], yerr=std_vec[i])

		# Add xticks on the middle of the group bars
		# plt.xlabel('Digits %d and %d'%(pair1[0],pair1[1]), fontweight='bold')
		# plt.title('Digits %d and %d'%(pair1[0],pair1[1]), fontweight='bold')
		plt.xticks([r + 1.5*barWidth for r in t_pos1], bars, fontname='Arial')
		run_id1, run_id2 = run_idlist[0][0], run_idlist[-1][0]
		# plt.title('I: %d II: %d'%(run_id1,run_id2),fontweight='bold')
		plt.title('RT prediction',**title_font)

		# Create legend & Show graphic
		if l==0:
			plt.legend(loc='upper right',frameon=False,labelspacing=0.1,prop={'size': 12})
		plt.show()
 
	# # Create bars
	# plt.bar(y_pos, y_value)
	# # Create names on the x-axis
	# plt.xticks(y_pos, bars)
	# # Show graphic
	# plt.show()

	plt.savefig(output_filename,dpi=300)

	return True

# plot and comparison: average performance
def plot_3_sub1(filename_list_1,output_filename,config):

	params = {
		 'axes.labelsize': 12,
		 'axes.titlesize': 16,
		 'xtick.labelsize':12,
		 'ytick.labelsize':12}
	pylab.rcParams.update(params)

	title_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal'} # Bottom vertical alignment for more space
	axis_font = {'fontname':'Arial', 'size':'16'}

	compare_annotation = config['compare_annotation']
	sel_idx = config['sel_idx']
	run_idlist = config['run_idlist']
	barWidth = config['barWidth']

	# fig = plt.figure(figsize=(10,8))
	fig = plt.figure(figsize=(30,28))
	# sel_idx = [2,4,7]	# Pearsonr, explained variance, r2 score

	for l in range(9):
		# plt.subplot(2,3,l+1)
		plt.subplot(3,3,l+1)
		cell_id, filename_list = filename_list_1[l]
		# bars = ['NMI', 'AMI', 'ARI', 'Precision', 'Recall', 'F1']

		# bars = ['NMI', 'AMI', 'ARI']
		# vec1 = ['LR','XGBR','Single DNN','CONCERT']
		# bars = ['Pearson correlation','Explained variance','R2 score','Spearman correlation']
		bars = ['Pearson correlation','Spearman correlation','Explained variance','R2 score']
		# if sel_idx==[2,4]:
		# 	bars = bars[0:2]
		# bars = np.asarray(bars)
		# bars = list(bars[sel_idx])

		vec1 = compare_annotation
		y_pos = np.arange(len(bars))
		num1 = len(bars)	# number of evaluation metrics
		num2 = len(vec1)	# number of methods

		mean_value = np.zeros((num2,num1))
		std_vec = np.zeros((num2,num1))

		# run_idlist = [1,2,3]
		num3 = len(filename_list)
		for id2 in range(num3):
			filename_list1 = filename_list[id2]
			list1, chrom_idvec1, chrom_idvec2 = [], [], []
			for filename1 in filename_list1:
				data_1 = pd.read_csv(filename1,header=None,sep='\t')
				data_1 = np.asarray(data_1)
				list1.append(data_1)
				chrom_id = np.int64(data_1[:,0])
				b1 = np.where(chrom_id>0)[0]
				chrom_idvec1.extend(chrom_id[b1])
				chrom_idvec2.append(chrom_id[b1])

			chrom_idvec1 = np.sort(np.unique(chrom_idvec1))
			chrom_num = len(chrom_idvec1)
			print('chrom num',chrom_num,chrom_idvec2)

			num_train = len(chrom_idvec2)
			mtx1 = -np.ones((chrom_num,len(sel_idx),num_train))
			flag_vec1 = np.zeros((chrom_num,num_train))
			mtx2 = -np.ones((chrom_num,len(sel_idx)))
			flag_vec2 = np.zeros((chrom_num))
			for i in range(num_train):
				data_1 = list1[i]
				value1 = data_1[1:-1,sel_idx]
				id1 = mapping_Idx(chrom_idvec1,chrom_idvec2[i])
				b2 = np.where(id1<0)[0]
				if len(b2)>0:
					print('error!',chrom_idvec2[i][b2])
					return
				flag_vec1[id1,i] = 1
				mtx1[id1,:,i] = value1
				b1 = np.where(flag_vec2[id1]<=0)[0]
				flag_vec2[id1[b1]] = 1
				mtx2[id1[b1]] = value1[b1]

			if np.sum(flag_vec2)!=chrom_num:
				print('error!', np.where(flag_vec2<=0)[0])

			mean_value[id2] = np.mean(mtx2,axis=0)
			std_vec[id2] = np.std(mtx2,axis=0)/np.sqrt(chrom_num)

		# set width of bar
		# barWidth = 2	
		# Set position of bar on X axis
		print(mtx2)
		print(mean_value)
		c1 = barWidth+1
		t_pos1 = np.arange(0,num1*(num2*barWidth+c1),num2*barWidth+c1)
		# color_vec = ['#7f6d5f','#557f2d','#2d7f5e']
		if 'color_vec' in config:
			color_vec = config['color_vec']
		else:
			if num2>4:
				color_vec = ['steelblue','orange','mediumseagreen','wheat','olive','maroon','khaki','green','blue',
							 '#abd9e9','#d7191c','#fdae61','#ffffbf','#abdda4','#2b8']
			else:
				color_vec = ['steelblue','orange','olive','maroon']

		for i in range(0,num2):
			t_pos2 = [x + barWidth*i for x in t_pos1]
			plt.bar(t_pos2, mean_value[i], color=color_vec[i], width=barWidth, 
					edgecolor='white', 
					label=vec1[i], yerr=std_vec[i])

		# Add xticks on the middle of the group bars
		# plt.xlabel('Digits %d and %d'%(pair1[0],pair1[1]), fontweight='bold')
		# plt.title('Digits %d and %d'%(pair1[0],pair1[1]), fontweight='bold')
		# plt.xticks([r + 1.5*barWidth for r in t_pos1], bars, fontname='Arial')
		plt.gca().set_xticks([])
		run_id1, run_id2 = run_idlist[0][0], run_idlist[-1][0]
		# plt.title('I: %d II: %d'%(run_id1,run_id2),fontweight='bold')
		# plt.title('RT prediction',**title_font)
		plt.title(cell_id,**title_font)
		plt.ylim(0,0.95)

		# Create legend & Show graphic
		if l==0:
			plt.legend(loc='upper right',frameon=False,labelspacing=0.1,prop={'size': 12})
		plt.show()
 
	# # Create bars
	# plt.bar(y_pos, y_value)
	# # Create names on the x-axis
	# plt.xticks(y_pos, bars)
	# # Show graphic
	# plt.show()

	plt.savefig(output_filename,dpi=300)

	return True

# plot and comparison: average performance
def plot_3_1(filename_list,output_filename,config):

	params = {
		 'axes.labelsize': 16,
		 'axes.titlesize': 18,
		 'xtick.labelsize':16,
		 'ytick.labelsize':16}
	pylab.rcParams.update(params)

	title_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal'} # Bottom vertical alignment for more space
	axis_font = {'fontname':'Arial', 'size':'16'}

	compare_annotation = config['compare_annotation']
	sel_idx = config['sel_idx']
	run_idlist = config['run_idlist']
	barWidth = config['barWidth']

	fig = plt.figure(figsize=(22,8))
	# sel_idx = [2,4,7]	# Pearsonr, explained variance, r2 score

	for l in range(2,3):
		# plt.subplot(2,3,l+1)
		# bars = ['NMI', 'AMI', 'ARI', 'Precision', 'Recall', 'F1']

		# bars = ['NMI', 'AMI', 'ARI']
		# vec1 = ['LR','XGBR','Single DNN','CONCERT']
		bars = ['Pearson correlation','Explained variance','R2 score']
		bars = ['%d'%(i) for i in range(1,23)]
		if sel_idx==[2,4]:
			bars = bars[0:2]

		vec1 = compare_annotation
		y_pos = np.arange(len(bars))
		num1 = len(bars)	# number of evaluation metrics
		num2 = len(vec1)	# number of methods

		mean_value = np.zeros((num2,num1))
		std_vec = np.zeros((num2,num1))

		# run_idlist = [1,2,3]
		num3 = len(filename_list)
		for id2 in range(num3):
			filename_list1 = filename_list[id2]
			list1, chrom_idvec1, chrom_idvec2 = [], [], []
			for filename1 in filename_list1:
				data_1 = pd.read_csv(filename1,header=None,sep='\t')
				data_1 = np.asarray(data_1)
				list1.append(data_1)
				chrom_id = np.int64(data_1[:,0])
				b1 = np.where(chrom_id>0)[0]
				chrom_idvec1.extend(chrom_id[b1])
				chrom_idvec2.append(chrom_id[b1])

			chrom_idvec1 = np.sort(np.unique(chrom_idvec1))
			chrom_num = len(chrom_idvec1)
			print('chrom num',chrom_num,chrom_idvec2)

			num_train = len(chrom_idvec2)
			mtx1 = -np.ones((chrom_num,len(sel_idx),num_train))
			flag_vec1 = np.zeros((chrom_num,num_train))
			mtx2 = -np.ones((chrom_num,len(sel_idx)))
			flag_vec2 = np.zeros((chrom_num))
			for i in range(num_train):
				data_1 = list1[i]
				value1 = data_1[1:-1,sel_idx]
				id1 = mapping_Idx(chrom_idvec1,chrom_idvec2[i])
				b2 = np.where(id1<0)[0]
				if len(b2)>0:
					print('error!',chrom_idvec2[i][b2])
					return
				flag_vec1[id1,i] = 1
				mtx1[id1,:,i] = value1
				b1 = np.where(flag_vec2[id1]<=0)[0]
				flag_vec2[id1[b1]] = 1
				mtx2[id1[b1]] = value1[b1]

			if np.sum(flag_vec2)!=chrom_num:
				print('error!', np.where(flag_vec2<=0)[0])

			# mean_value[id2] = np.mean(mtx2,axis=0)
			# std_vec[id2] = np.std(mtx2,axis=0)/np.sqrt(chrom_num)
			mean_value[id2] = mtx2[:,l]

		# set width of bar
		# barWidth = 2	
		# Set position of bar on X axis
		print(mtx2)
		print(mean_value)
		t_pos1 = np.arange(0,num1*(num2*barWidth+2),num2*barWidth+2)
		# color_vec = ['#7f6d5f','#557f2d','#2d7f5e']
		if num2==3:
			color_vec = ['steelblue','orange','maroon']
		elif num2==4:
			color_vec = ['steelblue','orange','olive','maroon']
		else:
			color_vec = ['steelblue','orange','mediumseagreen','yellow','olive','maroon']
		for i in range(0,num2):
			t_pos2 = [x + barWidth*i for x in t_pos1]
			plt.bar(t_pos2, mean_value[i], color=color_vec[i], width=barWidth, 
					edgecolor='white', 
					label=vec1[i], yerr=std_vec[i])

		# Add xticks on the middle of the group bars
		# plt.xlabel('Digits %d and %d'%(pair1[0],pair1[1]), fontweight='bold')
		# plt.title('Digits %d and %d'%(pair1[0],pair1[1]), fontweight='bold')
		plt.xticks([r + 1.5*barWidth for r in t_pos1], bars)
		run_id1, run_id2 = run_idlist[0][0], run_idlist[-1][0]
		# plt.title('I: %d II: %d'%(run_id1,run_id2),fontweight='bold')
		plt.title('RT prediction',**title_font)
		plt.ylim(0,1)

		# Create legend & Show graphic
		if l>=0:
			plt.legend(loc='upper left', frameon=False,labelspacing=0.1,prop={'size': 12})
		plt.show()
 
	# # Create bars
	# plt.bar(y_pos, y_value)
	# # Create names on the x-axis
	# plt.xticks(y_pos, bars)
	# # Show graphic
	# plt.show()

	plt.savefig(output_filename,dpi=300)

	return True

def compute_mean_std_ori(filename_list,run_idlist):
	# run_idlist = [1,2,3]
	num3 = len(filename_list)
	mean_value = np.zeros((num2,num1))
	std_vec = np.zeros((num2,num1))
	dict1 = dict()

	for id2 in range(num3):
		run_id1 = run_idlist[id2]
		filename_list1 = filename_list[id2]
		list1, chrom_idvec1, chrom_idvec2 = [], [], []
		for filename1 in filename_list1:
			data_1 = pd.read_csv(filename1,header=None,sep='\t')
			data_1 = np.asarray(data_1)
			list1.append(data_1)
			chrom_id = np.int64(data_1[:,0])
			b1 = np.where(chrom_id>0)[0]
			chrom_idvec1.extend(chrom_id[b1])
			chrom_idvec2.append(chrom_id[b1])

		chrom_idvec1 = np.sort(np.unique(chrom_idvec1))
		chrom_num = len(chrom_idvec1)
		print('chrom num',chrom_num,chrom_idvec2)

		num_train = len(chrom_idvec2)
		mtx1 = -np.ones((chrom_num,len(sel_idx),num_train))
		flag_vec1 = np.zeros((chrom_num,num_train))
		mtx2 = -np.ones((chrom_num,1+len(sel_idx)))
		flag_vec2 = np.zeros((chrom_num))
		for i in range(num_train):
			data_1 = list1[i]
			value1 = data_1[1:-1,sel_idx]
			id1 = mapping_Idx(chrom_idvec1,chrom_idvec2[i])
			b2 = np.where(id1<0)[0]
			if len(b2)>0:
				print('error!',chrom_idvec2[i][b2])
				return
			flag_vec1[id1,i] = 1
			mtx1[id1,:,i] = value1
			b1 = np.where(flag_vec2[id1]<=0)[0]
			flag_vec2[id1[b1]] = 1
			mtx2[id1[b1],1:] = value1[b1]

		if np.sum(flag_vec2)!=chrom_num:
			print('error!', np.where(flag_vec2<=0)[0])

		t_mean_value = np.mean(mtx2,axis=0)
		t_std_vec = np.std(mtx2,axis=0)/np.sqrt(chrom_num)

		dict1[run_id1] = {'mtx':mtx2,'mean_value':t_mean_value,'std':t_std_vec}

	return dict1

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
	

def plot_3_2(output_filename,config):

	params = {
		 'axes.labelsize': 16,
		 'axes.titlesize': 18,
		 'xtick.labelsize':16,
		 'ytick.labelsize':16}
	pylab.rcParams.update(params)
	pylab.rcParams.update(params)

	compare_annotation = config['compare_annotation']
	sel_idx = config['sel_idx']
	run_idlist = config['run_idlist']
	barWidth = config['barWidth']
	mean_value = config['mean_value']
	std_vec = config['std']
	bars = config['metrics']
	title1 = config['fig_title']

	fig = plt.figure(figsize=(10,8))
	# sel_idx = [2,4,7]	# Pearsonr, explained variance, r2 score

	for l in range(1):
		# bars = ['Pearson correlation','Explained variance','R2 score','Spearman correlation']
		# if sel_idx==[2,4]:
		# 	bars = bars[0:2]
		bars = np.asarray(bars)
		bars = list(bars[sel_idx])

		vec1 = compare_annotation
		y_pos = np.arange(len(bars))
		num1 = len(bars)	# number of evaluation metrics
		num2 = len(vec1)	# number of methods

		# set width of bar
		# barWidth = 2	
		# Set position of bar on X axis
		# print(mtx2)
		print(mean_value)
		c1 = barWidth+1
		t_pos1 = np.arange(0,num1*(num2*barWidth+c1),num2*barWidth+c1)
		# color_vec = ['#7f6d5f','#557f2d','#2d7f5e']
		if 'color_vec' in config:
			color_vec = config['color_vec']
		else:
			if num2>4:
				color_vec = ['steelblue','orange','mediumseagreen','wheat','olive','maroon','khaki','green','blue',
							 '#abd9e9','#d7191c','#fdae61','#ffffbf','#abdda4','#2b8']
			else:
				color_vec = ['steelblue','orange','olive','maroon']

		for i in range(0,num2):
			t_pos2 = [x + barWidth*i for x in t_pos1]
			plt.bar(t_pos2, mean_value[i], color=color_vec[i], width=barWidth, 
					edgecolor='white', 
					label=vec1[i], yerr=std_vec[i])

		# Add xticks on the middle of the group bars
		# plt.xlabel('Digits %d and %d'%(pair1[0],pair1[1]), fontweight='bold')
		# plt.title('Digits %d and %d'%(pair1[0],pair1[1]), fontweight='bold')
		plt.xticks([r + 1.5*barWidth for r in t_pos1], bars)
		run_id1, run_id2 = run_idlist[0][0], run_idlist[-1][0]
		# plt.title('I: %d II: %d'%(run_id1,run_id2),fontweight='bold')
		plt.title(title1)

		# Create legend & Show graphic
		if l==0:
			plt.legend(loc='upper right', frameon=False,labelspacing=0.1,prop={'size': 12})
		plt.show()
 
	# # Create bars
	# plt.bar(y_pos, y_value)
	# # Create names on the x-axis
	# plt.xticks(y_pos, bars)
	# # Show graphic
	# plt.show()
	plt.savefig(output_filename,dpi=300)

	return True

# plot of distribution
def plot_5(filename_list,output_filename,config):

	params = {
		 'axes.labelsize': 16,
		 'axes.titlesize': 18,
		 'xtick.labelsize':16,
		 'ytick.labelsize':16}
	pylab.rcParams.update(params)

	sns.set(style="whitegrid")
	tips = sns.load_dataset("tips")
	ax = sns.violinplot(x=tips["total_bill"])

	ax = sns.violinplot(x="day", y="total_bill", data=tips)

	ax = sns.violinplot(x="day", y="total_bill", hue="smoker",
					data=tips, palette="muted")

	ax = sns.violinplot(x="day", y="total_bill", hue="smoker",
					data=tips, palette="muted", split=True)

	ax = sns.violinplot(x="time", y="tip", data=tips,
					order=["Dinner", "Lunch"])

	ax = sns.violinplot(x="day", y="total_bill", hue="sex",
					data=tips, palette="Set2", split=True,
					scale="count")


	plt.savefig(output_filename,dpi=300)

	return True

# # input is logits
# class Sample_Concrete(Layer):
# 	"""
# 	Layer for sample Concrete / Gumbel-Softmax variables. 

# 	"""
# 	def __init__(self, tau0, k, n_steps, **kwargs): 
# 	# def __init__(self, tau0, k, n_steps): 
# 		self.tau0 = tau0
# 		self.k = k
# 		self.n_steps = n_steps
# 		super(Sample_Concrete, self).__init__(**kwargs)

# 	def call(self, logits):
# 		logits_ = K.permute_dimensions(logits, (0,2,1))
# 		#[batchsize, 1, MAX_SENTS]

# 		unif_shape = tf.shape(logits_)[0]
# 		uniform = tf.random.uniform(shape =(unif_shape, self.k, self.n_steps), 
# 			minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
# 			maxval = 1.0)

# 		gumbel = - K.log(-K.log(uniform))
# 		# noisy_logits = (gumbel + logits_)/self.tau0
# 		# logits_ = K.log(logits_) # the input is probability
# 		noisy_logits = (gumbel + logits_)/self.tau0
# 		samples = K.softmax(noisy_logits)
# 		samples = K.max(samples, axis = 1)
# 		samples = K.expand_dims(samples, -1)

# 		discrete_logits = K.one_hot(K.argmax(logits_,axis=-1), num_classes = self.n_steps)
# 		discrete_logits = K.permute_dimensions(discrete_logits,(0,2,1))

# 		# return K.in_train_phase(samples, discrete_logits)
# 		return samples

# 	def compute_output_shape(self, input_shape):
# 		return input_shape

# # input is probability
# class Sample_Concrete1(Layer):
# 	"""
# 	Layer for sample Concrete / Gumbel-Softmax variables. 

# 	"""
# 	def __init__(self, tau0, k, n_steps, type_id, **kwargs): 
# 	# def __init__(self, tau0, k, n_steps): 
# 		self.tau0 = tau0
# 		self.k = k
# 		self.n_steps = n_steps
# 		self.type_id = type_id
# 		super(Sample_Concrete1, self).__init__(**kwargs)

# 	def call(self, logits):
# 		logits_ = K.permute_dimensions(logits, (0,2,1))
# 		#[batchsize, 1, MAX_SENTS]

# 		unif_shape = tf.shape(logits_)[0]
# 		uniform = tf.random.uniform(shape =(unif_shape, self.k, self.n_steps), 
# 			minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
# 			maxval = 1.0)

# 		gumbel = - K.log(-K.log(uniform))
# 		eps = tf.compat.v1.keras.backend.constant(1e-12)
# 		# print('eps:', eps)
# 		# noisy_logits = (gumbel + logits_)/self.tau0
# 		# logits_ = K.log(logits_) # the input is probability
# 		if self.type_id==2:
# 			logits_ = -K.log(-K.log(logits_ + eps))	# the input is probability
# 		elif self.type_id==3:
# 			logits_ = K.log(logits_ + eps) # the input is probability
# 		# elif self.type_id==5:
# 		# 	logits_ = -logits_
# 		elif self.type_id==5:
# 			eps1 = tf.compat.v1.keras.backend.constant(1+1e-12)
# 			# x = Lambda(lambda x: x * 2)(layer)
# 			logits_ = K.log(logits_ + eps1)
# 		else:
# 			pass
# 		noisy_logits = (gumbel + logits_)/self.tau0
# 		samples = K.softmax(noisy_logits)
# 		samples = K.max(samples, axis = 1)
# 		samples = K.expand_dims(samples, -1)

# 		discrete_logits = K.one_hot(K.argmax(logits_,axis=-1), num_classes = self.n_steps)
# 		discrete_logits = K.permute_dimensions(discrete_logits,(0,2,1))

# 		# return K.in_train_phase(samples, discrete_logits)
# 		return samples

# 	def compute_output_shape(self, input_shape):
# 		return input_shape

# TODO: Write any helper functions that you need
def Gaussian_loss(self, targets, params):

	mean, logvar = self.get_output(params)
	t1 = -0.5*K.sum(K.square(targets-mean)/(K.exp(logvar)),axis=1)
	t2 = -0.5*K.sum(logvar,axis=1)
	# t3 = -0.5*self.state_dim*np.log(2*pi)

	negative_loglikelihood = -K.mean(t1+t2)

	return negative_loglikelihood

# TODO: Write any helper functions that you need
def MSE_loss(self, targets, logits_T):

	mean, logvar = self.get_output(params)
	t1 = -0.5*K.sum(K.square(targets-mean)/(K.exp(logvar)),axis=1)
	t2 = -0.5*K.sum(logvar,axis=1)
	# t3 = -0.5*self.state_dim*np.log(2*pi)

	negative_loglikelihood = -K.mean(t1+t2)

	return negative_loglikelihood

# construct gumbel selector 1
def construct_gumbel_selector(input_1, bin_size, feature_dim, n_steps, feature_dim_vec1):

	# encoder_1 = TimeDistributed(sentEncoder)(review_input) # [batch_size, max_sents, 100] 
	# net = encoder_1
	n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0], feature_dim_vec1[1], feature_dim_vec1[2], feature_dim_vec1[3], feature_dim_vec1[4], feature_dim_vec1[5]
	if n_filter1>0:
		layer_1 = Conv1D(n_filter1, 3, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(input_1)
	else:
		layer_1 = input_1

	# local info
	if n_local_conv>0:
		layer_2 = Conv1D(n_filter2, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(layer_1)
		if n_local_conv>1:
			local_info = Conv1D(n_filter2, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(layer_2)
		else:
			local_info = layer_2
	else:
		local_info = layer_1

	# global info, shape: (batch_size, feature_dim)
	if concat>0:
		x1 = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(layer_1)
		if dim1>0:
			global_info = Dense(dim1, name = 'new_dense_1', activation='relu')(x1)
		else:
			global_info = x1

		# concatenated feature, shape: (batch_size, n_steps, dim1+dim2)
		x2 = Concatenate_1()([global_info,local_info])
	else:
		x2 = local_info

	# x2 = Dropout(0.2, name = 'new_dropout_2')(x2)
	if dim2>0:
		x2 = Conv1D(dim2, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(x2)

	# logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(x2)
	logits_T = TimeDistributed(Dense(1,name='dense_1'))(x2)
	logits_T = TimeDistributed(BatchNormalization(name='batchnorm_1'))(logits_T)
	logits_T = TimeDistributed(Activation("sigmoid",name='activation_1'))(logits_T)

	return logits_T

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

# construct gumbel selector 1
def construct_gumbel_selector2(input1,input2,config,number=1,type_id=1):

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

	# local info
	# if n_local_conv>0:
	# 	layer_2 = Conv1D(n_filter2, 3, padding='same', activation=activation1, strides=1, name = 'conv2_gumbel_%d'%(number))(layer_1)
	# 	if n_local_conv>1:
	# 		local_info = Conv1D(n_filter2, 3, padding='same', activation=activation1, strides=1, name = 'conv3_gumbel_%d'%(number))(layer_2)
	# 	else:
	# 		local_info = layer_2
	# else:
	# 	local_info = layer_1

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


	number = number + 1
	layer_2 = Dense(dim_2, name='new_dense_%d'%(number), activation=activation1)(input2)
	layer_2 = RepeatVector(n_steps)(layer_2)

	x2 = Concatenate_1()([x2,layer_2])
		
	# layer_2 = Conv1D(n_filter1, 1, padding='same', strides=1, name = 'conv1_1_gumbel_%d'%(number))(dense_layer_1)
	# layer_2 = BatchNormalization()(layer_1)
	# layer_2 = Activation(activation1, name='conv1_gumbel_%d'%(number))(layer_1)


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

# construct gumbel selector 1
def construct_gumbel_selector1_ori(input1,config,number=1,type_id=1):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation1 = config['activation']
	# activation1 = 'relu'
	feature_dim_vec1 = config['feature_dim_vec']

	# input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		# encode the input, shape: (batch_size,n_steps,units_1)
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)	
	else:
		dense_layer_1 = input1

	# default: n_filter1:50, dim1:25, n_filter2: 50, dim2: 25, n_local_conv: 0, concat: 0
	n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0], feature_dim_vec1[1], feature_dim_vec1[2], feature_dim_vec1[3], feature_dim_vec1[4], feature_dim_vec1[5]
	if n_filter1>0:
		layer_1 = Conv1D(n_filter1, 3, padding='same', activation=activation1, strides=1, name = 'conv1_gumbel_%d'%(number))(dense_layer_1)
	else:
		layer_1 = dense_layer_1

	# local info
	if n_local_conv>0:
		layer_2 = Conv1D(n_filter2, 3, padding='same', activation=activation1, strides=1, name = 'conv2_gumbel_%d'%(number))(layer_1)
		if n_local_conv>1:
			local_info = Conv1D(n_filter2, 3, padding='same', activation=activation1, strides=1, name = 'conv3_gumbel_%d'%(number))(layer_2)
		else:
			local_info = layer_2
	else:
		local_info = layer_1

	# global info, shape: (batch_size, feature_dim)
	if concat>0:
		x1 = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_%d'%(numbwer))(layer_1)
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
		x2 = Conv1D(dim2, 1, padding='same', activation=activation1, strides=1, name = 'conv4_gumbel_%d'%(number))(x2)
	if ('batchnorm1' in config) and config['batchnorm1']==1:
		x2 = TimeDistributed(BatchNormalization(),name ='conv4_gumbel_bn%d'%(number))(x2)

	if 'regularizer1' in config:
		regularizer1 = config['regularizer1']
	else:
		regularizer1 = 1e-04

	if 'regularizer2' in config:
		regularizer2 = config['regularizer2']
	else:
		regularizer2 = 1e-04

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
			# x2 = TimeDistributed(Activation(activation2,name='activation_1_%d'%(number)))(x2)
			if 'regularizer_1' in config and config['regularizer_1']==1:
				print('regularization after activation',activation3)
				print(regularizer1,regularizer2)
				x2 = TimeDistributed(Dense(1,
									activation=activation3,
									kernel_regularizer=regularizers.l2(regularizer2),
									activity_regularizer=regularizers.l1(regularizer1),
									),name='logits_T_%d'%(number))(x2)
				# x2 = TimeDistributed(Activation(activation3,
				# 					activity_regularizer=regularizers.l1(regularizer1)),
				# 					name='logits_T_%d'%(number))(x2)
			else:
				x2 = TimeDistributed(Dense(1),name='dense_1_%d'%(number))(x2)
				x2 = TimeDistributed(BatchNormalization(),name='batchnorm_1_%d'%(number))(x2)
				x2 = TimeDistributed(Activation(activation3),name='logits_T_%d'%(number))(x2)
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

def get_model2a1_basic1_1(input1,config):

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

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation2!='':
		x1 = Activation(activation2,name='activation1')(x1)

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

# def get_model2a1_attention_1_2_2_sample(input_shape,config):

# 	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
# 	n_steps = config['context_size']
# 	lr = config['lr']
# 	activation = config['activation']
# 	activation_self = config['activation_self']
# 	activation3 = config['activation3']
# 	typeid_sample = config['typeid_sample']
# 	if not('loss_function' in config):
# 		loss_function = 'mean_squared_error'
# 	else:
# 		loss_function = config['loss_function']

# 	input1 = Input(shape = (n_steps,feature_dim))

# 	number = 1
# 	# if 'typeid2' in config:
# 	# 	typeid = config['typeid2']
# 	# else:
# 	# 	typeid = 2
# 	typeid = 2
# 	logits_T = construct_gumbel_selector1(input1,config,number,typeid)

# 	# k = 10
# 	if not('sample_attention' in config) or config['sample_attention']==1:
# 		tau = 0.5
# 		k = 5
# 		print('sample_attention',tau,k,typeid,activation3,typeid_sample)
# 		if 'tau' in config:
# 			tau = config['tau']
# 		if 'n_select' in config:
# 			k = config['n_select']
# 		if typeid<2:
# 			attention1 = Sample_Concrete(tau,k,n_steps)(logits_T)
# 		else:
# 			if activation3=='linear':
# 				typeid_sample = 1
# 			elif activation3=='tanh':
# 				typeid_sample = 5
# 			else:
# 				pass
# 			attention1 = Sample_Concrete1(tau,k,n_steps,typeid_sample)(logits_T) # output shape: (batch_size, n_steps, 1)
# 	else:
# 		attention1 = logits_T
# 	# attention1 = logits_T

# 	# attention1 = Flatten()(attention1)
# 	# attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
# 	# attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)

# 	# encode the input 2
# 	units_2 = config['units2']
# 	if units_2>0:
# 		dim2 = units_2
# 		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
# 	else:
# 		dim2 = feature_dim
# 		dense_layer_output1 = input1

# 	if config['select2']==1:
# 		units1 = config['units1']
# 		config['units1'] = 0
# 		typeid = 0
# 		number = 2
# 		dense_layer_output1 = construct_gumbel_selector1(dense_layer_output1,config,number,typeid)
# 		config['units1'] = units1

# 	layer_1 = Multiply()([dense_layer_output1, attention1])

# 	output = get_model2a1_basic1(layer_1,config)

# 	# output = Activation("softmax")(output)
# 	model = Model(input = input1, output = output)
# 	# adam = Adam(lr = lr)
# 	adam = Adam(lr = lr, clipnorm=CLIPNORM1)
# 	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
# 	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
# 	# model.compile(adam,loss = 'mean_absolute_percentage_error')
# 	model.compile(adam,loss = loss_function)

# 	model.summary()

# 	return model

def get_model_pre():

	feature_dim, n_steps = config['feature_dim'], config['context_size']
	lr = config['lr']

	input1 = Input(shape = (n_steps,feature_dim))

	number, typeid = 1, 2
	output = construct_gumbel_selector1(input1,config,number,typeid)

	# tau = 0.5
	# k = 5
	# if 'tau' in config:
	# 	tau = config['tau']
	# if 'n_select' in config:
	# 	k = config['n_select']

	# # k = 10
	# if typeid<2:
	# 	attention1 = Sample_Concrete(tau,k,n_steps,typeid)(logits_T)
	# else:
	# 	if activation3=='linear':
	# 		typeid_sample = 1
	# 	attention1 = Sample_Concrete1(tau,k,n_steps,typeid_sample)(logits_T) # output shape: (batch_size, n_steps, 1)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	# adam = Adam(lr = lr)
	if CLIPNORM1>0:
		adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	else:
		adam = Adam(lr = lr)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss='mean_squared_error')

	model.summary()

	return model

# method 11: network + self-attention
def get_model2a1_attention1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (None,feature_dim))
	lr = config['lr']
	activation = config['activation']

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')

	# method 23
	if config['attention1']==1:
		units_1 = config['units1']
		if units_1>0:
			dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)
			dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_1)
		else:
			dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(input1)
		
		# attention1 = Flatten()(dense_layer_2)
		# attention1 = Activation('softmax',name='attention1')(attention1)
		# attention1 = TimeDistributed(Activation("sigmoid",name='activation_1'))(dense_layer_2)
		# attention1 = Flatten()(attention1)
		x1, attention_1 = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention_1')(dense_layer_2)
	else:
		x1 = input1

	x1 = biLSTM_layer1(x1)
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

# network1 for estimating weights, self-attention and network2 for predicting signals
# method 15: network2 + self-attention
# method 18: network2

# multiple convolution layers, self-attention
# method 31
def get_model2a1_attention1_1_ori(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']

	input1 = Input(shape = (n_steps,feature_dim))

	layer_1 = construct_gumbel_selector1(input1,config)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(layer_1)
	x1, attention = SeqSelfAttention(return_attention=True, attention_activation=activation_self,name='attention1')(layer_1)

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(x1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation!='':
		x1 = Activation(activation,name='activation1')(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if config['attention2']==1:
		x1, attention = SeqSelfAttention(return_attention=True, attention_activation=activation_self,name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(x1)
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

# method 21 and 22: network1 for estimating weights, self-attention and network2 for predicting signals
# without gumbel sampling
def get_model2a1_attention_1_2_2(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	feature_dim_vec1 = config['feature_dim_vec']
	activation_self = config['activation_self']

	input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		# encode the input, shape: (batch_size,n_steps,units_1)
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)	
	else:
		dense_layer_1 = input1

	# calculate attention
	# dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_1)
	# dense_layer_2 = TimeDistributed(BatchNormalization(name='batchnorm_1'))(dense_layer_2)
	# attention1 = TimeDistributed(Activation("sigmoid",name='activation_1'))(dense_layer_2)
	# attention1 = Flatten()(attention1)

	# feature_dim_vec = [50,25,50,25,0,0]
	# feature_dim_vec1 = [50,25,50,25,0,0]
	# feature_dim_vec1 = config['feature_dim_vec']
	# print(feature_dim_vec1)
	# shape: (batch_size,n_steps,1)
	# logits_T = construct_gumbel_selector(dense_layer_1, feature_dim, feature_dim, n_steps, feature_dim_vec)

	input_1 = dense_layer_1
	n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0], feature_dim_vec1[1], feature_dim_vec1[2], feature_dim_vec1[3], feature_dim_vec1[4], feature_dim_vec1[5]
	if n_filter1>0:
		layer_1 = Conv1D(n_filter1, 3, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(input_1)
	else:
		layer_1 = input_1

	# local info
	if n_local_conv>0:
		layer_2 = Conv1D(n_filter2, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(layer_1)
		if n_local_conv>1:
			local_info = Conv1D(n_filter2, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(layer_2)
		else:
			local_info = layer_2
	else:
		local_info = layer_1

	# global info, shape: (batch_size, feature_dim)
	if concat>0:
		x1 = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(layer_1)
		if dim1>0:
			global_info = Dense(dim1, name = 'new_dense_1', activation='relu')(x1)
		else:
			global_info = x1

		# concatenated feature, shape: (batch_size, n_steps, dim1+dim2)
		x2 = Concatenate_1()([global_info,local_info])
	else:
		x2 = local_info

	# x2 = Dropout(0.2, name = 'new_dropout_2')(x2)
	# current configuration: dense1 + conv1 + conv2
	if dim2>0:
		x2 = Conv1D(dim2, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(x2)

	# logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(x2)
	logits_T = TimeDistributed(Dense(1,name='dense_1'))(x2)
	logits_T = TimeDistributed(BatchNormalization(name='batchnorm_1'))(logits_T)
	# logits_T = TimeDistributed(Activation("sigmoid",name='activation_1'),name='logits_T')(logits_T)
	logits_T = TimeDistributed(Activation("sigmoid",name='activation_1'),name='logits_T_1')(logits_T)

	# tau = 0.5
	# k = config['n_select']
	# k = 10
	tau, k = config['tau'], config['n_select']
	# attention1 = Sample_Concrete(tau, k, n_steps)(logits_T) # output shape: (batch_size, n_steps, 1)
	attention1 = logits_T

	# encode the input 2
	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	attention1 = Flatten()(attention1)
	attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)
	layer_1 = Multiply()([dense_layer_output1, attention1])
	# dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)

	# method 21: attention1:1, method 22: attention1:0
	if config['attention1']==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, attention_activation=activation_self,name='attention1')(layer_1)

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation!='':
		x1 = Activation(activation,name='activation1')(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if config['attention2']==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, attention_activation=activation_self,name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(x1)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation("sigmoid",name='activation2')(output)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	adam = Adam(lr = lr)
	# adam = Adam(lr = lr, clipnorm=1.0)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

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
	# adam = Adam(lr = lr)
	adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
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

	# n_step_local,feature_dim = select_config['input_shape_1']
	# n_step_local = select_config['n_step_local_ori']
	return_sequences_flag, sample_local, pooling_local = select_config['local_vec_1']
	# print(feature_dim1, feature_dim2, return_sequences_flag)

	# conv_list1 = config['local_vec_1'] # 0: conv_layers: 1: bilstm 2: processing output of lstm 3: dense_layers 
	conv_list_ori = select_config['local_conv_list_ori'] # 0: conv_layers: 1: bilstm 2: processing output of lstm 3: dense_layers 

	# input_local = Input(shape=(n_step_local,feature_dim))
	# lstm_1 = Bidirectional(LSTM(feature_dim1, name = 'lstm_1'), 
	# 		name = 'bidirectional')(embedded_sequences)

	# layer_1 = TimeDistributed(Conv1D(feature_dim1,1,padding='same',activation=None,strides=1),name='conv_local_1')(input_local)
	# layer_1 = TimeDistributed(BatchNormalization(),name='batchnorm_local_1')(layer_1)
	# layer_1 = TimeDistributed(Activation('relu'),name='activation_local_1')(layer_1)

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
		
	# biLSTM_layer_1 = Bidirectional(LSTM(input_shape=(n_step_local, n_filters), 
	# 								units=feature_dim1,
	# 								return_sequences = return_sequences_flag,
	# 								kernel_regularizer = keras.regularizers.l2(1e-5),
	# 								dropout=0.1,
	# 								recurrent_dropout = 0.1),name='bilstm_local_1_1')

	# regularizer2 = select_config['regularizer2_2']
	# biLSTM_layer_1 = Bidirectional(LSTM(units=feature_dim1,
	# 								return_sequences = return_sequences_flag,
	# 								kernel_regularizer = keras.regularizers.l2(regularizer2),
	# 								dropout=0.1,
	# 								recurrent_dropout = 0.1),name='bilstm_local_1_1')

	# x1 = biLSTM_layer_1(layer_1)

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

# method 32, method 51 and 52: network1 for estimating weights, self-attention and network2 for predicting signals
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
	# adam = Adam(lr = lr)
	adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = loss_function)

	model.summary()

	return model

# method 32, method 51 and 52: network1 for estimating weights, self-attention and network2 for predicting signals
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
	# adam = Adam(lr = lr)
	adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = loss_function)

	model.summary()

	return model

# method 32, method 51 and 52: network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling and multiple convolution layers
def get_model2a1_attention_1_2_2_sample_basic(input1,config):

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

	# input1 = Input(shape = (n_steps,feature_dim))

	number = 1
	# if 'typeid2' in config:
	# 	typeid = config['typeid2']
	# else:
	# 	typeid = 2
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
	# attention1 = logits_T

	# attention1 = Flatten()(attention1)
	# attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	# attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)

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

	return output

# method 31 and 32: network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling
def get_model2a1_attention_1_2_2_sample1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']

	input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		# encode the input, shape: (batch_size,n_steps,units_1)
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)	
	else:
		dense_layer_1 = input1

	feature_dim_vec1 = config['feature_dim_vec']
	
	input_1 = dense_layer_1
	n_filter1, dim1, n_filter2, dim2, n_local_conv, concat = feature_dim_vec1[0], feature_dim_vec1[1], feature_dim_vec1[2], feature_dim_vec1[3], feature_dim_vec1[4], feature_dim_vec1[5]
	if n_filter1>0:
		layer_1 = Conv1D(n_filter1, 3, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(input_1)
	else:
		layer_1 = input_1

	# global info, shape: (batch_size, feature_dim)
	if concat>0:
		x1 = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(layer_1)
		if dim1>0:
			global_info = Dense(dim1, name = 'new_dense_1', activation='relu')(x1)
		else:
			global_info = x1

	# local info
	if n_local_conv>0:
		layer_2 = Conv1D(n_filter2, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(layer_1)
		if n_local_conv>1:
			local_info = Conv1D(n_filter2, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(layer_2)
		else:
			local_info = layer_2
	else:
		local_info = layer_1

	# concatenated feature, shape: (batch_size, n_steps, dim1+dim2)
	if concat>0:
		x2 = Concatenate_1()([global_info,local_info])
	else:
		x2 = local_info

	# x2 = Dropout(0.2, name = 'new_dropout_2')(x2)
	# current configuration: dense1 + conv1 + conv2
	if dim2>0:
		x2 = Conv1D(dim2, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(x2)

	# logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(x2)
	logits_T = TimeDistributed(Dense(1,name='dense_1'))(x2)
	# logits_T = TimeDistributed(BatchNormalization(name='batchnorm_1'))(logits_T)
	# logits_T = TimeDistributed(Activation("sigmoid",name='activation_1'),name='logits_T')(logits_T)

	tau = 0.5
	k = 5
	if 'tau' in config:
		tau = config['tau']
	if 'n_select' in config:
		k = config['n_select']
	# k = 10
	attention1 = Sample_Concrete(tau, k, n_steps)(logits_T) # output shape: (batch_size, n_steps, 1)
	# attention1 = logits_T

	# encode the input 2
	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	# attention1 = Flatten()(attention1)
	# attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	# attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)
	layer_1 = Multiply()([dense_layer_output1, attention1])
	# dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)

	# method 21: attention1:1, method 22: attention1:0
	if config['attention1']==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, attention_activation=activation_self,name='attention1')(layer_1)

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation!='':
		x1 = Activation(activation,name='activation1')(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if config['attention2']==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, attention_activation=activation_self,name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(x1)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation("sigmoid",name='activation2')(output)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	adam = Adam(lr = lr)
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()

	return model

def get_model2a1_basic5(select_config):

	n_step_local,feature_dim = select_config['input_shape_1']
	feature_dim1, feature_dim2, return_sequences_flag, sample_local = select_config['local_vec_1']
	print(feature_dim1, feature_dim2, return_sequences_flag)

	input_local = Input(shape=(n_step_local,feature_dim))
	# lstm_1 = Bidirectional(LSTM(feature_dim1, name = 'lstm_1'), 
	# 		name = 'bidirectional')(embedded_sequences)

	layer_1 = Conv1D(feature_dim1,1,padding='same',activation=None,strides=1)(input_local)
	layer_1 = BatchNormalization(name='batchnorm_local_1')(layer_1)
	layer_1 = Activation('relu',name='activation_local_1')(layer_1)

	return_sequences_flag = True
	biLSTM_layer_1 = Bidirectional(LSTM(input_shape=(n_step_local, feature_dim1), 
									units=feature_dim2,
									return_sequences = return_sequences_flag,
									kernel_regularizer = keras.regularizers.l2(1e-4),
									dropout=0.1,
									recurrent_dropout = 0.1),name='bilstm_local_1')

	x1 = biLSTM_layer_1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm_local_1')(x1)
	# if activation2!='':
	# 	x1 = Activation(activation2,name='activation_1')(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if return_sequences_flag==True:
		if select_config['attention2_local']==1:
			x1, attention2 = SeqSelfAttention(return_attention=True, attention_activation=activation_self,name='attention_local_1')(x1)

		x1 = GlobalMaxPooling1D(name='global_pooling_local_1')(x1)

	encoder_1 = Model(input_local, x1)

	return encoder_1

# convolution layer + LSTM + pooling
# 2D feature (n_step_local,feature_dim) to 1D
def get_model2a1_basic5_1(input_local,select_config):

	n_step_local,feature_dim = select_config['input_shape_1']
	feature_dim1, feature_dim2, return_sequences_flag, sample_local = select_config['local_vec_1']
	print(feature_dim1, feature_dim2, return_sequences_flag)

	# input_local = Input(shape=(n_step_local,feature_dim))
	# lstm_1 = Bidirectional(LSTM(feature_dim1, name = 'lstm_1'), 
	# 		name = 'bidirectional')(embedded_sequences)

	layer_1 = TimeDistributed(Conv1D(feature_dim1,1,padding='same',activation=None,strides=1),name='conv_local_1')(input_local)
	layer_1 = TimeDistributed(BatchNormalization(),name='batchnorm_local_1')(layer_1)
	layer_1 = TimeDistributed(Activation('relu'),name='activation_local_1')(layer_1)

	biLSTM_layer_1 = Bidirectional(LSTM(input_shape=(n_step_local, feature_dim1), 
									units=feature_dim2,
									return_sequences = return_sequences_flag,
									kernel_regularizer = keras.regularizers.l2(1e-4),
									dropout=0.1,
									recurrent_dropout = 0.1),name='bilstm_local_1_1')

	x1 = TimeDistributed(biLSTM_layer_1,name='bilstm_local_1')(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = TimeDistributed(LayerNormalization(),name='layernorm_local_1')(x1)

	if return_sequences_flag==True:
		if select_config['attention2_local']==1:
			x1, attention2 = TimeDistributed(SeqSelfAttention(return_attention=True, attention_activation=activation_self),name='attention_local_1')(x1)

		x1 = TimeDistributed(GlobalMaxPooling1D(),name='global_pooling_local_1')(x1)

	return x1

# convolution layer + selector + LSTM + pooling
# 2D feature (n_step_local,feature_dim) to 1D
def get_model2a1_basic5_2(input_local,select_config):

	n_step_local,feature_dim = select_config['input_shape_1']
	feature_dim_vec1, feature_dim1, feature_dim2, feature_dim3, return_sequences_flag, sample_local, pooling_local = select_config['local_vec_1']
	regularizer2, regularizer2_2 = select_config['regularizer2'], select_config['regularizer2_2']
	activation_self = select_config['activation_self']
	print(feature_dim1, feature_dim2, return_sequences_flag)

	cnt1 = 0
	if len(feature_dim_vec1)>0:
		size1 = n_step_local
		for feature_dim_1 in feature_dim_vec1:
			cnt1 += 1
			t_feature_dim1, kernel_size, stride1, dilation_rate1, pool_length, stride2 = feature_dim_1
			if t_feature_dim1>0:
				layer_1 = TimeDistributed(Conv1D(t_feature_dim1,kernel_size,kernel_regularizer=regularizers.l2(regularizer2),
													padding='valid',activation=None,strides=stride1),name='conv_local_1_%d'%(cnt1))(input_local)
				layer_1 = TimeDistributed(BatchNormalization(),name='batchnorm_local_1_%d'%(cnt1))(layer_1)
				layer_1 = TimeDistributed(Activation('relu'),name='activation_local_1_%d'%(cnt1))(layer_1)

			if pool_length>1:
				layer_1 = TimeDistributed(MaxPooling1D(pool_size=pool_length,strides=stride2),name='pooling_local_1_%d'%(cnt1))(layer_1)
			
			t1 = int((size1-kernel_size)/stride1)+1
			size1 = int((t1-pool_length)/stride2)+1
		
		n_step_local = size1
		feature_dim = t_feature_dim1
	else:
		layer_1 = input_local

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
		
	biLSTM_layer_1 = Bidirectional(LSTM(
									units=feature_dim1,
									return_sequences = return_sequences_flag,
									kernel_regularizer = keras.regularizers.l2(regularizer2_2),
									dropout=0.1,
									recurrent_dropout = 0.1),name='bilstm_local_1_1')

	x1 = TimeDistributed(biLSTM_layer_1,name='bilstm_local_1')(layer_1)

	# x1 = BatchNormalization()(x1)
	# x1 = TimeDistributed(LayerNormalization(),name='layernorm_local_1')(x1)
	if select_config['concatenate_1']==1:
		# x1 = TimeDistributed(Concatenate(axis=-1),name='concatenate_local_1')([x1,layer_1])
		x1 = Concatenate(axis=-1,name='concatenate_local_1')([x1,layer_1])

	if return_sequences_flag==True:
		if feature_dim2>0:
			cnt1 += 1
			x1 = TimeDistributed(Dense(feature_dim2,
										kernel_regularizer=regularizers.l2(regularizer2)),
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
								kernel_regularizer=regularizers.l2(regularizer2)),
								name = 'conv1_pre_%d'%(cnt1))(x1)
		x1 = TimeDistributed(BatchNormalization(),name='bnorm1_pre_%d'%(cnt1))(x1)
		x1 = TimeDistributed(Activation('relu'),name='activation1_pre_%d'%(cnt1))(x1)

	return x1

# convolution layer + selector + LSTM + pooling
# 2D feature (n_step_local,feature_dim) to 1D
def get_model2a1_basic5_3(input_local,select_config):

	n_step_local,feature_dim = select_config['input_shape_1']
	feature_dim_vec1, feature_dim1, feature_dim2, feature_dim3, return_sequences_flag, sample_local, pooling_local = select_config['local_vec_1']
	regularizer2, regularizer2_2 = select_config['regularizer2'], select_config['regularizer2_2']
	activation_self = select_config['activation_self']
	print(feature_dim1, feature_dim2, return_sequences_flag)

	# input_local = Input(shape=(n_step_local,feature_dim))
	# lstm_1 = Bidirectional(LSTM(feature_dim1, name = 'lstm_1'), 
	# 		name = 'bidirectional')(embedded_sequences)

	cnt1 = 0
	if len(feature_dim_vec1)>0:
		size1 = n_step_local
		for feature_dim_1 in feature_dim_vec1:
			cnt1 += 1
			t_feature_dim1, kernel_size, stride1, dilation_rate1, pool_length, stride2 = feature_dim_1
			layer_1 = TimeDistributed(Conv1D(t_feature_dim1,kernel_size,kernel_regularizer=regularizers.l2(regularizer2),
												padding='valid',activation=None,strides=stride1),name='conv_local_1_%d'%(cnt1))(input_local)
			layer_1 = TimeDistributed(BatchNormalization(),name='batchnorm_local_1_%d'%(cnt1))(layer_1)
			layer_1 = TimeDistributed(Activation('relu'),name='activation_local_1_%d'%(cnt1))(layer_1)

			if pool_length>1:
				layer_1 = TimeDistributed(MaxPooling1D(pool_size=pool_length,strides=stride2),name='pooling_local_1_%d'%(cnt1))(layer_1)
			
			t1 = int((size1-kernel_size)/stride1)+1
			size1 = int((t1-pool_length)/stride2)+1
		
		n_step_local = size1
		feature_dim = t_feature_dim1
	else:
		layer_1 = input_local

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
		
	biLSTM_layer_1 = Bidirectional(LSTM(input_shape=(n_step_local, feature_dim), 
									units=feature_dim1,
									return_sequences = return_sequences_flag,
									kernel_regularizer = keras.regularizers.l2(regularizer2_2),
									dropout=0.1,
									recurrent_dropout = 0.1),name='bilstm_local_1_1')

	x1 = TimeDistributed(biLSTM_layer_1,name='bilstm_local_1')(layer_1)

	# x1 = BatchNormalization()(x1)
	# x1 = TimeDistributed(LayerNormalization(),name='layernorm_local_1')(x1)
	if select_config['concatenate_1']==1:
		# x1 = TimeDistributed(Concatenate(axis=-1),name='concatenate_local_1')([x1,layer_1])
		x1 = Concatenate(axis=-1,name='concatenate_local_1')([x1,layer_1])

	if return_sequences_flag==True:
		if feature_dim2>0:
			cnt1 += 1
			x1 = TimeDistributed(Dense(feature_dim2,
										kernel_regularizer=regularizers.l2(regularizer2)),
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
								kernel_regularizer=regularizers.l2(regularizer2)),
								name = 'conv1_pre_%d'%(cnt1))(x1)
		x1 = TimeDistributed(BatchNormalization(),name='bnorm1_pre_%d'%(cnt1))(x1)
		x1 = TimeDistributed(Activation('relu'),name='activation1_pre_%d'%(cnt1))(x1)

	return x1

# method 32, method 51 and 52: network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling and multiple convolution layers
def get_model2a1_attention_1_2_2_sample2(config):

	feature_dim, output_dim = config['feature_dim'], config['output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	loss_function = config['loss_function']

	n_step_local = config['n_step_local']
	input_shape_1 = [n_step_local,feature_dim]
	return_sequences_flag1 = True
	config.update({'input_shape_1':input_shape_1})
	# encoder_1 = get_model2a1_basic5(config)
	# encoder_1.summary()

	input_region = Input(shape=(n_steps,n_step_local,feature_dim))

	units_1 = config['units1']
	config['units1'] = 0
	layer_2 = get_model2a1_basic5_2(input_region,config)
	config['units1'] = units_1
	print(layer_2.shape)

	config['sample_attention'] = 1
	if config['sample_attention']>=1:
		number, typeid = 3, 2
		logits_T = construct_gumbel_selector1(layer_2,config,number,typeid) # shape: (n_steps,1)

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
		dense_layer_output1 = construct_basic1(layer_2,config)
		dense_layer_output2 = Multiply()([dense_layer_output1, attention1])
	else:
		dense_layer_output2 = layer_2

	output = get_model2a1_basic1(dense_layer_output2,config)

	# output = Activation("softmax")(output)
	model = Model(input = input_region, output = output)

	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	optimizer = Adam(learning_rate = lr, clipnorm=CLIPNORM1)
	# optimizer = find_optimizer(config)

	model.compile(optimizer=optimizer, loss = loss_function)

	model.summary()
	
	return model

# method 32, method 51 and 52: network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling and multiple convolution layers
def get_model2a1_attention_1_2_2_sample2_basic(input_region,config):

	feature_dim, output_dim = config['feature_dim'], config['output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	loss_function = config['loss_function']

	n_step_local = config['n_step_local']
	input_shape_1 = [n_step_local,feature_dim]
	return_sequences_flag1 = True
	config.update({'input_shape_1':input_shape_1})
	
	units_1 = config['units1']
	config['units1'] = 0
	layer_2 = get_model2a1_basic5_2(input_region,config)
	config['units1'] = units_1
	print(layer_2.shape)

	config['sample_attention'] = 1
	if config['sample_attention']>=1:
		number, typeid = 3, 2
		logits_T = construct_gumbel_selector1(layer_2,config,number,typeid) # shape: (n_steps,1)

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
		dense_layer_output1 = construct_basic1(layer_2,config)
		dense_layer_output2 = Multiply()([dense_layer_output1, attention1])
	else:
		dense_layer_output2 = layer_2

	output = get_model2a1_basic1(dense_layer_output2,config)
	
	return output

# method 32, method 51 and 52: network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling and multiple convolution layers
def get_model2a1_attention_1_2_2_sample2_1(config):

	feature_dim, output_dim = config['feature_dim'], config['output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	loss_function = config['loss_function']

	n_step_local = config['n_step_local']
	input_shape_1 = [n_step_local,feature_dim]
	return_sequences_flag1 = True
	config.update({'input_shape_1':input_shape_1})
	# encoder_1 = get_model2a1_basic5(config)
	# encoder_1.summary()

	input_region = Input(shape=(n_steps,n_step_local,feature_dim))
	# layer_2 = TimeDistributed(encoder_1,name='encoder_1')(input_region) # shape: (n_steps,feature_dim2*2)
	# print(layer_2.shape)

	# feature_dim1, feature_dim2, return_sequences_flag = config['local_vec_1']
	# if return_sequences_flag==True:
	# 	if config['attention2_local']==1:
	# 		layer_2, attention2 = TimeDistributed(SeqSelfAttention(return_attention=True, attention_activation=activation_self),name='attention_local_1')(layer_2)

	# 	layer_2 = TimeDistributed(GlobalMaxPooling1D(),name='global_pooling_local_1')(layer_2)

	# layer_2 = get_model2a1_basic5_1(input_region,config)
	units_1 = config['units1']
	config['units1'] = 0
	layer_2 = get_model2a1_basic5_2(input_region,config)
	config['units1'] = units_1
	print(layer_2.shape)

	config['sample_attention'] = 1
	if config['sample_attention']>=1:
		number, typeid = 3, 2
		logits_T = construct_gumbel_selector1(layer_2,config,number,typeid) # shape: (n_steps,1)

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
		dense_layer_output1 = construct_basic1(layer_2,config)
		dense_layer_output2 = Multiply()([dense_layer_output1, attention1])
	else:
		dense_layer_output2 = layer_2

	output = get_model2a1_basic1(dense_layer_output2,config)

	# output = Activation("softmax")(output)
	model = Model(input = input_region, output = output)

	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	optimizer = Adam(learning_rate = lr, clipnorm=CLIPNORM1)
	# optimizer = find_optimizer(config)

	model.compile(optimizer=optimizer, loss = loss_function)

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

# convolution + lstm + pooling
def get_model2a1_basic3_pre(input1,config):

	feature_dim1, n_steps = config['feature_dim'], config['context_size']
	vec1 = config['layer_1']
	lr = config['lr']
	loss_function = config['loss_function']
	activation_vec = ['sigmoid','relu','linear','tanh','softsign']
	conv_size_vec1, attention1, feature_dim2, dropout_rate2, attention2, pool_length1, stride1, activation1, activation2, activation_self, activation_basic = vec1
	dim1, conv_size1, dropout_rate1, valid1 = conv_size_vec1
	# activation, activation2 = activation_vec[activation_type], activation_vec[activation2_type]
	# activation_self = activation_vec[activation_self_type]
	# activation_basic = activation_basic[activation_basic_type]
	if dim1>0:
		x1 = Conv1D(dim1, conv_size1, padding=valid1, activation=activation1, strides=1, name = 'local_conv1')(input1)
		feature_dim1 = dim1
		if dropout_rate1>0:
			x1 = Dropout(dropout_rate1)(x1)
	else:
		x1 = input1

	# method 21: attention1:1, method 22: attention1:0
	if attention1==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, 
												attention_activation=activation_self,
												name='attention1')(x1)
	else:
		layer_1 = x1

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim1), 
									units=feature_dim2,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	# x1 = LayerNormalization(name='layernorm1')(x1)
	# if activation2!='':
	# 	x1 = Activation(activation2,name='activation1')(x1)

	if dropout_rate2>0:
		x1 = Dropout(dropout_rate2)(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if attention2==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, 
											attention_activation=activation_self,
											name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])

	x1 = MaxPooling1D(pool_size = pool_length1, strides = stride1, name='local_pooling1')(x1)
	x1 = Flatten()(x1)

	output = Dense(1,name='dense2')(x1)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation(activation_basic,name='activation2')(output)

	return output

def get_model2a1_basic3(config):

	feature_dim1, n_steps = config['feature_dim'], config['context_size']
	# activation = config['activation']
	# activation2 = config['activation2']
	# activation_self = config['activation_self']
	# if 'activation_basic' in config:
	# 	activation_basic = config['activation_basic']
	# else:
	# 	activation_basic = 'sigmoid'

	input1 = Input(shape = (n_steps,feature_dim1))

	vec1 = config['layer_1']
	lr = config['lr']
	loss_function = config['loss_function']
	activation_vec = ['sigmoid','relu','linear','tanh','softsign']
	conv_size_vec1, attention1, feature_dim2, dropout_rate2, attention2, pool_length1, stride1, activation1, activation2, activation_self, activation_basic = vec1
	dim1, conv_size1, dropout_rate1, valid1 = conv_size_vec1
	# activation, activation2 = activation_vec[activation_type], activation_vec[activation2_type]
	# activation_self = activation_vec[activation_self_type]
	# activation_basic = activation_basic[activation_basic_type]
	if dim1>0:
		x1 = Conv1D(dim1, conv_size1, padding=valid1, activation=activation1, strides=1, name = 'local_conv1')(input1)
		feature_dim1 = dim1
		if dropout_rate1>0:
			x1 = Dropout(dropout_rate1)(x1)
	else:
		x1 = input1

	# method 21: attention1:1, method 22: attention1:0
	if attention1==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, 
												attention_activation=activation_self,
												name='attention1')(x1)
	else:
		layer_1 = x1

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim1), 
									units=feature_dim2,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	# if activation2!='':
	# 	x1 = Activation(activation2,name='activation1')(x1)

	if dropout_rate2>0:
		x1 = Dropout(dropout_rate2)(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if attention2==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, 
											attention_activation=activation_self,
											name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])

	x1 = MaxPooling1D(pool_size = pool_length1, strides = stride1, name='local_pooling1')(x1)
	x1 = Flatten()(x1)

	output = Dense(1,name='dense2')(x1)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation(activation_basic,name='activation2')(output)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	# adam = Adam(lr = lr)

	init_lr = 0.005
	if 'init_lr' in config:
		init_lr = config['init_lr']
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		initial_learning_rate=init_lr,
		decay_steps=1000,
		decay_rate=0.96,
		staircase=True)

	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	optimizer = Adam(learning_rate = lr, clipnorm=CLIPNORM1)
	# optimizer = find_optimizer(config)

	model.compile(optimizer=optimizer, loss = loss_function)

	model.summary()

	return model

def get_model2a1_basic3_sequential(config):

	feature_dim1, n_steps = config['feature_dim'], config['context_size']
	# activation = config['activation']
	# activation2 = config['activation2']
	# activation_self = config['activation_self']
	# if 'activation_basic' in config:
	# 	activation_basic = config['activation_basic']
	# else:
	# 	activation_basic = 'sigmoid'

	input1 = Input(shape = (n_steps,feature_dim1))

	# vec1 = config['layer_1']
	lr = config['lr']
	loss_function = config['loss_function']
	# activation_vec = ['sigmoid','relu','linear','tanh','softsign']
	# conv_size_vec1, attention1, feature_dim2, dropout_rate2, attention2, pool_length1, stride1, activation1, activation2, activation_self, activation_basic = vec1
	# dim1, conv_size1, dropout_rate1, valid1 = conv_size_vec1

	conv_list1 = config['layer1']
	recurrent_list1 = config['layer2']
	dense_list1 = config['layer3']
	# activation, activation2 = activation_vec[activation_type], activation_vec[activation2_type]
	# activation_self = activation_vec[activation_self_type]
	# activation_basic = activation_basic[activation_basic_type]
	x1 = input1
	cnt1 = 1
	for conv_size_vec1 in conv_list1:
		dim1, conv_size1, valid1, activation1, batch_norm, pool_length1, stride1, dropout_rate1, resNet = conv_size_vec1
		
		x2 = Conv1D(dim1, conv_size1, padding=valid1, activation='linear', strides=1, name = 'local_conv%d'%(cnt1))(x1)
		if resNet>0:
			x1 = Add()([x1,x2])
		else:
			x1 = x2
		if batch_norm==1:
			x1 = BatchNormalization(name='batchnorm%d'%(cnt1))(x1)
		x1 = Activation(activation1,name='activation%d'%(cnt1))(x1)

		if pool_length1>1:
			x1 = MaxPooling1D(pool_size = pool_length1, strides = stride1, name='local_pooling%d'%(cnt1))(x1)
		if dropout_rate1>0:
			x1 = Dropout(dropout_rate1)(x1)
		feature_dim1 = dim1
		cnt1 += 1

	for recurrent_vec1 in recurrent_list1:
		feature_dim2, attention1, attention2, activation_self, activation2, dropout_rate2, pool_length1, stride1, dropout_rate3, dim1, activation1, resNet, resNet1 = recurrent_vec1
		# method 21: attention1:1, method 22: attention1:0
		if attention1==1:
			x1, attention1 = SeqSelfAttention(return_attention=True, 
													attention_activation=activation_self,
													name='attention1_%d'%(cnt1))(x1)

		biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim1), 
										units=feature_dim2,
										return_sequences = True,
										recurrent_dropout = 0.1),name='bilstm%d'%(cnt1))
		x2 = biLSTM_layer1(x1)
		# x1 = BatchNormalization()(x1)
		if resNet>0:
			x1 = Add()([x1,x2])
		else:
			x1 = x2

		x1 = LayerNormalization(name='layernorm%d'%(cnt1))(x1)

		# if activation2!='':
		# 	x1 = Activation(activation2,name='activation%d'%(cnt1))(x1)
		if dropout_rate2>0:
			x1 = Dropout(dropout_rate2)(x1)

		# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
		if attention2==1:
			x1, attention2 = SeqSelfAttention(return_attention=True, 
											attention_activation=activation_self,
											name='attention2_%d'%(cnt1))(x1)
		feature_dim1 = 2*feature_dim2

		# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
		# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
		if pool_length1>1:
			x1 = MaxPooling1D(pool_size = pool_length1, strides = stride1, name='local_pooling1')(x1)
		if dropout_rate3>0:
			x1 = Dropout(dropout_rate3)(x1)

		if dim1>0:
			x2 = Dense(dim1,name='dense%d'%(cnt1))(x1)
			if resNet1>0:
				x1 = Add()([x1,x2])
			else:
				x1 = x2
			x1 = BatchNormalization(name='batchnorm%d'%(cnt1))(x1)
			x1 = Activation(activation1,name='activation%d'%(cnt1))(x1)

		cnt1 += 1
		
	x1 = Flatten()(x1)

	for dense_vec1 in dense_list1:
		dim1, activation1, pool_length1, stride1, dropout_rate1, resNet = dense_vec1

		x2 = Dense(dim1,name='dense%d'%(cnt1))(x1)

		if resNet>0:
			x1 = Add()([x1,x2])
		else:
			x1 = x2

		x1 = BatchNormalization(name='batchnorm%d'%(cnt1))(x1)
		x1 = Activation(activation1,name='activation%d'%(cnt1))(x1)

		if pool_length1>1:
			x1 = MaxPooling1D(pool_size = pool_length1, strides = stride1, name='local_pooling%d'%(cnt1))(x1)
		if dropout_rate1>0:
			x1 = Dropout(dropout_rate1)(x1)
		cnt1 += 1

	output = x1
	# output = Dense(1,name='dense%d'%(cnt1))(x1)
	# output = BatchNormalization(name='batchnorm%d'%(cnt1))(output)
	# output = Activation(activation_basic,name='activation%d'%(cnt1))(output)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	# adam = Adam(lr = lr)
	adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	
	model.compile(adam,loss = loss_function)

	model.summary()

	return model

def get_model2a1_basic3_sequential_config(config):

	type_id3 = config['conv_typeid']
	conv_type = ['same','valid']
	valid1 = conv_type[type_id3]
	list1 = []
	batch_norm, pool_length1, stride1, dropout_rate1 = 1, 0, 0, 0
	resNet = 0
	pool_type = [[5,5],[2,2]]
	pool_length1, stride1 = pool_type[type_id3]
	# attention1, attention2 = 0, 1
	dim1, conv_size1 = config['dim1'], config['conv_size1']
	activation1 = 'relu'
	conv_size_vec1 = [dim1, conv_size1, valid1, activation1, batch_norm, pool_length1, stride1, dropout_rate1, resNet]
	list1.append(conv_size_vec1)

	feature_dim2 = config['feature_dim2']
	# attention1, attention2 = 0, 0
	# activation_self, activation2 = 'tanh', 'softsign'
	attention1, attention2 = config['attention1'], config['attention2']
	activation_self, activation2 = config['activation_self'], config['activation2']
	pool_length1, stride1 = pool_type[type_id3]
	# pool_length1, stride1 = 0, 0
	dropout_rate2, dropout_rate3 = 0.2,0
	dim1, activation1 = 0, 'relu'
	resNet, resNet1 = 0, 0
	list2 = []
	recurrent_vec1 = [feature_dim2, attention1, attention2, activation_self, activation2, dropout_rate2, pool_length1, stride1, dropout_rate3, dim1, activation1, resNet, resNet1]
	feature_dim1, n_steps = config['feature_dim'], config['context_size']
	list2.append(recurrent_vec1)

	list3 = []
	# dim1, activation1, pool_length1, stride1, dropout_rate1, resNet = 16,'relu',0,0,0,0
	# dense_vec1 = [dim1, activation1, pool_length1, stride1, dropout_rate1, resNet]
	# list3.append(dense_vec1)
	dim1, activation1, pool_length1, stride1, dropout_rate1, resNet = 1,'sigmoid',0,0,0,0
	dense_vec1 = [dim1, activation1, pool_length1, stride1, dropout_rate1, resNet]
	list3.append(dense_vec1)

	config.update({'layer1':list1,'layer2':list2,'layer3':list3})

	model = get_model2a1_basic3_sequential(config)

	return model

def train_init_config1(config):

	conv_typeid = config['conv_typeid']
	activation1 = 'relu'
	resNet = 0
	type_id3 = conv_typeid
	feature_dim2 = 16
	attention1, attention2 = 0, 0
	activation_self, activation2 = 'tanh', 'softsign'
	activation_basic = 'sigmoid'

	loss_function = 'binary_crossentropy'
	config['loss_function'] = loss_function

	conv_type = ['same','valid']
	valid1 = conv_type[type_id3]
	
	list1 = []
	dim1, conv_size1 = 16,3
	batch_norm, pool_length1, stride1, dropout_rate1 = 1, 0, 0, 0
	resNet = 0
	pool_type = [[5,5],[2,2]]
	# pool_length1, stride1 = pool_type[type_id3]

	conv_size_vec1 = [dim1, conv_size1, valid1, activation1, batch_norm, pool_length1, stride1, dropout_rate1, resNet]
	list1.append(conv_size_vec1)

	# # pool_length1, stride1 = pool_type[type_id3]
	# dim1, conv_size1 = 16,3
	# conv_size_vec1 = [dim1, conv_size1, valid1, activation1, batch_norm, pool_length1, stride1, dropout_rate1, resNet]
	# list1.append(conv_size_vec1)

	feature_dim2 = config['feature_dim2']
	activation_self, activation2 = 'tanh', 'softsign'
	attention1, attention2 = config['attention1'], config['attention2']
	activation_self, activation2 = config['activation_self'], config['activation2']
	pool_length1, stride1 = pool_type[type_id3]
	dropout_rate2, dropout_rate3 = 0.2,0
	dim1, activation1 = 0, 'relu'
	resNet, resNet1 = 0, 0
	list2 = []
	recurrent_vec1 = [feature_dim2, attention1, attention2, activation_self, activation2, dropout_rate2, pool_length1, stride1, dropout_rate3, dim1, activation1, resNet, resNet1]
	list2.append(recurrent_vec1)

	list3 = []
	# dim1, activation1, pool_length1, stride1, dropout_rate1, resNet = 16,'relu',0,0,0,0
	# dense_vec1 = [dim1, activation1, pool_length1, stride1, dropout_rate1, resNet]
	# list3.append(dense_vec1)
	dim1, activation1, pool_length1, stride1, dropout_rate1, resNet = 1,activation_basic,0,0,0,0
	dense_vec1 = [dim1, activation1, pool_length1, stride1, dropout_rate1, resNet]
	list3.append(dense_vec1)

	config.update({'layer1':list1,'layer2':list2,'layer3':list3})

	return config

# method 32, method 51 and 52: network1 for estimating weights, self-attention and network2 for predicting signals
# with gumbel sampling and multiple convolution layers
# initiation zone predition
def get_model2a1_attention_1_2_2_sample3(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']
	activation_self = config['activation_self']
	activation3 = config['activation3']
	typeid_sample = config['typeid_sample']
	loss_function = config['loss_function']

	input1 = Input(shape = (n_steps,feature_dim))

	number = 1
	# if 'typeid2' in config:
	# 	typeid = config['typeid2']
	# else:
	# 	typeid = 2
	typeid = 2
	x1 = input1
	logits_T = construct_gumbel_selector1(x1,config,number,typeid)

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
			else:
				pass
			attention1 = Sample_Concrete1(tau,k,n_steps,typeid_sample)(logits_T) # output shape: (batch_size, n_steps, 1)
	else:
		attention1 = logits_T
	# attention1 = logits_T

	# attention1 = Flatten()(attention1)
	# attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	# attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)

	# encode the input 2
	dense_layer_output1 = construct_basic1(x1,config)
	layer_1 = Multiply()([dense_layer_output1, attention1])

	output = get_model2a1_basic3_pre(layer_1,config)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)

	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	optimizer = Adam(learning_rate = lr, clipnorm=CLIPNORM1)
	# optimizer = find_optimizer(config)

	model.compile(optimizer=optimizer, loss = loss_function)

	model.summary()

	return model

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

# method 32, method 51 and 52: network1 for estimating weights, self-attention and network2 for predicting signals
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

# method 32, method 51 and 52: network1 for estimating weights, self-attention and network2 for predicting signals
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

# method 70: dilated convolutions
def get_model2a1_attention_1_2_2_sample6(config):

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
	layer_2 = get_model2a1_basic5_convolution1(input_region,config)
	print(layer_2.shape)

	# config['sample_attention'] = 1
	# if config['sample_attention']>=1:
	# 	number, typeid = 3, 2
	# 	units1 = config['units1']
	# 	config['units1'] = 0
	# 	logits_T = construct_gumbel_selector1(layer_2,config,number,typeid) # shape: (n_steps,1)
	# 	config['units1'] = units1

	# 	# k = 10
	# 	if config['sample_attention']==1:
	# 		tau, k, typeid_sample = config['tau'], config['n_select'], config['typeid_sample']
	# 		print('sample_attention',tau,k,typeid,activation3,typeid_sample)
			
	# 		if activation3=='linear':
	# 			typeid_sample = 1
	# 		elif activation3=='tanh':
	# 			typeid_sample = 5
	# 		else:
	# 			pass
	# 		attention1 = Sample_Concrete1(tau,k,n_steps,typeid_sample)(logits_T) # output shape: (batch_size, n_steps, 1)
	# 	else:
	# 		attention1 = logits_T

	# 	# encode the input 2
	# 	if config['select2']==1:
	# 		dense_layer_output1 = construct_basic1(layer_2,config)
	# 	else:
	# 		dense_layer_output1 = layer_2

	# 	dense_layer_output2 = Multiply()([dense_layer_output1, attention1])
	# else:
	# 	dense_layer_output2 = layer_2

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
	model = Model(input = input_region, output = output)
	# adam = Adam(lr = lr, clipnorm=CLIPNORM1)
	optimizer = Adam(learning_rate = lr, clipnorm=CLIPNORM1)
	# optimizer = find_optimizer(config)
	model.compile(optimizer=optimizer, loss = loss_function)

	model.summary()
	
	return model

# method 71: dilated convolutions, sequence features
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

	# if config['select2']==1:
	# 	units1 = config['units1']
	# 	config['units1'] = 0
	# 	typeid = 0
	# 	number = 2
	# 	dense_layer_output1 = construct_gumbel_selector1(dense_layer_output1,config,number,typeid)
	# 	config['units1'] = units1

	# layer_1 = Multiply()([dense_layer_output1, attention1])
	layer_1 = dense_layer_output1

	# layer_2 = get_model2a1_basic5_1(input_region,config)
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

def get_model2a1_word():
	MAX_NUM_WORDS = 20000
	EMBEDDING_DIM = 100
	MAX_SENT_LENGTH = 100
	MAX_SENTS = 15
	embedding_layer = Embedding(MAX_NUM_WORDS + 1,
										EMBEDDING_DIM, 
										input_length=MAX_SENT_LENGTH,
										name = 'embedding',
										trainable=True)

	sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sentence_input)
	l_lstm = Bidirectional(LSTM(100, name = 'lstm'), 
			name = 'bidirectional')(embedded_sequences)
	sentEncoder = Model(sentence_input, l_lstm)

	review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
	review_encoder = TimeDistributed(sentEncoder)(review_input)
	l_lstm_sent = Bidirectional(LSTM(100, name = 'lstm2'), 
			name = 'bidirectional2')(review_encoder)
	preds = Dense(2, activation='softmax', name = 'dense')(l_lstm_sent)
	model = Model(review_input, preds)

	model.compile(loss='categorical_crossentropy',
					  optimizer='rmsprop',
					  metrics=['acc'])
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
	
# network1 for estimating weights, self-attention and network2 for predicting signals
def get_model2a1_attention_1_2_select_ori(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']

	input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		# encode the input, shape: (batch_size, n_steps,units_1)
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)	
	else:
		dense_layer_1 = input1

	# calculate attention
	# dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_1)
	# dense_layer_2 = TimeDistributed(BatchNormalization(name='batchnorm_1'))(dense_layer_2)
	# attention1 = TimeDistributed(Activation("sigmoid",name='activation_1'))(dense_layer_2)
	# attention1 = Flatten()(attention1)

	# feature_dim_vec = [50,25,50,25,0,1]
	feature_dim_vec = [50,25,50,25,0,1]
	# shape: (batch_size,n_steps,1)
	logits_T = construct_gumbel_selector(dense_layer_1, feature_dim, feature_dim, n_steps, feature_dim_vec)
	tau = 0.5
	# k = config['n_select']
	k = 10
	attention1 = Sample_Concrete(tau, k, n_steps)(logits_T) # output shape: (batch_size, n_steps, 1)

	# encode the input 2
	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	attention1 = Flatten()(attention1)
	attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)
	layer_1 = Multiply()([dense_layer_output1, attention1])
	# dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)

	if config['attention1']==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(layer_1)

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation!='':
		x1 = Activation(activation,name='activation1')(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if config['attention2']==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(x1)
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

# network1 for estimating weights, self-attention and network2 for predicting signals
def get_model2a1_attention_1_2_select(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']

	input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		# encode the input, shape: (batch_size, n_steps,units_1)
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)	
	else:
		dense_layer_1 = input1

	# calculate attention
	# dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_1)
	# dense_layer_2 = TimeDistributed(BatchNormalization(name='batchnorm_1'))(dense_layer_2)
	# attention1 = TimeDistributed(Activation("sigmoid",name='activation_1'))(dense_layer_2)
	# attention1 = Flatten()(attention1)

	feature_dim_vec = [50,25,50,25,0,0]
	# shape: (batch_size,n_steps,1)

	# construct_gumbel_selector
	logits_T = construct_gumbel_selector(dense_layer_1, feature_dim, feature_dim, n_steps, feature_dim_vec)
	
	tau = 0.5
	# k = config['n_select']
	k = 10
	attention1 = Sample_Concrete(tau, k, n_steps)(logits_T) # output shape: (batch_size, n_steps, 1)

	# encode the input 2
	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	attention1 = Flatten()(attention1)
	attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)
	layer_1 = Multiply()([dense_layer_output1, attention1])
	# dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)

	if config['attention1']==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(layer_1)

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation!='':
		x1 = Activation(activation,name='activation1')(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if config['attention2']==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(x1)
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

# network1 for estimating weights, self-attention and network2 for predicting signals
def get_model2a1_attention_1_2_select_1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']

	input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		# encode the input, shape: (batch_size, n_steps,units_1)
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)	
	else:
		dense_layer_1 = input1

	# calculate attention
	# dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_1)
	# dense_layer_2 = TimeDistributed(BatchNormalization(name='batchnorm_1'))(dense_layer_2)
	# attention1 = TimeDistributed(Activation("sigmoid",name='activation_1'))(dense_layer_2)
	# attention1 = Flatten()(attention1)

	feature_dim_vec = [50,25,50,25,1]
	# shape: (batch_size,n_steps,1)
	logits_T = construct_gumbel_selector(dense_layer_1, feature_dim, feature_dim, n_steps, feature_dim_vec)
	tau = 0.5
	# k = config['n_select']
	k = 10
	attention1 = Sample_Concrete(tau, k, n_steps)(logits_T) # output shape: (batch_size, n_steps, 1)

	# encode the input 2
	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	# attention1 = Flatten()(attention1)
	# attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	# attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)
	layer_1 = Multiply()([dense_layer_output1, attention1])
	# dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)

	if config['attention1']==1:
		layer_1, attention1 = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(layer_1)

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(n_steps, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')
	x1 = biLSTM_layer1(layer_1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	if activation!='':
		x1 = Activation(activation,name='activation1')(x1)

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	if config['attention2']==1:
		x1, attention2 = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention2')(x1)

	# x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(x1)
	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(x1)
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

# network1 for estimating weights, self-attention and single network for predicting signals
def get_model2a1_attention_1_3(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	n_steps = config['context_size']
	lr = config['lr']
	activation = config['activation']

	input1 = Input(shape = (n_steps,feature_dim))
	units_1 = config['units1']
	if units_1>0:
		dense_layer_1 = TimeDistributed(Dense(units_1,name='dense_0'))(input1)
		dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(dense_layer_1)
	else:
		dense_layer_2 = TimeDistributed(Dense(1,name='dense_1'))(input1)
	
	# attention1 = Flatten()(dense_layer_2)
	# attention1 = Activation('softmax',name='attention1')(attention1)
	attention1 = TimeDistributed(Activation("sigmoid",name='activation_1'))(dense_layer_2)
	attention1 = Flatten()(attention1)

	units_2 = config['units2']
	if units_2>0:
		dim2 = units_2
		dense_layer_output1 = TimeDistributed(Dense(units_2,name='dense_2'))(input1)
	else:
		dim2 = feature_dim
		dense_layer_output1 = input1

	attention1 = RepeatVector(dim2)(attention1) # shape: (batch_size,dim2,context_size)
	attention1 = Permute([2,1])(attention1)		# shape: (batch_size,context_size,dim2)
	layer_1 = Multiply()([dense_layer_output1, attention1])
	# dense_layer_output = Lambda(lambda x: K.sum(x,axis=1))(layer_1)

	x1, attention = SeqSelfAttention(return_attention=True, attention_activation='sigmoid',name='attention1')(layer_1)

	x1 = TimeDistributed(Dense(output_dim,name='dense1'))(x1)
	x1 = TimeDistributed(BatchNormalization(name='batchnorm1'))(x1)
	x1 = TimeDistributed(Activation("relu",name='activation1'))(x1)

	x1 = TimeDistributed(Dense(output_dim,name='dense2'))(x1)
	x1 = TimeDistributed(BatchNormalization(name='batchnorm2'))(x1)
	x1 = TimeDistributed(Activation("relu",name='activation2'))(x1)

	output = TimeDistributed(Dense(1,name='dense3'))(x1)
	output = TimeDistributed(BatchNormalization(name='batchnorm3'))(output)
	output = TimeDistributed(Activation("sigmoid",name='activation3'))(output)

	# output = Activation("softmax")(output)
	model = Model(input = input1, output = output)
	adam = Adam(lr = lr)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()

	return model

# def get_model2a1_attention_sequential(input_shape,config):

# 	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
# 	# input1 = Input(shape = (input_shape,feature_dim))

# 	model = keras.models.Sequential()
# 	model.add(Bidirectional(LSTM(units=output_dim,
# 								return_sequences = True,
# 								recurrent_dropout = 0.1),input_shape=(None, feature_dim),name='bilstm1'))
# 	model.add(LayerNormalization())
# 	model.add(SeqSelfAttention(attention_activation='sigmoid',name='attention1'))

# 	if fc1_output_dim>0:
# 		model.add(Dense(units=fc1_output_dim,name='dense1'))
# 		model.add(BatchNormalization(name='batchnorm1'))
# 		model.add(Activation("relu",name='activation1'))
# 		model.add(Dropout(0.5))
# 	else:
# 		pass

# 	model.add(Dense(units=1,name='dense2'))
# 	model.add(BatchNormalization(name='batchnorm2'))
# 	model.add(Activation("sigmoid",name='activation2'))

# 	adam = Adam(lr = learning_rate)
# 	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
# 	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
# 	# model.compile(adam,loss = 'mean_absolute_percentage_error')
# 	model.compile(adam,loss = 'mean_squared_error')
# 	adam = Adam(lr = learning_rate)

# 	model.summary()

# 	return model

def get_model2a2_attention(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (None,feature_dim))
	input2 = Input(shape = (None,1))

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	# x1 = Flatten()(x1)

	x1 = SeqSelfAttention(attention_activation='sigmoid',name='attention1')(x1)
	x1 = Concatenate(axis=-1,name='concate1')([x1,input2])
	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim,name='dense1')(x1)
		dense1 = BatchNormalization(name='batchnorm1')(dense1)
		dense1 = Activation("relu",name='activation1')(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
	else:
		dense_layer_output = x1

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(dense_layer_output)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation("sigmoid",name='activation2')(output)
	# output = Activation("softmax")(output)

	model = Model(input = [input1,input2], output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()
	return model

def get_model2a2_1(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (None,feature_dim))
	input2 = Input(shape = (None,1))

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1),name='bilstm1')

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization(name='layernorm1')(x1)
	# x1 = Flatten()(x1)

	# x1 = SeqSelfAttention(attention_activation='sigmoid')(x1)
	x1 = Concatenate(axis=-1,name='concate1')([x1,input2])
	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim,name='dense1')(x1)
		dense1 = BatchNormalization(name='batchnorm1')(dense1)
		dense1 = Activation("relu",name='activation1')(dense1)
		dense_layer_output = Dropout(0.5,name='dropout1')(dense1)
	else:
		dense_layer_output = x1

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(dense_layer_output)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation("sigmoid",name='activation2')(output)
	# output = Activation("softmax")(output)

	model = Model(input = [input1,input2], output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()
	return model

def get_model2a2_1_predict(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (None,output_dim*2))
	input2 = Input(shape = (None,1))

	# biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
	# 								units=output_dim,
	# 								return_sequences = True,
	# 								recurrent_dropout = 0.1),name='bilstm1')

	# x1 = biLSTM_layer1(input1)
	# # x1 = BatchNormalization()(x1)
	# x1 = LayerNormalization()(x1)
	# # x1 = Flatten()(x1)

	# # x1 = SeqSelfAttention(attention_activation='sigmoid')(x1)
	x1 = Concatenate(axis=-1,name='concate1')([input1,input2])
	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim,name='dense1')(x1)
		dense1 = BatchNormalization(name='batchnorm1')(dense1)
		dense1 = Activation("relu",name='activation1')(dense1)
		dense_layer_output = Dropout(0.5,name='dropout1')(dense1)
	else:
		dense_layer_output = x1

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1,name='dense2')(dense_layer_output)
	output = BatchNormalization(name='batchnorm2')(output)
	output = Activation("sigmoid",name='activation2')(output)
	# output = Activation("softmax")(output)

	model = Model(input = [input1,input2], output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()
	return model

def get_model2a2_2(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (None,feature_dim))
	input2 = Input(shape = (None,1))

	LSTM_layer1 = LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1)

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization()(x1)
	# x1 = Flatten()(x1)

	# x1 = SeqSelfAttention(attention_activation='sigmoid')(x1)
	x1 = Concatenate(axis=-1)([x1,input2])
	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim)(x1)
		dense1 = BatchNormalization()(dense1)
		dense1 = Activation("relu")(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
	else:
		dense_layer_output = x1

	# concat_layer_output = Concatenate(axis=-1)([dense_layer_output,input2])
	output = Dense(1)(dense_layer_output)
	output = BatchNormalization()(output)
	output = Activation("sigmoid")(output)
	# output = Activation("softmax")(output)

	model = Model(input = [input1,input2], output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()
	return model

def get_model2a1_attention1_sequential(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	# input1 = Input(shape = (input_shape,feature_dim))

	model = keras.models.Sequential()
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1),input_shape=(None, feature_dim)))
	model.add(LayerNormalization())
	model.add(SeqSelfAttention(kernel_regularizer=keras.regularizers.l2(1e-4),
					   bias_regularizer=keras.regularizers.l1(1e-4),
					   attention_regularizer_weight=1e-4,
					   name='Attention'))

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

	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')
	adam = Adam(lr = learning_rate)

	model.summary()

	return model

def get_model2a1_attention2_sequential(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	# input1 = Input(shape = (input_shape,feature_dim))

	model = keras.models.Sequential()
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1),input_shape=(None, feature_dim)))
	model.add(LayerNormalization())
	model.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
					   kernel_regularizer=keras.regularizers.l2(1e-4),
					   bias_regularizer=keras.regularizers.l1(1e-4),
					   attention_regularizer_weight=1e-4,
					   name='Attention'))

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

	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')
	adam = Adam(lr = learning_rate)

	model.summary()

	return model

# Two-layer two-layer bidirectional LSTM
def get_model2a1_attention1_2_sequential(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	# input1 = Input(shape = (input_shape,feature_dim))

	model = keras.models.Sequential()
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1),input_shape=(None, feature_dim)))
	model.add(LayerNormalization())
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1)))
	model.add(LayerNormalization())
	model.add(SeqSelfAttention(
					   kernel_regularizer=keras.regularizers.l2(1e-4),
					   bias_regularizer=keras.regularizers.l1(1e-4),
					   attention_regularizer_weight=1e-4,
					   name='Attention'))

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

	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')
	adam = Adam(lr = learning_rate)

	model.summary()

	return model

# Two-layer two-layer bidirectional LSTM
def get_model2a1_attention2_2_sequential(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	# input1 = Input(shape = (input_shape,feature_dim))

	model = keras.models.Sequential()
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1),input_shape=(None, feature_dim)))
	model.add(LayerNormalization())
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1)))
	model.add(LayerNormalization())
	model.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
					   kernel_regularizer=keras.regularizers.l2(1e-4),
					   bias_regularizer=keras.regularizers.l1(1e-4),
					   attention_regularizer_weight=1e-4,
					   name='Attention'))

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

	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')
	adam = Adam(lr = learning_rate)

	model.summary()

	return model

# Flatten
def get_model2a_attention1_sequential(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	# input1 = Input(shape = (input_shape,feature_dim))

	model = keras.models.Sequential()
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1),input_shape=(None, feature_dim)))
	model.add(LayerNormalization())
	model.add(SeqSelfAttention(kernel_regularizer=keras.regularizers.l2(1e-4),
					   bias_regularizer=keras.regularizers.l1(1e-4),
					   attention_regularizer_weight=1e-4,
					   name='Attention'))

	model.add(Flatten())
	
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

	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')
	adam = Adam(lr = learning_rate)

	model.summary()

	return model

# Flatten
def get_model2a_attention2_sequential(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	# input1 = Input(shape = (input_shape,feature_dim))

	model = keras.models.Sequential()
	model.add(Bidirectional(LSTM(units=output_dim,
								return_sequences = True,
								recurrent_dropout = 0.1),input_shape=(None, feature_dim)))
	model.add(LayerNormalization())
	model.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
					   kernel_regularizer=keras.regularizers.l2(1e-4),
					   bias_regularizer=keras.regularizers.l1(1e-4),
					   attention_regularizer_weight=1e-4,
					   name='Attention'))

	model.add(Flatten())
	
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

	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')
	adam = Adam(lr = learning_rate)

	model.summary()

	return model

def get_model2a2(input_shape,config):

	feature_dim, output_dim, fc1_output_dim = config['feature_dim'], config['output_dim'], config['fc1_output_dim']
	input1 = Input(shape = (input_shape,feature_dim))

	biLSTM_layer1 = Bidirectional(LSTM(input_shape=(None, feature_dim), 
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1))
	biLSTM_layer2 = Bidirectional(LSTM(input_shape=(None, output_dim),
									units=output_dim,
									return_sequences = True,
									recurrent_dropout = 0.1))

	x1 = biLSTM_layer1(input1)
	# x1 = BatchNormalization()(x1)
	x1 = LayerNormalization()(x1)

	x2 = biLSTM_layer2(x1)
	x2 = LayerNormalization()(x2)

	# x1 = Flatten()(x1)

	if fc1_output_dim>0:
		dense1 = Dense(fc1_output_dim)(x1)
		dense1 = BatchNormalization()(dense1)
		dense1 = Activation("relu")(dense1)
		dense_layer_output = Dropout(0.5)(dense1)
	else:
		dense_layer_output = x1

	output = Dense(1)(dense_layer_output)
	output = BatchNormalization()(output)
	output = Activation("sigmoid")(output)
	# output = Activation("softmax")(output)

	model = Model(input = input1, output = output)
	adam = Adam(lr = learning_rate)
	# model.compile(adam,loss = 'binary_crossentropy',metrics=['accuracy'])
	# model.compile(adam,loss = 'kullback_leibler_divergence',metrics=['accuracy'])
	# model.compile(adam,loss = 'mean_absolute_percentage_error')
	model.compile(adam,loss = 'mean_squared_error')

	model.summary()
	return model

# select sample
def sample_select1(x_mtx, idx_sel_list, tol=5, L=5):

	num1 = len(idx_sel_list)
	feature_dim = x_mtx.shape[1]
	# L = 5
	size1 = 2*L+1
	feature_dim = x_mtx.shape[1]
	vec1_list = np.zeros((num1,size1))
	# feature_list = np.zeros((num1,size1*feature_dim))
	feature_list = np.zeros((num1,size1,feature_dim))
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
		feature_list[i] = t_feature

		if i%10000==0:
			print(i,t_feature.shape,vec1,vec1_list[i])

	return feature_list, vec1_list

# select sample
def sample_select2(x_mtx, y, idx_sel_list, tol=5, L=5):

	num1 = len(idx_sel_list)
	feature_dim = x_mtx.shape[1]
	# L = 5
	size1 = 2*L+1
	feature_dim = x_mtx.shape[1]
	vec1_list = np.zeros((num1,size1))
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
		feature_list[i] = t_feature
		signal_list[i] = y[vec1]

		if i%10000==0:
			print(i,t_feature.shape,vec1,vec1_list[i])

	signal_list = np.expand_dims(signal_list, axis=2)
	return feature_list, signal_list, vec1_list

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

# select sample
def sample_select2a_3(x_mtx, y, idx_sel_list, tol=5, L=5):

	num1 = len(idx_sel_list)
	feature_dim = x_mtx.shape[1]
	# L = 5
	size1 = 2*L+1
	feature_dim = x_mtx.shape[1]
	vec1_list = np.zeros((num1,size1))
	vec2_list = np.zeros((num1,size1))
	# feature_list = np.zeros((num1,size1*feature_dim))
	feature_list = np.zeros((num1,size1,feature_dim))
	print(y.shape)
	signal_list = np.zeros((num1,size1,y.shape[1]))
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

	# signal_list = np.expand_dims(signal_list, axis=-1)

	return feature_list, signal_list, vec1_list, vec2_list

# select sample
def sample_select2a_pre(x_mtx, y, idx_sel_list, tol=5, L=5):

	num1 = len(idx_sel_list)
	feature_dim = x_mtx.shape[1]
	
	# L = 5
	size1 = 2*L+1
	vec1_list = np.zeros((num1,size1))
	vec2_list = np.zeros((num1,size1))
	vec1_list_pre = np.zeros((num1,size1))
	vec2_list_pre = np.zeros((num1,size1))
	# feature_list = np.zeros((num1,size1*feature_dim))
	feature_list = np.zeros((num1,size1,feature_dim))
	signal_list = np.zeros((num1,size1))
	signal_list_pre = np.zeros((num1,size1))

	for i in range(0,num1):
		temp1 = idx_sel_list[i]
		t_chrom, t_serial = temp1[0], temp1[1]
		id1 = []
		id1_pre = []
		for k in range(-L,L+1):
			id2 = np.min((np.max((i+k,0)),num1-1))
			id1.append(id2)
			id2_pre = np.min((np.max((i+k-1,0)),num1-1))
			id1_pre.append(id2_pre)
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

		# find the previous position
		start1 = vec1[0]
		start_chrom, start_serial = idx_sel_list[start1,0], idx_sel_list[start1,1]
		b1 = np.where((idx_sel_list[:,0]==start_chrom)&(idx_sel_list[:,1]>=start_serial-tol)&(idx_sel_list[:,1]<start_serial))[0]
		if len(b1)==0:
			start_pre = start1
		else:
			start_pre = b1[-1]

		vec1_list[i] = idx_sel_list[vec1,1]
		vec2_list[i] = vec1

		vec1_pre = np.hstack(([start_pre],vec1[0:-1]))
		vec2_list_pre[i] = vec1_pre
		vec1_list_pre[i] = idx_sel_list[vec1_pre,1]

		feature_list[i] = t_feature
		signal_list[i] = y[vec1]
		signal_list_pre[i] = y[vec1_pre]

		if i%10000==0:
			print(i,t_feature.shape,vec1,vec1_list[i])

	signal_list = np.expand_dims(signal_list, axis=-1)
	signal_list_pre = np.expand_dims(signal_list_pre, axis=-1)

	return feature_list, signal_list, vec1_list, vec2_list, signal_list_pre, vec1_list_pre, vec2_list_pre

# select sample
def sample_select2a_pre_1(y, idx_sel_list, tol=5, L=5):

	num1 = len(idx_sel_list)
	
	# L = 5
	size1 = 2*L+1
	vec1_list = np.zeros((num1,size1))
	vec2_list = np.zeros((num1,size1))
	vec1_list_pre = np.zeros((num1,size1))
	vec2_list_pre = np.zeros((num1,size1))
	# feature_list = np.zeros((num1,size1*feature_dim))
	signal_list = np.zeros((num1,size1))
	signal_list_pre = np.zeros((num1,size1))

	for i in range(0,num1):
		temp1 = idx_sel_list[i]
		t_chrom, t_serial = temp1[0], temp1[1]
		id1 = []
		id1_pre = []
		for k in range(-L,L+1):
			id2 = np.min((np.max((i+k,0)),num1-1))
			id1.append(id2)

		# print(id1)
		
		vec1 = []
		start1 = t_serial
		t_id = i
		# serial of the first L loci serial of the first L loci
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

		# find the previous position
		start1 = vec1[0]
		start_chrom, start_serial = idx_sel_list[start1,0], idx_sel_list[start1,1]
		b1 = np.where((idx_sel_list[:,0]==start_chrom)&(idx_sel_list[:,1]>=start_serial-tol)&(idx_sel_list[:,1]<start_serial))[0]
		if len(b1)==0:
			start_pre = start1
		else:
			start_pre = b1[-1]

		vec1_list[i] = idx_sel_list[vec1,1]
		vec2_list[i] = vec1

		vec1_pre = np.hstack(([start_pre],vec1[0:-1]))
		vec2_list_pre[i] = vec1_pre
		vec1_list_pre[i] = idx_sel_list[vec1_pre,1]

		# feature_list[i] = t_feature
		signal_list[i] = y[vec1]
		signal_list_pre[i] = y[vec1_pre]

		if i%10000==0:
			print(i,vec1_list[i],vec1_list_pre[i])

	signal_list = np.expand_dims(signal_list, axis=-1)
	signal_list_pre = np.expand_dims(signal_list_pre, axis=-1)

	return signal_list, vec1_list, vec2_list, signal_list_pre, vec1_list_pre, vec2_list_pre

# select sample
def sample_select2a_pre_2(idx_sel_list, tol=5, L=5):

	num1 = len(idx_sel_list)

	# L = 5
	size1 = 2*L+1
	vec1_list_pre = np.zeros(num1)
	vec2_list_pre = np.zeros(num1)

	for i in range(0,num1):
		temp1 = idx_sel_list[i]
		t_chrom, t_serial = temp1[0], temp1[1]
		id1 = []
		id1_pre = []
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
		# start1 = t_serial
		# t_id = i
		# vec1.append(t_id)
		# for k in range(1,L+1):
		# 	id2 = id1[L+k]
		# 	if (idx_sel_list[id2,0]==t_chrom) and (idx_sel_list[id2,1]<=start1+tol):
		# 		vec1.append(id2)
		# 		t_id = id2
		# 		start1 = idx_sel_list[id2,1]
		# 	else:
		# 		vec1.append(t_id)

		# find the previous position
		start1 = vec1[0]
		start_chrom, start_serial = idx_sel_list[start1,0], idx_sel_list[start1,1]
		b1 = np.where((idx_sel_list[:,0]==start_chrom)&(idx_sel_list[:,1]>=start_serial-tol)&(idx_sel_list[:,1]<start_serial))[0]
		if len(b1)==0:
			start_pre = start1
		else:
			start_pre = b1[-1]

		vec1_list_pre[i] = idx_sel_list[start_pre,1]
		vec2_list_pre[i] = start_pre

		if i%10000==0:
			print(i,vec1_list_pre[i])

	return vec1_list_pre, vec2_list_pre

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

def read_predict1(y, vec, idx, flanking1=3, type_id=0, base1=0.25):

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

	idx_range = np.asarray(range(L-flanking1,L+flanking1+1))
	weight[idx_range] = 1
	mtx2 = np.outer(a2,weight)
	print(num1,context_size,L,idx_range)
	# print(weight)

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

def read_predict_3(y, vec, idx, flanking1=3, type_id=0, base1=0.25):

	num1, context_size = vec.shape[0], vec.shape[1]
	if len(idx)==0:
		idx = range(0,num1)

	a1 = np.asarray(range(0,context_size))
	a2 = np.ones((num1,1))
	mtx1 = np.outer(a2,a1)
	weight = 0.5*np.ones(context_size)
	L = int((context_size-1)*0.5)

	if type_id==1:
		base1 = base1
		for i in range(0,L+1):
			weight[i] = base1+(1-base1)*i/L
		for i in range(L,context_size):
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
	# y1 = np.ravel(y)
	y1 = y.reshape((y.shape[0]*y.shape[1],y.shape[-1]))
	value = np.zeros((num1,y.shape[-1]))
	for i in range(0,num1):
		b1 = np.where(idx_vec==idx[i])[0]
		if len(b1)==0:
			print("error! %d %d"%(i,idx[i]))
		elif len(b1)>context_size:
			print('%d %d'%(i,idx[i]))
		else:
			pass
		t_weight = weight_vec[b1]
		t_weight = t_weight*1.0/np.sum(t_weight)
		value[i] = np.dot(t_weight,y1[b1])

	return value

def dot_layer(inputs):
	x,y = inputs

	return K.sum(x*y,axis = -1,keepdims=True)

def corr(y_true, y_pred):
	return np.min(np.corrcoef(y_true,y_pred))

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

# class roc_callback(keras.callbacks.Callback):
# 	def __init__(self,X,y):
# 		self.x = X
# 		self.y = y

# 	def on_train_begin(self, logs={}):
# 		return

# 	def on_train_end(self, logs={}):
# 		return

# 	def on_epoch_begin(self, epoch, logs={}):
# 		return

# 	def on_epoch_end(self, epoch, logs={}):
# 		y_proba = model.predict(self.x,batch_size = BATCH_SIZE)
# 		y_pred = ((y_proba > 0.5) * 1.0).reshape((-1))
# 		y_test = self.y
# 		print("accuracy", np.sum(y_test == y_pred) / len(y_test))

# 		print("roc", roc_auc_score(y_test, y_proba))
# 		print("aupr", average_precision_score(y_test, y_proba))

# 		print("precision", precision_score(y_test, y_pred))
# 		print("recall", recall_score(y_test, y_pred))
# 		return

# 	def on_batch_begin(self, batch, logs={}):
# 		return

# 	def on_batch_end(self, batch, logs={}):
# 		return

#prep_data()
#gen_Seq(100)
#load_file("../LabelSeq",223,"vec.npy")
#get_target()
#load_file("../TargetSeq",23,"t.npy")

# def signal_normalize_query(query_point, scale_ori, scale):

# 	s1, s2 = scale[0], scale[1]
# 	s_min, s_max = scale_ori[0], scale_ori[1]
# 	scaled_signal = s1+(query_point-s_min)*1.0/(s_max-s_min)*(s2-s1)

# 	return scaled_signal

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

# def score_function1(y_test, y_pred, y_proba):

# 	auc = roc_auc_score(y_test,y_proba)
# 	aupr = average_precision_score(y_test,y_proba)
# 	precision = precision_score(y_test,y_pred)
# 	recall = recall_score(y_test,y_pred)
# 	accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))
# 	F1 = 2*precision*recall/(precision+recall)

# 	# print(auc,aupr,precision,recall)
	
# 	return accuracy, auc, aupr, precision, recall, F1

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

def gc_compare_single(species_name, train_chromvec, test_chromvec, feature_idx):
	
	path2 = '/volume01/yy3/seq_data/dl/replication_timing'
	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
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
	for chrom_id in train_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id1 = np.where(chrom==chrom_id1)[0]
		train_sel_idx.extend(id1)

	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)
	x_train1 = gc_signal[train_sel_idx]
	x_train1 = x_train1[:,feature_idx]
	x_test = gc_signal[test_sel_idx]
	x_test = x_test[:,feature_idx]
	print(x_train1.shape,x_test.shape)

	y_signal_train1 = signal[train_sel_idx]
	y_signal_test = signal[test_sel_idx]

	x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.2, random_state=42)

	vec1 = []
	reg = LinearRegression().fit(x_train, y_train)
	y_predicted_valid = reg.predict(x_valid)
	y_predicted_test = reg.predict(x_test)
	print(reg.coef_,reg.intercept_)

	score1 = mean_squared_error(y_valid, y_predicted_valid)
	score2 = pearsonr(y_valid,y_predicted_valid)
	vec1.append([score1,score2])

	score1 = mean_squared_error(y_signal_test, y_predicted_test)
	score2 = pearsonr(y_signal_test,y_predicted_test)
	vec1.append([score1,score2])
	print(vec1)

	xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
				 gamma=0,                 
				 learning_rate=0.07,
				 max_depth=3,
				 min_child_weight=1.5,
				 n_estimators=5000,                                                                    
				 reg_alpha=0.75,
				 reg_lambda=0.45,
				 objective='reg:squarederror',
				 subsample=0.6,
				 seed=42) 

	# objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	xgb_model.fit(x_train, y_train)
	y_predicted_valid = xgb_model.predict(x_valid)
	y_predicted_test = xgb_model.predict(x_test)

	print("train",train_chromvec)
	score1 = mean_squared_error(y_valid, y_predicted_valid)
	score2 = pearsonr(y_valid,y_predicted_valid)
	vec1.append([score1,score2])
	score1 = mean_squared_error(y_valid, y_predicted_valid)
	score2 = pearsonr(y_valid,y_predicted_valid)
	vec1.append([score1,score2])
	print(score1,score2)

	print("test",test_chromvec)
	score1 = mean_squared_error(y_signal_test, y_predicted_test)
	score2 = pearsonr(y_signal_test,y_predicted_test)
	vec1.append([score1,score2])
	print(score1,score2)
	print(vec1)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return True

def score_2(y, y_predicted):

	score1 = mean_squared_error(y, y_predicted)
	score2 = pearsonr(y, y_predicted)

	return score1, score2

# def score_2a(y, y_predicted):

# 	score1 = mean_squared_error(y, y_predicted)
# 	score2 = pearsonr(y, y_predicted)
# 	score3 = explained_variance_score(y, y_predicted)
# 	score4 = mean_absolute_error(y, y_predicted)
# 	score5 = median_absolute_error(y, y_predicted)
# 	score6 = r2_score(y, y_predicted)
# 	vec1 = [score1, score2, score3, score4, score5, score6]

# 	return vec1

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

	path1 = '/volume01/yy3/seq_data/dl/replication_timing'

	filename3 = '%s/estimate_rt/estimate_rt_%s.txt'%(path1,species_name)
	filename3a = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path1,species_name)
	temp1 = pd.read_csv(filename3,sep='\t')
	temp2 = pd.read_csv(filename3a,sep='\t')
	colname1, colname2 = list(temp1), list(temp2)
	chrom1, start1, stop1, serial1 = temp1[colname1[0]], temp1[colname1[1]], temp1[colname1[2]], temp1[colname1[3]]
	chrom2, start2, stop2, serial2 = temp2[colname2[0]], temp2[colname2[1]], temp2[colname2[2]], temp2[colname2[3]]

	map_idx = mapping_Idx(serial1,serial2)

	return serial1, serial2, map_idx

# compare using kmer features
def kmer_compare_single1(species_vec1, train_chromvec, test_chromvec, feature_idx, type_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

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

def kmer_compare_single2(species_vec1, train_chromvec, test_chromvec, feature_idx, type_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

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

	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)

	vec1 = []
	print("LR")
	reg = LinearRegression().fit(x_train, y_train)
	y_predicted_valid = reg.predict(x_valid)
	y_predicted_test = reg.predict(x_test)
	print(reg.coef_,reg.intercept_)

	# score1, score2 = score_2(y_valid, y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)
	temp1 = score_2a(y_valid, y_predicted_valid)
	vec1.append(temp1)
	print(temp1)
	
	# score1, score2 = score_2(y_signal_test, y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	temp1 = score_2a(y_signal_test, y_predicted_test)
	vec1.append(temp1)
	print(temp1)

	# print('SVR')
	# clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
	# clf.fit(x_train, y_train)
	# y_predicted_valid1 = reg.predict(x_valid)
	# y_predicted_test1 = reg.predict(x_test)

	# temp1 = score_2a(y_valid, y_predicted_valid1)
	# vec1.append(temp1)
	# print(temp1)

	# temp1 = score_2a(y_signal_test, y_predicted_test1)
	# vec1.append(temp1)
	# print(temp1)

	dict1 = dict()
	dict1['vec1'] = vec1
	dict1['y_valid'], dict1['y_test'] = y_valid, y_signal_test
	dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test
	# dict1['y_predicted_valid1'], dict1['y_predicted_test1'] = y_predicted_valid1, y_predicted_test1

	# print("Regressor")
	# xgb_model = xgboost.XGBRegressor(colsample_bytree=0.5,
	# 			 gamma=0,                 
	# 			 learning_rate=0.07,
	# 			 max_depth=10,
	# 			 min_child_weight=1.5,
	# 			 n_estimators=1000,                                                                    
	# 			 reg_alpha=0.75,
	# 			 reg_lambda=0.45,
	# 			 objective='reg:squarederror',
	# 			 n_jobs=50,
	# 			 subsample=0.6,
	# 			 seed=42) 

	# # objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	# xgb_model.fit(x_train, y_train)
	# y_predicted_valid = xgb_model.predict(x_valid)
	# y_predicted_test = xgb_model.predict(x_test)

	# print("train",train_chromvec)
	# score1 = mean_squared_error(y_valid, y_predicted_valid)
	# score2 = pearsonr(y_valid,y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)

	# print("test",test_chromvec)
	# score1 = mean_squared_error(y_signal_test, y_predicted_test)
	# score2 = pearsonr(y_signal_test,y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	print(vec1)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1,dict1

def kmer_compare_single2a(species_vec1, train_chromvec, test_chromvec, feature_idx, type_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

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

	filename1 = '%s/Datavecs/datavecs_GM_%s.1.npy'%(path2,species_name)
	t_signal_ori = np.load(filename1)
	n_dim = int(t_signal_ori.shape[1]*0.5)
	t_signal_ori = t_signal_ori[:,0:n_dim]
	print(t_signal_ori.shape)
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
		print(id1.shape,id1_ori.shape)
		# filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		# file2 = np.load(filename2)
		# t_signal = np.asarray(file2)
		# trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		trans_id1 = mapping_Idx(serial_ori,serial[id1])	# mapped index
		print("trans_id1", trans_id1.shape)
		t_signal = t_signal_ori[trans_id1]
		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		print(id2.shape,id2_ori.shape)
		# filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		# file2 = np.load(filename2)
		# t_signal = np.asarray(file2)
		# trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		trans_id2 = mapping_Idx(serial_ori,serial[id2])	# mapped index
		print("trans_id2", trans_id2.shape)
		t_signal = t_signal_ori[trans_id2]
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
			# filename1 = '%s/training2_kmer_%s.npy'%(path2,species_name)
			filename1 = '%s/Datavecs/datavecs_GM_%s.npy'%(path2,species_name)
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

	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

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

	# print("Regressor")
	# xgb_model = xgboost.XGBRegressor(colsample_bytree=0.5,
	# 			 gamma=0,                 
	# 			 learning_rate=0.07,
	# 			 max_depth=10,
	# 			 min_child_weight=1.5,
	# 			 n_estimators=1000,                                                                    
	# 			 reg_alpha=0.75,
	# 			 reg_lambda=0.45,
	# 			 objective='reg:squarederror',
	# 			 n_jobs=50,
	# 			 subsample=0.6,
	# 			 seed=42) 

	# # objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	# xgb_model.fit(x_train, y_train)
	# y_predicted_valid = xgb_model.predict(x_valid)
	# y_predicted_test = xgb_model.predict(x_test)

	# print("train",train_chromvec)
	# score1 = mean_squared_error(y_valid, y_predicted_valid)
	# score2 = pearsonr(y_valid,y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)

	# print("test",test_chromvec)
	# score1 = mean_squared_error(y_signal_test, y_predicted_test)
	# score2 = pearsonr(y_signal_test,y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	print(vec1)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1

def kmer_compare_single2a1(species_vec1, train_chromvec, test_chromvec, feature_idx, type_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

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

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

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
		t_signal_ori = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal = t_signal_ori[trans_id1]
		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]
		t_signal = np.hstack((t_gc,t_signal))
		print("trans_id1", trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal = t_signal_ori[trans_id2]
		trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		t_gc = gc_signal[trans_id2a]
		t_gc = t_gc[:,feature_idx]
		t_signal = np.hstack((t_gc,t_signal))
		print("trans_id2", trans_id2.shape, t_signal.shape)

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
	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	print(x_train1.shape,y_signal_train1.shape)

	x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.2, random_state=42)

	vec1 = []
	print("LR")
	reg = LinearRegression().fit(x_train, y_train)
	y_predicted_valid = reg.predict(x_valid)
	y_predicted_test = reg.predict(x_test)
	print(reg.coef_,reg.intercept_)

	temp1 = score_2a(y_valid, y_predicted_valid)
	vec1.append(temp1)
	print(temp1)
	
	temp1 = score_2a(y_signal_test, y_predicted_test)
	vec1.append(temp1)
	print(temp1)

	# print("RFLR")
	# RandomForestRegressor
	# regr = RandomForestRegressor(max_depth=2, random_state=0,
	# 						n_estimators=10)
	# regr.fit(x_train, y_train)
	# print(regr.feature_importances_)
	# np.save('regr_featureImp',regr.feature_importances_)
	# y_predicted_valid = reg.predict(x_valid)
	# y_predicted_test = reg.predict(x_test)
	# print(reg.coef_,reg.intercept_)

	# temp1 = score_2a(y_valid, y_predicted_valid)
	# vec1.append(temp1)
	# print(temp1)
	
	# temp1 = score_2a(y_signal_test, y_predicted_test)
	# vec1.append(temp1)
	# print(temp1)

	# print("Regressor")
	# xgb_model = xgboost.XGBRegressor(colsample_bytree=0.5,
	# 			 gamma=0,                 
	# 			 learning_rate=0.07,
	# 			 max_depth=10,
	# 			 min_child_weight=1.5,
	# 			 n_estimators=1000,                                                                    
	# 			 reg_alpha=0.75,
	# 			 reg_lambda=0.45,
	# 			 objective='reg:squarederror',
	# 			 n_jobs=50,
	# 			 subsample=0.6,
	# 			 seed=42) 

	# # objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	# xgb_model.fit(x_train, y_train)
	# y_predicted_valid = xgb_model.predict(x_valid)
	# y_predicted_test = xgb_model.predict(x_test)

	# print("train",train_chromvec)
	# score1 = mean_squared_error(y_valid, y_predicted_valid)
	# score2 = pearsonr(y_valid,y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)

	# print("test",test_chromvec)
	# score1 = mean_squared_error(y_signal_test, y_predicted_test)
	# score2 = pearsonr(y_signal_test,y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	print(vec1)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1

def kmer_compare_single2a2(species_vec1, train_chromvec, test_chromvec, feature_idx, type_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

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

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

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
		t_signal_ori = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal = t_signal_ori[trans_id1]
		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]
		t_signal = np.hstack((t_gc,t_signal))
		print("trans_id1", trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal = t_signal_ori[trans_id2]
		trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		t_gc = gc_signal[trans_id2a]
		t_gc = t_gc[:,feature_idx]
		t_signal = np.hstack((t_gc,t_signal))
		print("trans_id2", trans_id2.shape, t_signal.shape)

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

	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.2, random_state=42)

	vec1 = []
	print("LR")
	reg = LinearRegression().fit(x_train, y_train)
	y_predicted_valid = reg.predict(x_valid)
	y_predicted_test = reg.predict(x_test)
	print(reg.coef_,reg.intercept_)

	temp1 = score_2a(y_valid, y_predicted_valid)
	vec1.append(temp1)
	print(temp1)
	
	temp1 = score_2a(y_signal_test, y_predicted_test)
	vec1.append(temp1)
	print(temp1)

	# print("RFLR")
	# # RandomForestRegressor
	# regr = RandomForestRegressor(max_depth=5, random_state=0,
	# 						n_estimators=200)
	# regr.fit(x_train, y_train)
	# print(regr.feature_importances_)
	# np.save('regr_featureImp',regr.feature_importances_)
	# y_predicted_valid = reg.predict(x_valid)
	# y_predicted_test = reg.predict(x_test)
	# print(reg.coef_,reg.intercept_)

	# temp1 = score_2a(y_valid, y_predicted_valid)
	# vec1.append(temp1)
	# print(temp1)
	
	# temp1 = score_2a(y_signal_test, y_predicted_test)
	# vec1.append(temp1)
	# print(temp1)

	# print("Regressor")
	# xgb_model = xgboost.XGBRegressor(colsample_bytree=0.5,
	# 			 gamma=0,                 
	# 			 learning_rate=0.07,
	# 			 max_depth=10,
	# 			 min_child_weight=1.5,
	# 			 n_estimators=1000,                                                                    
	# 			 reg_alpha=0.75,
	# 			 reg_lambda=0.45,
	# 			 objective='reg:squarederror',
	# 			 n_jobs=50,
	# 			 subsample=0.6,
	# 			 seed=42) 

	# # objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	# xgb_model.fit(x_train, y_train)
	# y_predicted_valid = xgb_model.predict(x_valid)
	# y_predicted_test = xgb_model.predict(x_test)

	# print("train",train_chromvec)
	# score1 = mean_squared_error(y_valid, y_predicted_valid)
	# score2 = pearsonr(y_valid,y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)

	# print("test",test_chromvec)
	# score1 = mean_squared_error(y_signal_test, y_predicted_test)
	# score2 = pearsonr(y_signal_test,y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	print(vec1)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1

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

def read_phyloP_single(species_name,input_filename,chrom_id):

	path1 = '/volume01/yy3/seq_data/dl/replication_timing'
	# filename1 = '%s/estimate_rt/estimate_rt_%s.2.txt'%(path1,species_name)
	if input_filename==-1:
		filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path1,species_name)
	else:
		filename1 = input_filename
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
		data1.to_csv('phyloP_%s.2.txt'%(chrom_id),sep='\t',index=False)

	return vec1

# phyloP
def kmer_compare_single2a2_1(species_vec1, train_chromvec, test_chromvec, feature_idx, feature_idx1, type_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

	path2 = '/volume01/yy3/seq_data/dl/replication_timing'

	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])

	# filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])
	print(signal.shape)
	print(feature_idx)

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

	# phyloP_score = read_phyloP(species_name)
	# print(gc_signal.shape,phyloP_score.shape)

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
		# id1_ori = np.where(chrom_ori==chrom_id1)[0]
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori = np.asarray(file2)
		# trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		# t_signal = t_signal_ori[trans_id1]
		t_signal = t_signal_ori

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,serial[id1])	# mapped index
		if feature_idx1==-5:
			# t_phyloP = phyloP_score[trans_id1]
			t_phyloP = phyloP_score
		elif feature_idx1==-6:
			# t_phyloP = phyloP_score[trans_id1,0:-4]
			t_phyloP = phyloP_score[:,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id1,feature_idx1]
			# t_phyloP = phyloP_score[trans_id1]
			# t_phyloP = t_phylo[:,feature_idx1]
			t_phyloP = phyloP_score[:,feature_idx1]
			# if len(feature_idx1)==1:P
			# 	t_phyloP = np.expand_dims(t_phyloP,axis=1)

		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]
		
		print(t_gc.shape,t_phyloP.shape,t_signal.shape)
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id1", chrom_id1, trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal = t_signal_ori[trans_id2]

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id2 = mapping_Idx(t_serial,serial[id2])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id2]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id2,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id2,feature_idx1]
			t_phyloP = phyloP_score[trans_id2]
			t_phyloP = t_phyloP[:,feature_idx1]
			# if len(feature_idx1)==1:
			# 	t_phyloP = np.expand_dims(t_phyloP,axis=1)
		
		# trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		# t_gc = gc_signal[trans_id2a]
		# t_gc = t_gc[:,feature_idx]
		t_gc = gc_signal[:,feature_idx]
		
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id2", chrom_id, trans_id2.shape, t_signal.shape)

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
	# print(x_train1.shape,x_test.shape)

	if type_id==1 or type_id==2:
		y_signal_train1 = np.hstack((signal[train_sel_idx],t_signal))
	else:
		y_signal_train1 = signal[train_sel_idx]

	y_signal_test = signal[test_sel_idx]
	# print(x_train1.shape,y_signal_train1.shape)
	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)
	print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_signal_test.shape)

	vec1 = []
	print("LR")
	reg = LinearRegression().fit(x_train, y_train)
	y_predicted_valid = reg.predict(x_valid)
	y_predicted_test = reg.predict(x_test)
	print(reg.coef_,reg.intercept_)

	temp1 = score_2a(y_valid, y_predicted_valid)
	vec1.append(temp1)
	print(temp1)
	
	temp1 = score_2a(y_signal_test, y_predicted_test)
	vec1.append(temp1)
	print(temp1)

	# print('SVR')
	# clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
	# clf.fit(x_train, y_train)
	# y_predicted_valid1 = reg.predict(x_valid)
	# y_predicted_test1 = reg.predict(x_test)

	# temp1 = score_2a(y_valid, y_predicted_valid1)
	# vec1.append(temp1)
	# print(temp1)

	# temp1 = score_2a(y_signal_test, y_predicted_test1)
	# vec1.append(temp1)
	# print(temp1)

	dict1 = dict()
	dict1['vec1'] = vec1
	dict1['y_valid'], dict1['y_test'] = y_valid, y_signal_test
	dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test
	# dict1['y_predicted_valid1'], dict1['y_predicted_test1'] = y_predicted_valid1, y_predicted_test1

	# print("RFLR")
	# # RandomForestRegressor
	# regr = RandomForestRegressor(max_depth=5, random_state=0,
	# 						n_estimators=200)
	# regr.fit(x_train, y_train)
	# print(regr.feature_importances_)
	# np.save('regr_featureImp',regr.feature_importances_)
	# y_predicted_valid = reg.predict(x_valid)
	# y_predicted_test = reg.predict(x_test)
	# print(reg.coef_,reg.intercept_)

	# temp1 = score_2a(y_valid, y_predicted_valid)
	# vec1.append(temp1)
	# print(temp1)
	
	# temp1 = score_2a(y_signal_test, y_predicted_test)
	# vec1.append(temp1)
	# print(temp1)

	# print("Regressor")
	# xgb_model = xgboost.XGBRegressor(colsample_bytree=0.5,
	# 			 gamma=0,                 
	# 			 learning_rate=0.07,
	# 			 max_depth=10,
	# 			 min_child_weight=1.5,
	# 			 n_estimators=1000,                                                                    
	# 			 reg_alpha=0.75,
	# 			 reg_lambda=0.45,
	# 			 objective='reg:squarederror',
	# 			 n_jobs=50,
	# 			 subsample=0.6,
	# 			 seed=42) 

	# # objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	# xgb_model.fit(x_train, y_train)
	# y_predicted_valid = xgb_model.predict(x_valid)
	# y_predicted_test = xgb_model.predict(x_test)

	# print("train",train_chromvec)
	# score1 = mean_squared_error(y_valid, y_predicted_valid)
	# score2 = pearsonr(y_valid,y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)

	# print("test",test_chromvec)
	# score1 = mean_squared_error(y_signal_test, y_predicted_test)
	# score2 = pearsonr(y_signal_test,y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	print(vec1)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1,dict1

# phyloP and GC profile
def kmer_compare_single2a2_1a(species_vec1, train_chromvec, test_chromvec, feature_idx, feature_idx1, type_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

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

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

	# phyloP_score = read_phyloP(species_name)
	# print(gc_signal.shape,phyloP_score.shape)

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
		# filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		# file2 = np.load(filename2)
		# t_signal_ori = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		# t_signal = t_signal_ori[trans_id1]

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,serial[id1])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id1]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id1,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id1,feature_idx1]
			t_phyloP = phyloP_score[trans_id1]
			t_phyloP = t_phyloP[:,feature_idx1]
			# if len(feature_idx1)==1:
			# 	t_phyloP = np.expand_dims(t_phyloP,axis=1)

		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]
		
		# print(t_gc.shape,t_phyloP.shape,t_signal.shape)
		print(t_gc.shape,t_phyloP.shape)
		# t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		# t_signal = np.hstack((t_gc,t_phyloP))
		t_signal = t_gc
		print("trans_id1", chrom_id1, trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		# filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		# file2 = np.load(filename2)
		# t_signal_ori = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		# t_signal = t_signal_ori[trans_id2]

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id2 = mapping_Idx(t_serial,serial[id2])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id2]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id2,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id2,feature_idx1]
			t_phyloP = phyloP_score[trans_id2]
			t_phyloP = t_phyloP[:,feature_idx1]
			# if len(feature_idx1)==1:
			# 	t_phyloP = np.expand_dims(t_phyloP,axis=1)
		
		trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		t_gc = gc_signal[trans_id2a]
		t_gc = t_gc[:,feature_idx]
		
		print(t_gc.shape,t_phyloP.shape)
		# t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		# t_signal = np.hstack((t_gc,t_phyloP))
		t_signal = t_gc
		print("trans_id2", chrom_id, trans_id2.shape, t_signal.shape)

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
	# print(x_train1.shape,x_test.shape)

	if type_id==1 or type_id==2:
		y_signal_train1 = np.hstack((signal[train_sel_idx],t_signal))
	else:
		y_signal_train1 = signal[train_sel_idx]

	y_signal_test = signal[test_sel_idx]
	# print(x_train1.shape,y_signal_train1.shape)
	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)
	print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_signal_test.shape)

	vec1 = []
	print("LR")
	reg = LinearRegression().fit(x_train, y_train)
	y_predicted_valid = reg.predict(x_valid)
	y_predicted_test = reg.predict(x_test)
	print(reg.coef_,reg.intercept_)

	temp1 = score_2a(y_valid, y_predicted_valid)
	vec1.append(temp1)
	print(temp1)
	
	temp1 = score_2a(y_signal_test, y_predicted_test)
	vec1.append(temp1)
	print(temp1)

	# print('SVR')
	# clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
	# clf.fit(x_train, y_train)
	# y_predicted_valid1 = reg.predict(x_valid)
	# y_predicted_test1 = reg.predict(x_test)

	# temp1 = score_2a(y_valid, y_predicted_valid1)
	# vec1.append(temp1)
	# print(temp1)

	# temp1 = score_2a(y_signal_test, y_predicted_test1)
	# vec1.append(temp1)
	# print(temp1)

	dict1 = dict()
	dict1['vec1'] = vec1
	dict1['y_valid'], dict1['y_test'] = y_valid, y_signal_test
	dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test
	# dict1['y_predicted_valid1'], dict1['y_predicted_test1'] = y_predicted_valid1, y_predicted_test1
	print(vec1)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1,dict1

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

def dimension_reduction1(x_train,x_test,feature_dim,shuffle,sub_sample,type_id):

	if shuffle==1 and sub_sample>0:
		idx = np.random.permutation(x_train.shape[0])
	else:
		idx = np.asarray(range(0,x_train.shape[0]))
	if (sub_sample>0) and (type_id!=7) and (type_id!=11):
		id1 = idx[0:sub_sample]
	else:
		id1 = idx

	if type_id==0:
		# PCA
		pca = PCA(n_components=feature_dim, whiten = False, random_state = 0)
		if sub_sample>0:
			pca.fit(x_train[id1,:])
		else:
			pca.fit(x_train)
		x = pca.transform(x_train)
		x1 = pca.transform(x_test)
		# X_pca_reconst = pca.inverse_transform(x)
	elif type_id==1:
		# Incremental PCA
		n_batches = 10
		inc_pca = IncrementalPCA(n_components=feature_dim)
		for X_batch in np.array_split(x_train, n_batches):
			inc_pca.partial_fit(X_batch)
		x = inc_pca.transform(x_train)
		x1 = inc_pca.transform(x_test)
		# X_ipca_reconst = inc_pca.inverse_transform(x)
	elif type_id==2:
		# Kernel PCA
		kpca = KernelPCA(kernel="rbf",n_components=feature_dim, gamma=None, fit_inverse_transform=True, random_state = 0, n_jobs=50)
		kpca.fit(x_train[id1,:])
		x = kpca.transform(x_train)
		x1 = kpca.transform(x_test)
		# X_kpca_reconst = kpca.inverse_transform(x)
	elif type_id==3:
		# Sparse PCA
		sparsepca = SparsePCA(n_components=feature_dim, alpha=0.0001, random_state=0, n_jobs=50)
		sparsepca.fit(x_train[id1,:])
		x = sparsepca.transform(x_train)
		x1 = sparsepca.transform(x_test)
	elif type_id==4:
		# SVD
		SVD_ = TruncatedSVD(n_components=feature_dim,algorithm='randomized', random_state=0, n_iter=5)
		SVD_.fit(x_train[id1,:])
		x = SVD_.transform(x_train)
		x1 = SVD_.transform(x_test)
		# X_svd_reconst = SVD_.inverse_transform(x)
	elif type_id==5:
		# Gaussian Random Projection
		GRP = GaussianRandomProjection(n_components=feature_dim,eps = 0.5, random_state=2019)
		GRP.fit(x_train[id1,:])
		x = GRP.transform(x_train)
		x1 = GRP.transform(x_test)
	elif type_id==6:
		# Sparse random projection
		SRP = SparseRandomProjection(n_components=feature_dim,density = 'auto', eps = 0.5, random_state=2019, dense_output = False)
		SRP.fit(x_train[id1,:])
		x = SRP.transform(x_train)
		x1 = SRP.transform(x_test)
	# elif type_id==7:
	# 	# MDS
	# 	mds = MDS(n_components=feature_dim, n_init=12, max_iter=1200, metric=True, n_jobs=4, random_state=2019)
	# 	x = mds.fit_transform(x_train[id1])
	elif type_id==8:
		# ISOMAP
		isomap = Isomap(n_components=feature_dim, n_jobs = 4, n_neighbors = 5)
		isomap.fit(x_train[id1,:])
		x = isomap.transform(x_train)
		x1 = isomap.transform(x_test)
	elif type_id==9:
		# MiniBatch dictionary learning
		miniBatchDictLearning = MiniBatchDictionaryLearning(n_components=feature_dim,batch_size = 1000,alpha = 1,n_iter = 25,  random_state=2019)
		if sub_sample>0:
			miniBatchDictLearning.fit(x_train[id1,:])
			x = miniBatchDictLearning.transform(x_train)
		else:
			x = miniBatchDictLearning.fit_transform(x_train)
		x1 = miniBatchDictLearning.transform(x_test)
	elif type_id==10:
		# ICA
		fast_ICA = FastICA(n_components=feature_dim, algorithm = 'parallel',whiten = True,max_iter = 100,  random_state=2019)
		if sub_sample>0:
			fast_ICA.fit(x_train[id1])
			x = fast_ICA.transform(x_train)
		else:
			x = fast_ICA.fit_transform(x_train)
		x1 = fast_ICA.transform(x_test)
		# X_fica_reconst = FastICA.inverse_	# elif type_id==11:
	# 	# t-SNE
	# 	tsne = TSNE(n_components=feature_dim,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
	# 	x = tsne.fit_transform(x_ori)transform(x)

	elif type_id==12:
		# Locally linear embedding
		lle = LocallyLinearEmbedding(n_components=feature_dim, n_neighbors = np.max((int(feature_dim*1.5),500)),method = 'modified', n_jobs = 20,  random_state=2019)
		lle.fit(x_train[id1,:])
		x = lle.transform(x_train)
		x1 = lle.transform(x_test)
	elif type_id==13:
		# Autoencoder
		feature_dim_ori = x_train.shape[1]
		m = Sequential()
		m.add(Dense(512,  activation='elu', input_shape=(feature_dim_ori,)))
		# m.add(Dense(256,  activation='elu'))
		m.add(Dense(feature_dim,   activation='linear', name="bottleneck"))
		# m.add(Dense(256,  activation='elu'))
		m.add(Dense(512,  activation='elu'))
		m.add(Dense(feature_dim_ori,  activation='sigmoid'))
		m.compile(loss='mean_squared_error', optimizer = Adam())
		history = m.fit(x_train[id1], x_train[id1], batch_size=256, epochs=20, verbose=1)

		encoder = Model(m.input, m.get_layer('bottleneck').output)
		x = encoder.predict(x_train)
		Renc = m.predict(x_train)
		x1 = encoder.predict(x_test)

	return x, x1

def feature_transform1(x_train, x_test, feature_dim_kmer, feature_dim, shuffle, sub_sample_ratio, type_id, normalize):

	# x_ori1 = np.vstack((x_train,x_test))
	dim1 = x_train.shape[1]
	dim2 = dim1-feature_dim_kmer
	print("feature_dim_kmer",feature_dim_kmer,dim2)
	x_train1 = x_train[:,dim2:]
	x_test1 = x_test[:,dim2:]
	if normalize>=1:
		sc = StandardScaler()
		x_train1 = sc.fit_transform(x_train1)	# normalize data
		x_test1 = sc.transform(x_test1)	# normalize data
	# x_train_sub = sc.fit_transform(x_ori[0:num_train,:])
	# x_test_sub = sc.transform(x_ori[num_train+num_test,:])
	num_train, num_test = x_train1.shape[0], x_test1.shape[0]
	vec1 = ['PCA','Incremental PCA','Kernel PCA','Sparse PCA','SVD','GRP','SRP','MDS','ISOMAP','Minibatch','ICA','tSNE','LLE','Encoder']
	start = time.time()
	if sub_sample_ratio<1:
		sub_sample = int(x_train1.shape[0]*sub_sample_ratio)
	else:
		sub_sample = -1
	x, x1 = dimension_reduction1(x_train1, x_test1, feature_dim, shuffle, sub_sample, type_id)
	stop = time.time()
	print("feature transform %s"%(vec1[type_id]),stop - start)
	x_1 = np.hstack((x_train[:,0:dim2],x))
	x_2 = np.hstack((x_test[:,0:dim2],x1))
	if normalize>=2:
		sc = StandardScaler()
		x_1 = sc.fit_transform(x_1)
		x_2 = sc.transform(x_2)
	x_train1, x_test1 = x_1, x_2
	print(x_train.shape,x_train1.shape,x_test.shape,x_test1.shape)

	return x_train1, x_test1

def kmer_compare_single2a2_2(species_vec1, train_chromvec, test_chromvec, feature_idx, feature_idx1, type_id, feature_dim_transform, t_list, normalize, run_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

	path2 = '/volume01/yy3/seq_data/dl/replication_timing'

	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])

	filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	# filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])
	print(signal.shape)
	print(feature_idx)

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

	# phyloP_score = read_phyloP(species_name)
	# print(gc_signal.shape,phyloP_score.shape)

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
		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal = t_signal_ori[trans_id1]
		feature_dim_kmer = t_signal_ori.shape[1]

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,serial[id1])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id1]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id1,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id1,feature_idx1]
			t_phyloP = phyloP_score[trans_id1]
			t_phyloP = t_phyloP[:,feature_idx1]

		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]
		
		print(t_gc.shape,t_phyloP.shape,t_signal.shape)
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id1", chrom_id1, trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal = t_signal_ori[trans_id2]

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id2 = mapping_Idx(t_serial,serial[id2])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id2]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id2,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id2,feature_idx1]
			t_phyloP = phyloP_score[trans_id2]
			t_phyloP = t_phyloP[:,feature_idx1]
		
		trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		t_gc = gc_signal[trans_id2a]
		t_gc = t_gc[:,feature_idx]
		
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id2", chrom_id, trans_id2.shape, t_signal.shape)

		x_test.extend(t_signal)

	print(len(train_sel_idx),len(test_sel_idx))
	# type_id = 1 or type_id = 2: add new species
	# if type_id==1 or type_id==2:
	# 	train_sel_idx1 = map_idx[train_sel_idx]
	# 	test_sel_idx1 = map_idx[test_sel_idx]
	# 	t_signal = []
	# 	num1 = len(species_vec1)

	# 	# train in multiple species
	# 	for i in range(1,num1):
	# 		species_name = species_vec1[i]
	# 		filename1 = '%s/training2_kmer_%s.npy'%(path2,species_name)
	# 		data1 = np.load(filename1)
	# 		data1_sub = data1[train_sel_idx1]
	# 		# data_vec.append(data1_sub)
	# 		x_train1.extend(np.asarray(data1_sub))

	# 		filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	# 		# filename2a = 'test_seq_%s.1.txt'%(species_name)
	# 		file1 = pd.read_csv(filename1,sep='\t')
	# 		signal = np.asarray(file1['signal'])
	# 		t_signal.extend(signal[train_sel_idx])

	# 	t_signal = np.asarray(t_signal)
		
	x_train1_ori, x_test_ori = np.asarray(x_train1), np.asarray(x_test)
	# x_train1 = x_train1[:,feature_idx]
	# x_test = x_test[:,feature_idx]
	# print(x_train1.shape,x_test.shape)

	if type_id==1 or type_id==2:
		y_signal_train1 = np.hstack((signal[train_sel_idx],t_signal))
	else:
		y_signal_train1 = signal[train_sel_idx]

	y_signal_test = signal[test_sel_idx]
	# print(x_train1.shape,y_signal_train1.shape)
	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	feature_dim = feature_dim_transform
	sub_sample_ratio = 1
	shuffle = 0
	vec2 = dict()
	m_corr, m_explain = [0,0], [0,0]
	for type_id2 in t_list:
		x_train1, x_test = feature_transform(x_train1_ori, x_test_ori, feature_dim_kmer, feature_dim, shuffle, 
												sub_sample_ratio, type_id2, normalize)

		x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)
		print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_signal_test.shape)

		vec1 = []
		print("LR")
		reg = LinearRegression().fit(x_train, y_train)
		y_predicted_valid = reg.predict(x_valid)
		y_predicted_test = reg.predict(x_test)
		print(reg.coef_,reg.intercept_)

		temp1 = score_2a(y_valid, y_predicted_valid)
		vec1.append(temp1)
		print(temp1)
	
		temp1 = score_2a(y_signal_test, y_predicted_test)
		vec1.append(temp1)
		print(temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		dict1['y_valid'], dict1['y_test'] = y_valid, y_signal_test
		dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test

		vec2[type_id2] = dict1
		temp1 = vec1[1]
		if temp1[1][0]>m_corr[1]:
			m_corr = [type_id2,temp1[1][0]]
		if temp1[2]>m_explain[1]:
			m_explain = [type_id2,temp1[2]]

	print(m_corr,m_explain)
	np.save('feature_transform_%d_%d_%d'%(t_list[0],feature_dim,run_id),vec2,allow_pickle=True)

	# print('SVR')
	# clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
	# clf.fit(x_train, y_train)
	# y_predicted_valid1 = reg.predict(x_valid)
	# y_predicted_test1 = reg.predict(x_test)

	# temp1 = score_2a(y_valid, y_predicted_valid1)
	# vec1.append(temp1)
	# print(temp1)

	# temp1 = score_2a(y_signal_test, y_predicted_test1)
	# vec1.append(temp1)
	# print(temp1)

	# dict1['y_predicted_valid1'], dict1['y_predicted_test1'] = y_predicted_valid1, y_predicted_test1

	# print("RFLR")
	# # RandomForestRegressor
	# regr = RandomForestRegressor(max_depth=5, random_state=0,
	# 						n_estimators=200)
	# regr.fit(x_train, y_train)
	# print(regr.feature_importances_)
	# np.save('regr_featureImp',regr.feature_importances_)
	# y_predicted_valid = reg.predict(x_valid)
	# y_predicted_test = reg.predict(x_test)
	# print(reg.coef_,reg.intercept_)

	# temp1 = score_2a(y_valid, y_predicted_valid)
	# vec1.append(temp1)
	# print(temp1)
	
	# temp1 = score_2a(y_signal_test, y_predicted_test)
	# vec1.append(temp1)
	# print(temp1)

	# print("Regressor")
	# xgb_model = xgboost.XGBRegressor(colsample_bytree=0.5,
	# 			 gamma=0,                 
	# 			 learning_rate=0.07,
	# 			 max_depth=10,
	# 			 min_child_weight=1.5,
	# 			 n_estimators=1000,                                                                    
	# 			 reg_alpha=0.75,
	# 			 reg_lambda=0.45,
	# 			 objective='reg:squarederror',
	# 			 n_jobs=50,
	# 			 subsample=0.6,
	# 			 seed=42) 

	# # objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	# xgb_model.fit(x_train, y_train)
	# y_predicted_valid = xgb_model.predict(x_valid)
	# y_predicted_test = xgb_model.predict(x_test)

	# print("train",train_chromvec)
	# score1 = mean_squared_error(y_valid, y_predicted_valid)
	# score2 = pearsonr(y_valid,y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)

	# print("test",test_chromvec)
	# score1 = mean_squared_error(y_signal_test, y_predicted_test)
	# score2 = pearsonr(y_signal_test,y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	print(vec2)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1,dict1

def kmer_compare_single2a2_2a(species_vec1, train_chromvec, test_chromvec, feature_idx, feature_idx1, type_id, feature_dim_transform, t_list, normalize, run_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

	path2 = '/volume01/yy3/seq_data/dl/replication_timing'

	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])

	filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	# filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])
	print(signal.shape)
	print(feature_idx)

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

	# phyloP_score = read_phyloP(species_name)
	# print(gc_signal.shape,phyloP_score.shape)

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
		t_signal_ori1 = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal1 = t_signal_ori1[trans_id1]
		feature_dim_kmer1 = t_signal_ori1.shape[1]

		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal2 = t_signal_ori2[trans_id1]
		feature_dim_kmer2 = t_signal_ori2.shape[1]
		t_signal = np.hstack((t_signal1,t_signal2))
		feature_dim_kmer = t_signal.shape[1]
		print(feature_dim_kmer1,feature_dim_kmer2,feature_dim_kmer)

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,serial[id1])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id1]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id1,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id1,feature_idx1]
			t_phyloP = phyloP_score[trans_id1]
			t_phyloP = t_phyloP[:,feature_idx1]

		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]
		
		print(t_gc.shape,t_phyloP.shape,t_signal.shape)
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id1", chrom_id1, trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		# filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori1 = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal_ori1 = np.asarray(file2)
		t_signal1 = t_signal_ori1[trans_id2]
		feature_dim_kmer1 = t_signal_ori1.shape[1]

		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)

		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal2 = t_signal_ori2[trans_id2]
		feature_dim_kmer2 = t_signal_ori2.shape[1]
		t_signal = np.hstack((t_signal1,t_signal2))

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id2 = mapping_Idx(t_serial,serial[id2])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id2]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id2,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id2,feature_idx1]
			t_phyloP = phyloP_score[trans_id2]
			t_phyloP = t_phyloP[:,feature_idx1]
		
		trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		t_gc = gc_signal[trans_id2a]
		t_gc = t_gc[:,feature_idx]
		
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id2", chrom_id, trans_id2.shape, t_signal.shape)

		x_test.extend(t_signal)

	print(len(train_sel_idx),len(test_sel_idx))
	# type_id = 1 or type_id = 2: add new species
	# if type_id==1 or type_id==2:
	# 	train_sel_idx1 = map_idx[train_sel_idx]
	# 	test_sel_idx1 = map_idx[test_sel_idx]
	# 	t_signal = []
	# 	num1 = len(species_vec1)

	# 	# train in multiple species
	# 	for i in range(1,num1):
	# 		species_name = species_vec1[i]
	# 		filename1 = '%s/training2_kmer_%s.npy'%(path2,species_name)
	# 		data1 = np.load(filename1)
	# 		data1_sub = data1[train_sel_idx1]
	# 		# data_vec.append(data1_sub)
	# 		x_train1.extend(np.asarray(data1_sub))

	# 		filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	# 		# filename2a = 'test_seq_%s.1.txt'%(species_name)
	# 		file1 = pd.read_csv(filename1,sep='\t')
	# 		signal = np.asarray(file1['signal'])
	# 		t_signal.extend(signal[train_sel_idx])

	# 	t_signal = np.asarray(t_signal)
		
	x_train1_ori, x_test_ori = np.asarray(x_train1), np.asarray(x_test)
	# x_train1 = x_train1[:,feature_idx]
	# x_test = x_test[:,feature_idx]
	# print(x_train1.shape,x_test.shape)

	if type_id==1 or type_id==2:
		y_signal_train1 = np.hstack((signal[train_sel_idx],t_signal))
	else:
		y_signal_train1 = signal[train_sel_idx]

	y_signal_test = signal[test_sel_idx]
	# print(x_train1.shape,y_signal_train1.shape)
	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	feature_dim = feature_dim_transform
	sub_sample_ratio = 1
	shuffle = 0
	vec2 = dict()
	m_corr, m_explain = [0,0], [0,0]
	for type_id2 in t_list:
		x_train1, x_test = feature_transform(x_train1_ori, x_test_ori, feature_dim_kmer, feature_dim, shuffle, 
												sub_sample_ratio, type_id2, normalize)

		x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)
		print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_signal_test.shape)

		vec1 = []
		print("LR")
		reg = LinearRegression().fit(x_train, y_train)
		y_predicted_valid = reg.predict(x_valid)
		y_predicted_test = reg.predict(x_test)
		print(reg.coef_,reg.intercept_)

		temp1 = score_2a(y_valid, y_predicted_valid)
		vec1.append(temp1)
		print(temp1)
	
		temp1 = score_2a(y_signal_test, y_predicted_test)
		vec1.append(temp1)
		print(temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		dict1['y_valid'], dict1['y_test'] = y_valid, y_signal_test
		dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test

		vec2[type_id2] = dict1
		temp1 = vec1[1]
		if temp1[1][0]>m_corr[1]:
			m_corr = [type_id2,temp1[1][0]]
		if temp1[2]>m_explain[1]:
			m_explain = [type_id2,temp1[2]]

	print(m_corr,m_explain)
	np.save('feature_transform_%d_%d_%d'%(t_list[0],feature_dim,run_id),vec2,allow_pickle=True)

	# print('SVR')
	# clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
	# clf.fit(x_train, y_train)
	# y_predicted_valid1 = reg.predict(x_valid)
	# y_predicted_test1 = reg.predict(x_test)

	# temp1 = score_2a(y_valid, y_predicted_valid1)
	# vec1.append(temp1)
	# print(temp1)

	# temp1 = score_2a(y_signal_test, y_predicted_test1)
	# vec1.append(temp1)
	# print(temp1)

	# dict1['y_predicted_valid1'], dict1['y_predicted_test1'] = y_predicted_valid1, y_predicted_test1

	# print("RFLR")
	# # RandomForestRegressor
	# regr = RandomForestRegressor(max_depth=5, random_state=0,
	# 						n_estimators=200)
	# regr.fit(x_train, y_train)
	# print(regr.feature_importances_)
	# np.save('regr_featureImp',regr.feature_importances_)
	# y_predicted_valid = reg.predict(x_valid)
	# y_predicted_test = reg.predict(x_test)
	# print(reg.coef_,reg.intercept_)

	# temp1 = score_2a(y_valid, y_predicted_valid)
	# vec1.append(temp1)
	# print(temp1)
	
	# temp1 = score_2a(y_signal_test, y_predicted_test)
	# vec1.append(temp1)
	# print(temp1)

	# print("Regressor")
	# xgb_model = xgboost.XGBRegressor(colsample_bytree=0.5,
	# 			 gamma=0,                 
	# 			 learning_rate=0.07,
	# 			 max_depth=10,
	# 			 min_child_weight=1.5,
	# 			 n_estimators=1000,                                                                    
	# 			 reg_alpha=0.75,
	# 			 reg_lambda=0.45,
	# 			 objective='reg:squarederror',
	# 			 n_jobs=50,
	# 			 subsample=0.6,
	# 			 seed=42) 

	# # objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	# xgb_model.fit(x_train, y_train)
	# y_predicted_valid = xgb_model.predict(x_valid)
	# y_predicted_test = xgb_model.predict(x_test)

	# print("train",train_chromvec)
	# score1 = mean_squared_error(y_valid, y_predicted_valid)
	# score2 = pearsonr(y_valid,y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)

	# print("test",test_chromvec)
	# score1 = mean_squared_error(y_signal_test, y_predicted_test)
	# score2 = pearsonr(y_signal_test,y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	print(vec2)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1,dict1

def kmer_compare_single2a2_2a1(species_vec1, train_chromvec, test_chromvec, feature_idx, feature_idx1, type_id, feature_dim_transform, t_list, normalize, run_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

	path2 = '/volume01/yy3/seq_data/dl/replication_timing'

	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])

	filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	# filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])
	print(signal.shape)
	print(feature_idx)

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

	# phyloP_score = read_phyloP(species_name)
	# print(gc_signal.shape,phyloP_score.shape)

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
		t_signal_ori1 = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal1 = t_signal_ori1[trans_id1]
		feature_dim_kmer1 = t_signal_ori1.shape[1]

		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal2 = t_signal_ori2[trans_id1]
		feature_dim_kmer2 = t_signal_ori2.shape[1]

		filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal3 = t_signal_ori2[trans_id1]
		feature_dim_kmer3 = t_signal_ori2.shape[1]

		t_signal = np.hstack((t_signal1,t_signal2,t_signal3))
		feature_dim_kmer = t_signal.shape[1]
		print(feature_dim_kmer1,feature_dim_kmer2,feature_dim_kmer3,feature_dim_kmer)

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,serial[id1])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id1]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id1,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id1,feature_idx1]
			t_phyloP = phyloP_score[trans_id1]
			t_phyloP = t_phyloP[:,feature_idx1]

		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]
		
		print(t_gc.shape,t_phyloP.shape,t_signal.shape)
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id1", chrom_id1, trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		# filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori1 = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal_ori1 = np.asarray(file2)
		t_signal1 = t_signal_ori1[trans_id2]
		feature_dim_kmer1 = t_signal_ori1.shape[1]

		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal2 = t_signal_ori2[trans_id2]
		feature_dim_kmer2 = t_signal_ori2.shape[1]

		filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal3 = t_signal_ori2[trans_id2]
		feature_dim_kmer3 = t_signal_ori2.shape[1]
		t_signal = np.hstack((t_signal1,t_signal2,t_signal3))

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id2 = mapping_Idx(t_serial,serial[id2])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id2]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id2,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id2,feature_idx1]
			t_phyloP = phyloP_score[trans_id2]
			t_phyloP = t_phyloP[:,feature_idx1]
		
		trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		t_gc = gc_signal[trans_id2a]
		t_gc = t_gc[:,feature_idx]
		
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id2", chrom_id, trans_id2.shape, t_signal.shape)

		x_test.extend(t_signal)

	print(len(train_sel_idx),len(test_sel_idx))
	# type_id = 1 or type_id = 2: add new species
	# if type_id==1 or type_id==2:
	# 	train_sel_idx1 = map_idx[train_sel_idx]
	# 	test_sel_idx1 = map_idx[test_sel_idx]
	# 	t_signal = []
	# 	num1 = len(species_vec1)

	# 	# train in multiple species
	# 	for i in range(1,num1):
	# 		species_name = species_vec1[i]
	# 		filename1 = '%s/training2_kmer_%s.npy'%(path2,species_name)
	# 		data1 = np.load(filename1)
	# 		data1_sub = data1[train_sel_idx1]
	# 		# data_vec.append(data1_sub)
	# 		x_train1.extend(np.asarray(data1_sub))

	# 		filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	# 		# filename2a = 'test_seq_%s.1.txt'%(species_name)
	# 		file1 = pd.read_csv(filename1,sep='\t')
	# 		signal = np.asarray(file1['signal'])
	# 		t_signal.extend(signal[train_sel_idx])

	# 	t_signal = np.asarray(t_signal)
		
	x_train1_ori, x_test_ori = np.asarray(x_train1), np.asarray(x_test)
	# x_train1 = x_train1[:,feature_idx]
	# x_test = x_test[:,feature_idx]
	# print(x_train1.shape,x_test.shape)

	if type_id==1 or type_id==2:
		y_signal_train1 = np.hstack((signal[train_sel_idx],t_signal))
	else:
		y_signal_train1 = signal[train_sel_idx]

	y_signal_test = signal[test_sel_idx]
	# print(x_train1.shape,y_signal_train1.shape)
	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	feature_dim = feature_dim_transform
	sub_sample_ratio = 1
	shuffle = 0
	vec2 = dict()
	m_corr, m_explain = [0,0], [0,0]
	for type_id2 in t_list:
		x_train1, x_test = feature_transform(x_train1_ori, x_test_ori, feature_dim_kmer, feature_dim, shuffle, 
												sub_sample_ratio, type_id2, normalize)

		x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)
		print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_signal_test.shape)

		vec1 = []
		print("LR")
		reg = LinearRegression().fit(x_train, y_train)
		y_predicted_valid = reg.predict(x_valid)
		y_predicted_test = reg.predict(x_test)
		print(reg.coef_,reg.intercept_)

		temp1 = score_2a(y_valid, y_predicted_valid)
		vec1.append(temp1)
		print(temp1)
	
		temp1 = score_2a(y_signal_test, y_predicted_test)
		vec1.append(temp1)
		print(temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		dict1['y_valid'], dict1['y_test'] = y_valid, y_signal_test
		dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test

		vec2[type_id2] = dict1
		temp1 = vec1[1]
		if temp1[1][0]>m_corr[1]:
			m_corr = [type_id2,temp1[1][0]]
		if temp1[2]>m_explain[1]:
			m_explain = [type_id2,temp1[2]]

	print(m_corr,m_explain)
	np.save('feature_transform_%d_%d_%d'%(t_list[0],feature_dim,run_id),vec2,allow_pickle=True)

	# print('SVR')
	# clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
	# clf.fit(x_train, y_train)
	# y_predicted_valid1 = reg.predict(x_valid)
	# y_predicted_test1 = reg.predict(x_test)

	# temp1 = score_2a(y_valid, y_predicted_valid1)
	# vec1.append(temp1)
	# print(temp1)

	# temp1 = score_2a(y_signal_test, y_predicted_test1)
	# vec1.append(temp1)
	# print(temp1)

	# dict1['y_predicted_valid1'], dict1['y_predicted_test1'] = y_predicted_valid1, y_predicted_test1

	# print("RFLR")
	# # RandomForestRegressor
	# regr = RandomForestRegressor(max_depth=5, random_state=0,
	# 						n_estimators=200)
	# regr.fit(x_train, y_train)
	# print(regr.feature_importances_)
	# np.save('regr_featureImp',regr.feature_importances_)
	# y_predicted_valid = reg.predict(x_valid)
	# y_predicted_test = reg.predict(x_test)
	# print(reg.coef_,reg.intercept_)

	# temp1 = score_2a(y_valid, y_predicted_valid)
	# vec1.append(temp1)
	# print(temp1)
	
	# temp1 = score_2a(y_signal_test, y_predicted_test)
	# vec1.append(temp1)
	# print(temp1)

	# print("Regressor")
	# xgb_model = xgboost.XGBRegressor(colsample_bytree=0.5,
	# 			 gamma=0,                 
	# 			 learning_rate=0.07,
	# 			 max_depth=10,
	# 			 min_child_weight=1.5,
	# 			 n_estimators=1000,                                                                    
	# 			 reg_alpha=0.75,
	# 			 reg_lambda=0.45,
	# 			 objective='reg:squarederror',
	# 			 n_jobs=50,
	# 			 subsample=0.6,
	# 			 seed=42) 

	# # objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	# xgb_model.fit(x_train, y_train)
	# y_predicted_valid = xgb_model.predict(x_valid)
	# y_predicted_test = xgb_model.predict(x_test)

	# print("train",train_chromvec)
	# score1 = mean_squared_error(y_valid, y_predicted_valid)
	# score2 = pearsonr(y_valid,y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)

	# print("test",test_chromvec)
	# score1 = mean_squared_error(y_signal_test, y_predicted_test)
	# score2 = pearsonr(y_signal_test,y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	print(vec2)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1,dict1

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

# select sample
def sample_select1(x_mtx, idx_sel_list, tol=5, L=5):

	num1 = len(idx_sel_list)
	feature_dim = x_mtx.shape[1]
	# L = 5
	size1 = 2*L+1
	feature_dim = x_mtx.shape[1]
	vec1_list = np.zeros((num1,size1))
	# feature_list = np.zeros((num1,size1*feature_dim))
	feature_list = np.zeros((num1,size1,feature_dim))
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
		feature_list[i] = t_feature

		if i%10000==0:
			print(i,t_feature.shape,vec1,vec1_list[i])

	return feature_list, vec1_list

def kmer_compare_single2a2_2a2(species_vec1, train_chromvec, test_chromvec, feature_idx, feature_idx1, type_id, feature_dim_transform, t_list, normalize, flanking, run_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

	path2 = '/volume01/yy3/seq_data/dl/replication_timing'

	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])

	filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	# filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])
	print(signal.shape)
	print(feature_idx)

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

	# phyloP_score = read_phyloP(species_name)
	# print(gc_signal.shape,phyloP_score.shape)

	train_sel_idx, test_sel_idx = [], []
	data_vec = []
	x_train1, x_test = [], []
	train_sel_list, test_sel_list = [], []
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
		t_signal_ori1 = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal1 = t_signal_ori1[trans_id1]
		feature_dim_kmer1 = t_signal_ori1.shape[1]

		temp1 = serial[id1]
		n1 = len(temp1)
		temp2 = np.vstack(([int(chrom_id)]*n1,temp1)).T
		train_sel_list.extend(temp2)	# chrom_id, serial

		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal2 = t_signal_ori2[trans_id1]
		feature_dim_kmer2 = t_signal_ori2.shape[1]
		t_signal = np.hstack((t_signal1,t_signal2))
		feature_dim_kmer = t_signal.shape[1]
		print(feature_dim_kmer1,feature_dim_kmer2,feature_dim_kmer)

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,serial[id1])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id1]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id1,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id1,feature_idx1]
			t_phyloP = phyloP_score[trans_id1]
			t_phyloP = t_phyloP[:,feature_idx1]

		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]
		
		print(t_gc.shape,t_phyloP.shape,t_signal.shape)
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id1", chrom_id1, trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		# filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori1 = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal_ori1 = np.asarray(file2)
		t_signal1 = t_signal_ori1[trans_id2]
		feature_dim_kmer1 = t_signal_ori1.shape[1]

		temp2 = serial[id2]
		n2 = len(temp2)
		temp2 = np.vstack(([int(chrom_id)]*n2,temp2)).T
		test_sel_list.extend(temp2)

		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)

		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal2 = t_signal_ori2[trans_id2]
		feature_dim_kmer2 = t_signal_ori2.shape[1]
		t_signal = np.hstack((t_signal1,t_signal2))

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id2 = mapping_Idx(t_serial,serial[id2])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id2]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id2,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id2,feature_idx1]
			t_phyloP = phyloP_score[trans_id2]
			t_phyloP = t_phyloP[:,feature_idx1]
		
		trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		t_gc = gc_signal[trans_id2a]
		t_gc = t_gc[:,feature_idx]
		
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id2", chrom_id, trans_id2.shape, t_signal.shape)

		x_test.extend(t_signal)

	print(len(train_sel_idx),len(test_sel_idx))
		
	x_train1_ori, x_test_ori = np.asarray(x_train1), np.asarray(x_test)
	# x_train1 = x_train1[:,feature_idx]
	# x_test = x_test[:,feature_idx]
	# print(x_train1.shape,x_test.shape)

	if type_id==1 or type_id==2:
		y_signal_train1 = np.hstack((signal[train_sel_idx],t_signal))
	else:
		y_signal_train1 = signal[train_sel_idx]

	y_signal_test = signal[test_sel_idx]
	# print(x_train1.shape,y_signal_train1.shape)
	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	feature_dim = feature_dim_transform
	sub_sample_ratio = 1
	shuffle = 0
	vec2 = dict()
	m_corr, m_explain = [0,0], [0,0]
	tol = 5
	L = flanking
	train_sel_list, test_sel_list = np.asarray(train_sel_list), np.asarray(test_sel_list)
	print(train_sel_list[0:10])
	print(test_sel_list[0:10])
	for type_id2 in t_list:
		x_train1_trans, x_test1_trans = feature_transform(x_train1_ori, x_test_ori, feature_dim_kmer, feature_dim, shuffle, 
												sub_sample_ratio, type_id2, normalize)

		x_train1, vec_train1 = sample_select(x_train1_trans, train_sel_list, tol, L)
		x_test, vec_test = sample_select(x_test1_trans, test_sel_list, tol, L)

		print(x_train1.shape,x_test.shape)

		x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)
		print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_signal_test.shape)

		vec1 = []
		print("LR")
		reg = LinearRegression().fit(x_train, y_train)
		y_predicted_valid = reg.predict(x_valid)
		print("y_predicted_valid",np.max(y_predicted_valid),np.min(y_predicted_valid))
		y_predicted_test = reg.predict(x_test)
		coef1 = reg.coef_
		print(reg.coef_,reg.intercept_)
		coef1a_id = np.argsort(-np.abs(coef1))
		coef1a = coef1[coef1a_id]
		temp1 = np.vstack((coef1a_id,coef1a)).T
		print(temp1[0:20])

		temp1 = score_2a(y_valid, y_predicted_valid)
		vec1.append(temp1)
		print(temp1)
	
		temp1 = score_2a(y_signal_test, y_predicted_test)
		vec1.append(temp1)
		print(temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		dict1['y_valid'], dict1['y_test'] = y_valid, y_signal_test
		dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test

		vec2[type_id2] = dict1
		temp1 = vec1[1]
		if temp1[1][0]>m_corr[1]:
			m_corr = [type_id2,temp1[1][0]]
		if temp1[2]>m_explain[1]:
			m_explain = [type_id2,temp1[2]]

	print(m_corr,m_explain)
	np.save('feature_transform_%d_%d_%d'%(t_list[0],feature_dim,run_id),vec2,allow_pickle=True)

	print(vec2)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1,dict1

# context feature
def kmer_compare_single2a2_5(species_vec1, train_chromvec, test_chromvec, feature_idx, feature_idx1, type_id, feature_dim_transform, t_list, normalize, flanking, config, run_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

	path2 = '/volume01/yy3/seq_data/dl/replication_timing'

	filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom_ori, start_ori, stop_ori, serial_ori = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])

	filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	# filename1 = '%s/estimate_rt/estimate_rt_%s.txt'%(path2,species_name)
	# filename2a = 'test_seq_%s.1.txt'%(species_name)
	file1 = pd.read_csv(filename1,sep='\t')
	
	col1, col2, col3 = '%s.chrom'%(species_name), '%s.start'%(species_name), '%s.stop'%(species_name)
	chrom, start, stop, serial = np.asarray(file1[col1]), np.asarray(file1[col2]), np.asarray(file1[col3]), np.asarray(file1['serial'])
	label = np.asarray(file1['label'])
	group_label = np.asarray(file1['group_label'])
	signal = np.asarray(file1['signal'])
	print(signal.shape)
	print(feature_idx)

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

	# filename1 = '%s/Datavecs/datavecs_GM_%s.1.npy'%(path2,species_name)
	# t_signal_word = np.load(filename1)
	# n_dim = int(t_signal_word.shape[1]*0.5)
	# t_signal_word = t_signal_word[:,0:n_dim]
	# print(t_signal_word)

	# phyloP_score = read_phyloP(species_name)
	# print(gc_signal.shape,phyloP_score.shape)

	train_sel_idx, test_sel_idx = [], []
	train_sel_list, test_sel_list = [], []
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
		t_signal_ori1 = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal1 = t_signal_ori1[trans_id1]
		feature_dim_kmer1 = t_signal_ori1.shape[1]

		temp1 = serial[id1]
		n1 = len(temp1)
		temp2 = np.vstack(([int(chrom_id)]*n1,temp1)).T
		train_sel_list.extend(temp2)	# chrom_id, serial

		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal2 = t_signal_ori2[trans_id1]
		feature_dim_kmer2 = t_signal_ori2.shape[1]
		t_signal = np.hstack((t_signal1,t_signal2))
		feature_dim_kmer = t_signal.shape[1]
		print(feature_dim_kmer1,feature_dim_kmer2,feature_dim_kmer)

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,serial[id1])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id1]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id1,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id1,feature_idx1]
			t_phyloP = phyloP_score[trans_id1]
			t_phyloP = t_phyloP[:,feature_idx1]

		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]
		
		print(t_gc.shape,t_phyloP.shape,t_signal.shape)
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id1", chrom_id1, trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		# filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori1 = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal_ori1 = np.asarray(file2)
		t_signal1 = t_signal_ori1[trans_id2]
		feature_dim_kmer1 = t_signal_ori1.shape[1]

		temp2 = serial[id2]
		n2 = len(temp2)
		temp2 = np.vstack(([int(chrom_id)]*n2,temp2)).T
		test_sel_list.extend(temp2)

		# filename2 = '%s/training_mtx/Kmer7/training2_kmer_%s.npy'%(path2,chrom_id)
		filename2 = '%s/training_mtx/Kmer5/training2_kmer_%s.npy'%(path2,chrom_id)
		# filename2 = '%s/training_mtx/Kmer4/training2_kmer_%s.npy'%(path2,chrom_id)

		file2 = np.load(filename2)
		t_signal_ori2 = np.asarray(file2)
		# trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal2 = t_signal_ori2[trans_id2]
		feature_dim_kmer2 = t_signal_ori2.shape[1]
		t_signal = np.hstack((t_signal1,t_signal2))

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id2 = mapping_Idx(t_serial,serial[id2])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id2]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id2,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id2,feature_idx1]
			t_phyloP = phyloP_score[trans_id2]
			t_phyloP = t_phyloP[:,feature_idx1]
		
		trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		t_gc = gc_signal[trans_id2a]
		t_gc = t_gc[:,feature_idx]
		
		t_signal = np.hstack((t_gc,t_phyloP,t_signal))
		print("trans_id2", chrom_id, trans_id2.shape, t_signal.shape)

		x_test.extend(t_signal)

	print(len(train_sel_idx),len(test_sel_idx))
		
	# x_train1_ori, x_test_ori = np.asarray(x_train1), np.asarray(x_test)
	x_train1_ori, x_test_ori = np.asarray(x_train1), np.asarray(x_test)
	# x_train1 = x_train1[:,feature_idx]
	# x_test = x_test[:,feature_idx]
	# print(x_train1.shape,x_test.shape)

	if type_id==1 or type_id==2:
		y_signal_train1 = np.hstack((signal[train_sel_idx],t_signal))
	else:
		y_signal_train1 = signal[train_sel_idx]

	y_signal_test = signal[test_sel_idx]
	# print(x_train1.shape,y_signal_train1.shape)
	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	feature_dim = feature_dim_transform
	sub_sample_ratio = 1
	shuffle = 0
	vec2 = dict()
	m_corr, m_explain = [0,0], [0,0]
	# config = {'n_epochs':n_epochs,'feature_dim':feature_dim,'output_dim':output_dim,'fc1_output_dim':fc1_output_dim}
	train_sel_list, test_sel_list = np.asarray(train_sel_list), np.asarray(test_sel_list)
	tol = 5
	L = flanking
	print(train_sel_list[0:10])
	print(test_sel_list[0:10])
	for type_id2 in t_list:
		x_train1_trans, x_test1_trans = feature_transform(x_train1_ori, x_test_ori, feature_dim_kmer, feature_dim, shuffle, 
												sub_sample_ratio, type_id2, normalize)

		x_train1, vec_train = sample_select1(x_train1_trans, train_sel_list, tol, L)
		x_test, vec_test = sample_select1(x_test1_trans, test_sel_list, tol, L)
		# x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)
		# print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_signal_test.shape)

		# vec1 = []
		# print("LR")
		# reg = LinearRegression().fit(x_train, y_train)
		# y_predicted_valid = reg.predict(x_valid)
		# y_predicted_test = reg.predict(x_test)
		# print(reg.coef_,reg.intercept_)
		x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)
		print(x_train.shape,x_valid.shape)

		context_size = x_train.shape[1]
		# config = dict(fc1_output_dim=5,fc2_output_dim=0,n_epochs=10)
		# config = dict(feature_dim=x_train.shape[-1],output_dim=32,fc1_output_dim=0,fc2_output_dim=0,n_epochs=100,batch_size=128)
		config['feature_dim'] = x_train.shape[-1]
		BATCH_SIZE = config['batch_size']
		n_epochs = config['n_epochs']
		earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
		MODEL_PATH = 'test_%d'%(run_id)
		checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
		# roc_cb = roc_callback([x_test,t_test],y_test)
		# model = get_model2(x_mtx.shape[-2])
		model = get_model2a(context_size,config)
		# model.fit(X_train,y_train,epochs = 100,batch_size = BATCH_SIZE,validation_data = [X_test,y_test],class_weight=build_classweight(y_train),callbacks=[earlystop,checkpointer,roc_cb])
		# model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_test,y_test],callbacks=[earlystop,checkpointer])
		model.fit(x_train,y_train,epochs = n_epochs,batch_size = BATCH_SIZE,validation_data = [x_valid,y_valid],callbacks=[earlystop,checkpointer])
		# model.load_weights(MODEL_PATH)

		vec1 = []
		y_predicted_train = model.predict(x_train)
		y_predicted_valid = model.predict(x_valid)
		y_predicted_test = model.predict(x_test)
		y_predicted_train, y_predicted_valid, y_predicted_test = np.ravel(y_predicted_train), np.ravel(y_predicted_valid), np.ravel(y_predicted_test)

		temp1 = score_2a(y_valid, y_predicted_valid)
		vec1.append(temp1)
		print(temp1)
	
		temp1 = score_2a(y_signal_test, y_predicted_test)
		vec1.append(temp1)
		print(temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		dict1['y_valid'], dict1['y_test'] = y_valid, y_signal_test
		dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test

		vec2[type_id2] = dict1
		temp1 = vec1[1]
		if temp1[1][0]>m_corr[1]:
			m_corr = [type_id2,temp1[1][0]]
		if temp1[2]>m_explain[1]:
			m_explain = [type_id2,temp1[2]]

	print(m_corr,m_explain)
	np.save('feature_transform_%d_%d_%d'%(t_list[0],feature_dim,run_id),vec2,allow_pickle=True)
	print(vec2)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1,dict1

def kmer_compare_single2a2_3(species_vec1, train_chromvec, test_chromvec, feature_idx, feature_idx1, 
								type_id, feature_dim_transform, t_list, normalize, run_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

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

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

	filename1 = '%s/Datavecs/datavecs_GM_%s.1.npy'%(path2,species_name)
	t_signal_word = np.load(filename1)
	n_dim = int(t_signal_word.shape[1]*0.5)
	t_signal_word = t_signal_word[:,0:n_dim]
	print(t_signal_word)

	# phyloP_score = read_phyloP(species_name)
	# print(gc_signal.shape,phyloP_score.shape)

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
		t_signal_ori = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		t_signal = t_signal_ori[trans_id1]
		feature_dim_kmer = t_signal_ori.shape[1]

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,serial[id1])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id1]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id1,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id1,feature_idx1]
			t_phyloP = phyloP_score[trans_id1]
			t_phyloP = t_phyloP[:,feature_idx1]

		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]

		t_word = t_signal_word[trans_id1a]
		
		print(t_gc.shape,t_phyloP.shape,t_word.shape,t_signal.shape)
		t_signal = np.hstack((t_gc,t_phyloP,t_word,t_signal))
		print("trans_id1", chrom_id1, trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		file2 = np.load(filename2)
		t_signal_ori = np.asarray(file2)
		trans_id2 = mapping_Idx(serial_ori[id2_ori],serial[id2])	# mapped index
		t_signal = t_signal_ori[trans_id2]

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id2 = mapping_Idx(t_serial,serial[id2])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id2]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id2,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id2,feature_idx1]
			t_phyloP = phyloP_score[trans_id2]
			t_phyloP = t_phyloP[:,feature_idx1]
		
		trans_id2a = mapping_Idx(serial_ori,serial[id2])	# mapped index
		t_gc = gc_signal[trans_id2a]
		t_gc = t_gc[:,feature_idx]
		
		t_word = t_signal_word[trans_id2a]

		t_signal = np.hstack((t_gc,t_phyloP,t_word,t_signal))
		print("trans_id2", chrom_id, trans_id2.shape, t_signal.shape)

		x_test.extend(t_signal)

	print(len(train_sel_idx),len(test_sel_idx))
	# type_id = 1 or type_id = 2: add new species
	# if type_id==1 or type_id==2:
	# 	train_sel_idx1 = map_idx[train_sel_idx]
	# 	test_sel_idx1 = map_idx[test_sel_idx]
	# 	t_signal = []
	# 	num1 = len(species_vec1)

	# 	# train in multiple species
	# 	for i in range(1,num1):
	# 		species_name = species_vec1[i]
	# 		filename1 = '%s/training2_kmer_%s.npy'%(path2,species_name)
	# 		data1 = np.load(filename1)
	# 		data1_sub = data1[train_sel_idx1]
	# 		# data_vec.append(data1_sub)
	# 		x_train1.extend(np.asarray(data1_sub))

	# 		filename1 = '%s/estimate_rt/estimate_rt_%s.sel.txt'%(path2,species_name)
	# 		# filename2a = 'test_seq_%s.1.txt'%(species_name)
	# 		file1 = pd.read_csv(filename1,sep='\t')
	# 		signal = np.asarray(file1['signal'])
	# 		t_signal.extend(signal[train_sel_idx])

	# 	t_signal = np.asarray(t_signal)
		
	x_train1_ori, x_test_ori = np.asarray(x_train1), np.asarray(x_test)
	# x_train1 = x_train1[:,feature_idx]
	# x_test = x_test[:,feature_idx]
	# print(x_train1.shape,x_test.shape)

	if type_id==1 or type_id==2:
		y_signal_train1 = np.hstack((signal[train_sel_idx],t_signal))
	else:
		y_signal_train1 = signal[train_sel_idx]

	y_signal_test = signal[test_sel_idx]
	# print(x_train1.shape,y_signal_train1.shape)
	# normalize the signals
	y_signal_train1 = signal_normalize(y_signal_train1,[0,1])
	y_signal_test = signal_normalize(y_signal_test,[0,1])

	feature_dim = feature_dim_transform
	sub_sample_ratio = 1
	shuffle = 0
	vec2 = dict()
	m_corr, m_explain = [0,0], [0,0]
	for type_id2 in t_list:
		x_train1, x_test = feature_transform(x_train1_ori, x_test_ori, feature_dim_kmer, feature_dim, shuffle, 
												sub_sample_ratio, type_id2, normalize)

		x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_signal_train1, test_size=0.1, random_state=42)
		print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_signal_test.shape)

		vec1 = []
		print("LR")
		reg = LinearRegression().fit(x_train, y_train)
		y_predicted_valid = reg.predict(x_valid)
		y_predicted_test = reg.predict(x_test)
		print(reg.coef_,reg.intercept_)

		temp1 = score_2a(y_valid, y_predicted_valid)
		vec1.append(temp1)
		print(temp1)
	
		temp1 = score_2a(y_signal_test, y_predicted_test)
		vec1.append(temp1)
		print(temp1)

		dict1 = dict()
		dict1['vec1'] = vec1
		dict1['y_valid'], dict1['y_test'] = y_valid, y_signal_test
		dict1['y_predicted_valid'], dict1['y_predicted_test'] = y_predicted_valid, y_predicted_test

		vec2[type_id2] = dict1
		temp1 = vec1[1]
		if temp1[1][0]>m_corr[1]:
			m_corr = [type_id2,temp1[1][0]]
		if temp1[2]>m_explain[1]:
			m_explain = [type_id2,temp1[2]]

	print(m_corr,m_explain)
	np.save('feature_transform_%d_%d_%d'%(t_list[0],feature_dim,run_id),vec2,allow_pickle=True)

	# print('SVR')
	# clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
	# clf.fit(x_train, y_train)
	# y_predicted_valid1 = reg.predict(x_valid)
	# y_predicted_test1 = reg.predict(x_test)

	# temp1 = score_2a(y_valid, y_predicted_valid1)
	# vec1.append(temp1)
	# print(temp1)

	# temp1 = score_2a(y_signal_test, y_predicted_test1)
	# vec1.append(temp1)
	# print(temp1)

	# dict1['y_predicted_valid1'], dict1['y_predicted_test1'] = y_predicted_valid1, y_predicted_test1

	# print("RFLR")
	# # RandomForestRegressor
	# regr = RandomForestRegressor(max_depth=5, random_state=0,
	# 						n_estimators=200)
	# regr.fit(x_train, y_train)
	# print(regr.feature_importances_)
	# np.save('regr_featureImp',regr.feature_importances_)
	# y_predicted_valid = reg.predict(x_valid)
	# y_predicted_test = reg.predict(x_test)
	# print(reg.coef_,reg.intercept_)

	# temp1 = score_2a(y_valid, y_predicted_valid)
	# vec1.append(temp1)
	# print(temp1)
	
	# temp1 = score_2a(y_signal_test, y_predicted_test)
	# vec1.append(temp1)
	# print(temp1)

	# print("Regressor")
	# xgb_model = xgboost.XGBRegressor(colsample_bytree=0.5,
	# 			 gamma=0,                 
	# 			 learning_rate=0.07,
	# 			 max_depth=10,
	# 			 min_child_weight=1.5,
	# 			 n_estimators=1000,                                                                    
	# 			 reg_alpha=0.75,
	# 			 reg_lambda=0.45,
	# 			 objective='reg:squarederror',
	# 			 n_jobs=50,
	# 			 subsample=0.6,
	# 			 seed=42) 

	# # objective = {'reg:linear': [], 'reg:gamma': [], 'reg:linear - base_score': [], 'reg:gamma - base_score': []}
	# xgb_model.fit(x_train, y_train)
	# y_predicted_valid = xgb_model.predict(x_valid)
	# y_predicted_test = xgb_model.predict(x_test)

	# print("train",train_chromvec)
	# score1 = mean_squared_error(y_valid, y_predicted_valid)
	# score2 = pearsonr(y_valid,y_predicted_valid)
	# vec1.append([score1,score2])
	# print(score1,score2)

	# print("test",test_chromvec)
	# score1 = mean_squared_error(y_signal_test, y_predicted_test)
	# score2 = pearsonr(y_signal_test,y_predicted_test)
	# vec1.append([score1,score2])
	# print(score1,score2)
	print(vec2)

	# y_proba = data1['yprob']
	# print(y_test.shape,y_proba.shape)
	# corr1 = pearsonr(y_test, np.ravel(y_proba))
	# print(corr1)

	return vec1,dict1

def kmer_compare_single2a2_3a(species_vec1, train_chromvec, test_chromvec, feature_idx, feature_idx1, 
								type_id, feature_dim_transform, t_list, normalize, run_id):

	species_name = species_vec1[0]
	# data1_sub, map_idx = load_kmer_single(species_name)		# map_idx: subset of the indices
	# data_vec.append(data1_sub)
	serial1, serial2, map_idx = load_map_idx(species_name)		# map_idx: subset of the indices
	print("map_idx",map_idx.shape)

	# data_vec = []
	# for i in range(1,num1):
	# 	species_name = species_vec1[i]
	# 	data1_sub = load_kmer_single(species_name)
	# 	data_vec.append(data1_sub)

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

	filename2 = '%s/training2_gc_%s.txt'%(path2,species_name)
	file2 = pd.read_csv(filename2,sep='\t')
	gc_signal = np.asarray(file2)

	filename1 = '%s/Datavecs/datavecs_GM_%s.1.npy'%(path2,species_name)
	t_signal_word = np.load(filename1)
	n_dim = int(t_signal_word.shape[1]*0.5)
	t_signal_word = t_signal_word[:,0:n_dim]
	print(t_signal_word)

	# phyloP_score = read_phyloP(species_name)
	# print(gc_signal.shape,phyloP_score.shape)

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
		# filename2 = '%s/training_mtx/training2_kmer_%s.npy'%(path2,chrom_id)
		# file2 = np.load(filename2)
		# t_signal_ori = np.asarray(file2)
		trans_id1 = mapping_Idx(serial_ori[id1_ori],serial[id1])	# mapped index
		# t_signal = t_signal_ori[trans_id1]
		# feature_dim_kmer = t_signal_ori.shape[1]

		filename3 = 'phyloP_chr%s.txt'%(chrom_id)
		temp1 = pd.read_csv(filename3,sep='\t')
		temp1 = np.asarray(temp1)
		t_serial, phyloP_score = np.int64(temp1[:,0]), temp1[:,1:]
		trans_id1 = mapping_Idx(t_serial,serial[id1])	# mapped index
		if feature_idx1==-5:
			t_phyloP = phyloP_score[trans_id1]
		elif feature_idx1==-6:
			t_phyloP = phyloP_score[trans_id1,0:-4]
		else:
			# t_phyloP = phyloP_score[trans_id1,feature_idx1]
			t_phyloP = phyloP_score[trans_id1]
			t_phyloP = t_phyloP[:,feature_idx1]

		trans_id1a = mapping_Idx(serial_ori,serial[id1])	# mapped index
		t_gc = gc_signal[trans_id1a]
		t_gc = t_gc[:,feature_idx]

		t_word = t_signal_word[trans_id1a]
		
		# print(t_gc.shape,t_phyloP.shape,t_word.shape,t_signal.shape)
		# t_signal = np.hstack((t_gc,t_phyloP,t_word,t_signal))
		t_signal = np.hstack((t_gc,t_phyloP,t_word))
		print("trans_id1", chrom_id1, trans_id1.shape, t_signal.shape)

		x_train1.extend(t_signal)

	# test in one species
	for chrom_id in test_chromvec:
		chrom_id1 = 'chr%s'%(chrom_id)
		id2 = np.where(chrom==chrom_id1)[0]
		test_sel_idx.extend(id2)
		id2_ori = np.where(chrom_ori==chrom_id1)[0]
		
