
from __future__ import print_function

import string
import sys
import os
from collections import deque

import pandas as pd
import numpy as np
import processSeq

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import tensorflow as tf
import keras
keras.backend.image_data_format()
from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint

import sklearn as sk
from sklearn.base import BaseEstimator, _pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats
import xgboost
import pickle

import time
from timeit import default_timer as timer

import os.path
from optparse import OptionParser

import utility_1
from utility_1 import mapping_Idx, sample_select2a, read_predict, read_predict_weighted
import base_variant_1 as base_variant
from base_variant_1 import _Base1
from base_variant_1 import generate_sequences, sample_select2a1, score_2a
import h5py

class ConvergenceMonitor(object):
	"""Monitors and reports convergence
	"""
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
		"""Reports convergence
		"""
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

class RepliSeq(_Base1):

	def __init__(self, file_path, species_id, resolution, run_id, generate, 
					chromvec,test_chromvec,
					featureid,type_id,cell,method,ftype,ftrans,tlist,
					flanking,normalize,
					config,
					attention=1,feature_dim_motif=1,
					kmer_size=[6,5]):
		_Base1.__init__(self, 
					file_path=file_path, species_id=species_id, resolution=resolution, 
					run_id=run_id, generate=generate, 
					chromvec=chromvec, test_chromvec=test_chromvec,
					featureid=featureid,type_id=type_id,
					cell=cell,method=method,
					ftype=ftype,ftrans=ftrans,tlist=tlist,
					flanking=flanking,normalize=normalize,
					config=config,
					attention=attention, feature_dim_motif=feature_dim_motif,
					kmer_size=kmer_size)
		
		print('training chromvec',chromvec)
		print('test chromvec',test_chromvec)

	def average_prediction(self):
		x = 1

	def prepare_1(self,x_train1_trans,train_sel_list,idx_train):
		
		tol = self.tol
		L = self.flanking
		run_id = self.run_id
			
		seq_list = generate_sequences(train_sel_list[idx_train],region_list=self.region_boundary)
		x_train, y_train, vec_train, vec_train_local = sample_select2a1(x_train1_trans[idx_train], y_signal_train1[idx_train], 
																	train_sel_list[idx_train], seq_list, tol, L)
		
		return x_train, y_train, vec_train, vec_train_local

	# control function
	def control_1(self,path1,file_prefix,est_attention=1):

		vec2 = dict()
		for type_id2 in self.t_list:
			print("feature transform")
			start = time.time()
			feature_dim_transform = self.feature_dim_transform
			if self.load_type==0:
				self.prep_data(path1,file_prefix,type_id2,feature_dim_transform)
			else:
				pre_config = self.config['pre_config']
				print('prep_data_sequence_2')
				self.prep_data_sequence_2(pre_config)

			stop = time.time()
			print('prepare',stop-start)
			self.type_id2 = type_id2
			vec2 = dict()
			# dict1:'train':{'attention'}; 'valid':{'pred','score','attention'};'test1':{'pred','score','attention'}
			
			x_train, x_valid = self.x['train'], self.x['valid']
			y_train, y_valid = self.y['train'], self.y['valid']
			train_vec1, valid_vec1 = (x_train,y_train), (x_valid,y_valid)

			dict1 = self.kmer_compare_weighted3_1(train_vec1,valid_vec1,est_attention=est_attention)
			vec2[type_id2] = dict1

		# print(m_corr,m_explain,aver_score1)
		train_id, valid_id, test_id = self.idx_list['train'], self.idx_list['valid'], self.idx_list['test']
		print('test_sel_list',len(self.train_sel_list[test_id]))
		vec2.update({'train':self.train_sel_list[train_id],'valid':self.train_sel_list[valid_id],'test':self.train_sel_list[test_id]})
		filename1 = '%s/feature_transform_%d_%d_%d.npy'%(self.path,self.t_list[0],feature_dim_transform[0],self.run_id)

		# output score vector
		self.output_vec3_1(vec2,self.t_list)

		if 'pred_filename2' in self.config:
			output_filename = self.config['pred_filename2']
		else:
			# output_filename = '%s/feature_transform_%d_%d.txt'%(self.path,self.run_id,self.method)
			output_filename = '%s/feature_transform_%d_%d.1.txt'%(self.path,self.run_id,self.method)
		# output_filename = '%s/feature_transform_%d.txt'%(self.path,self.run_id)

		if self.est_attention_type1==1:
			data1 = self.test_result_3(filename1,output_filename,vec2)
		else:
			data1 = self.test_result_3_1(filename1,output_filename,vec2)
		
		data2 = self.attention_test_1(data1)
		vec2.update({'test_value1':data2})
		# np.save(filename1,vec2,allow_pickle=True)
		
		return True

	# control function
	# predict the estimated feature importance
	def control_2(self,path1,file_prefix):

		vec2 = dict()
		for type_id2 in self.t_list:
			print("feature transform")
			start = time.time()
			feature_dim_transform = self.feature_dim_transform
			self.prep_data(path1,file_prefix,type_id2,feature_dim_transform)
			self.prep_test_data()
			stop = time.time()
			print('prepare',stop-start)
			self.type_id2 = type_id2
			dict1 = self.kmer_compare_weighted3_test_mode()
			vec2[type_id2] = dict1

		# print(m_corr,m_explain,aver_score1)
		train_id, valid_id, test_id = self.idx_list['train'], self.idx_list['valid'], self.idx_list['test']
		print('test_sel_list',len(self.train_sel_list[test_id]))
		vec2.update({'train':self.train_sel_list[train_id],'valid':self.train_sel_list[valid_id],'test':self.train_sel_list[test_id]})
		filename1 = '%s/feature_transform_%d_%d_%d_%d.test.npy'%(self.path,self.t_list[0],
													feature_dim_transform[0],self.run_id,self.method)
		# np.save(filename1,vec2,allow_pickle=True)

		# output score vector
		self.output_vec3_1(vec2,self.t_list)
		output_filename = '%s/feature_transform_%d_%d.txt'%(self.path,self.run_id,self.method)
		# output_filename = '%s/feature_transform_%d.txt'%(self.path,self.run_id)
		temp1 = [22,32,52,51,55]
		data1 = self.test_result_3(filename1,output_filename,vec2)

		if self.method in temp1:
			data2 = self.attention_test_1(data1)
			vec2.update({'test_value1':data2})

		np.save(filename1,vec2,allow_pickle=True)

		return True

	# control function
	# training with sequences
	def control_3(self,path1,file_prefix,select_config):

		vec2 = dict()

		if self.config['encoding_pre']==1:
			self.prep_data_sequence_ori()

		t_list = [0]
		for type_id2 in t_list:
			# np.save(filename1)
			print("feature transform")
			start = time.time()
			feature_dim_transform = self.feature_dim_transform
			# self.prep_data(path1,file_prefix,type_id2,feature_dim_transform)
			self.prep_data_sequence_1(select_config)
			# self.prep_test_data()
			stop = time.time()
			print('prepare',stop-start)
			
			# return
			self.type_id2 = type_id2
			# dict1 = self.kmer_compare_weighted3_1()
			dict1 = self.kmer_compare_weighted3_2()
			vec2[type_id2] = dict1
			
		# print(m_corr,m_explain,aver_score1)
		train_id, valid_id, test_id = self.idx_list['train'], self.idx_list['valid'], self.idx_list['test']
		print('test_sel_list',len(self.train_sel_list[test_id]))
		vec2.update({'train':self.train_sel_list[train_id],'valid':self.train_sel_list[valid_id],'test':self.train_sel_list[test_id]})
		filename1 = '%s/feature_transform_%d.npy'%(self.path,self.run_id)
		
		# np.save(filename1,vec2,allow_pickle=True)

		# output score vector
		self.output_vec3_1(vec2,self.t_list)
		output_filename = '%s/feature_transform_%d_%d.txt'%(self.path,self.run_id,self.method)
		data1 = self.test_result_3(filename1,output_filename,vec2)
		data2 = self.attention_test_1(data1)

		vec2.update({'test_value1':data2})
		np.save(filename1,vec2,allow_pickle=True)
		
		return True

	def control_3_test(self,path1,file_prefix):

		config = self.config.copy()
		units1=[50,50,50,25,50,25,0,0]
		config['feature_dim_vec'] = units1[2:]
		units1=[50,50,50,25,50,0,0,0]
		config['feature_dim_vec_basic'] = units1[2:]
		print('get_model2a1_attention_1_2_2_sample2')
		flanking = 50
		context_size = 2*flanking+1
		n_step_local_ori = 5000
		region_unit_size = 1
		feature_dim = 4

		local_conv_list1 = []
		regularizer2, bnorm, activation = 1e-05, 1, 'relu'
		n_filters, kernel_size1, stride, dilation_rate1, pool_length1, stride1, drop_out_rate, boundary = 32, 10, 5, 1, 1, 1, 0.2, 1
		conv_1 = [n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate]
		local_conv_list1.append(conv_1)

		n_filters, kernel_size1, stride, dilation_rate1, pool_length1, stride1, drop_out_rate, boundary = 32, 5, 1, 1, 5, 5, 0.2, 1
		conv_1 = [n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate]
		local_conv_list1.append(conv_1)

		n_filters, kernel_size1, stride, dilation_rate1, pool_length1, stride1, drop_out_rate, boundary = 16, 5, 1, 1, 5, 5, 0.2, 1
		conv_1 = [n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate]
		local_conv_list1.append(conv_1)
		config['local_conv_list1'] = local_conv_list1
		print(local_conv_list1)

		feature_dim1, feature_dim2, return_sequences_flag1, sample_local, pooling_local = 16, 50, True, 0, 1
		n_step_local1 = 15
		local_vec_1 = [feature_dim1, feature_dim2, return_sequences_flag1, sample_local, pooling_local]
		attention2_local = 0
		config.update({'feature_dim':feature_dim})
		config.update({'attention1':0,'attention2':1,'select2':0,'context_size':context_size,'n_step_local':n_step_local1,'n_step_local_ori':n_step_local_ori})
		config.update({'local_vec_1':local_vec_1,'attention2_local':attention2_local})

		model = utility_1.get_model2a1_attention_1_2_2_sample5(config)

		run_id = 100
		type_id2 = 2
		MODEL_PATH = 'test1.h5'
		n_epochs = 1
		BATCH_SIZE = 2
		n_step_local = n_step_local_ori

		file_path1 = './data1/H1-hESC'
		filename1 = '%s/H1-hESC_label_ID.txt'%(file_path1)
		label_ID = pd.read_csv(filename1,sep='\t')
		label_ID = np.asarray(label_ID)
		print(label_ID.shape)
		filename1 = '%s/H1-hESC_label.h5'%(file_path1)
		with h5py.File(filename1,'r') as fid:
			y_signal = fid["vec"][:]
		print(y_signal.shape)
		train_id = np.arange(128)
		valid_id = np.arange(128,160)
		train_num, valid_num = len(train_id), len(valid_id)

		x_train1 = []
		y_train1 = []
		interval = 200
		for i in range(10):
			label_id,label_serial,t_filename,local_id = label_ID[i*interval]
			t_filename1 = '%s/%s'%(file_path1,t_filename)
			with h5py.File(t_filename1,'r') as fid:
				t_mtx = fid["vec"][:]
				x_train1.extend(t_mtx)
				y_train1.extend(y_signal[(i*interval):(i*interval+t_mtx.shape[0])])

		x_train1, y_train1 = np.asarray(x_train1), np.asarray(y_train1)
		print(x_train1.shape,y_train1.shape)
		n_epochs = 10

		x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.2, random_state=42)
		train_num = x_train.shape[0]
		print('x_train, y_train', x_train.shape, y_train.shape)
		print('x_valid, y_valid', x_valid.shape, y_valid.shape)

		model = utility_1.get_model2a1_attention_1_2_2_sample5(config)
		file_path1 = './data1/H1-hESC'
		train_id = range(train_num)

		label_ID = np.arange(len(y_train))
		training_generator = DataGenerator2(label_ID, y_train, train_data=x_train, file_path=file_path1, 
												batch_size=BATCH_SIZE, 
												dim=(context_size,n_step_local,4), 
												shuffle=True)

		earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
		
		model.fit(x=training_generator,epochs = n_epochs, validation_data = [x_valid,y_valid],
							callbacks=[earlystop,checkpointer],
							max_queue_size = 1000,
		 					workers=20, use_multiprocessing=True)

		return model

	def control_3_test1(self,path1,file_prefix):

		config = self.config.copy()
		units1=[50,50,50,25,50,25,0,0]
		config['feature_dim_vec'] = units1[2:]
		units1=[50,50,50,25,50,0,0,0]
		config['feature_dim_vec_basic'] = units1[2:]
		print('get_model2a1_attention_1_2_2_sample2')
		flanking = 50
		context_size = 2*flanking+1
		n_step_local_ori = 5000
		region_unit_size = 1
		feature_dim = 4

		local_conv_list1 = []
		regularizer2, bnorm, activation = 1e-05, 1, 'relu'
		n_filters, kernel_size1, stride, dilation_rate1, pool_length1, stride1, drop_out_rate, boundary = 64, 10, 5, 1, 1, 1, 0.2, 1
		conv_1 = [n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate]
		local_conv_list1.append(conv_1)

		n_filters, kernel_size1, stride, dilation_rate1, pool_length1, stride1, drop_out_rate, boundary = 32, 5, 1, 1, 5, 5, 0.2, 1
		conv_1 = [n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate]
		local_conv_list1.append(conv_1)

		n_filters, kernel_size1, stride, dilation_rate1, pool_length1, stride1, drop_out_rate, boundary = 16, 5, 1, 1, 5, 5, 0.2, 1
		conv_1 = [n_filters, kernel_size1, stride, regularizer2, dilation_rate1, boundary, bnorm, activation, pool_length1, stride1, drop_out_rate]
		local_conv_list1.append(conv_1)
		config['local_conv_list1'] = local_conv_list1
		print(local_conv_list1)

		feature_dim1, feature_dim2, return_sequences_flag1, sample_local, pooling_local = 16, 25, True, 0, 0
		n_step_local1 = 15
		local_vec_1 = [feature_dim1, feature_dim2, return_sequences_flag1, sample_local, pooling_local]
		attention2_local = 0
		config.update({'feature_dim':feature_dim})
		config.update({'attention1':0,'attention2':1,'select2':0,'context_size':context_size,'n_step_local':n_step_local1,'n_step_local_ori':n_step_local_ori})
		config.update({'local_vec_1':local_vec_1,'attention2_local':attention2_local})

		model = utility_1.get_model2a1_attention_1_2_2_sample5(config)

		run_id = 100
		type_id2 = 2
		MODEL_PATH = 'test%d.h5'%(self.run_id)
		n_epochs = 1
		BATCH_SIZE = 32
		n_step_local = n_step_local_ori

		file_path1 = './data1/H1-hESC'
		filename1 = '%s/H1-hESC_label_ID.txt'%(file_path1)
		label_ID = pd.read_csv(filename1,sep='\t')
		label_ID = np.asarray(label_ID)
		print(label_ID.shape)
		filename1 = '%s/H1-hESC_label.h5'%(file_path1)
		with h5py.File(filename1,'r') as fid:
			y_signal = fid["vec"][:]
		print(y_signal.shape)
		train_id = np.arange(128)
		valid_id = np.arange(128,160)
		train_num, valid_num = len(train_id), len(valid_id)

		earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

		num_sample1 = 20
		select_num = 25
		interval = 200
		select_num1 = select_num*interval
		print(num_sample1,select_num,interval,select_num1)

		x_valid, y_valid = [], []
		start1 = time.time()
		for l in range(select_num):
			t_id1 = num_sample1*select_num1+l*interval
			label_id,label_serial,t_filename,local_id = label_ID[t_id1]
			# id1, t_filename = ID
			t_filename1 = '%s/%s'%(file_path1,t_filename)
			start = time.time()
			with h5py.File(t_filename1,'r') as fid:
				# serial2 = fid["serial"][:]
				t_mtx = fid["vec"][:]
				#print(t_mtx.shape)
				#print(local_id)
				x_valid.extend(t_mtx)
				if local_id==0:
					local_id = 200
				t_id2 = label_id-local_id+1
				num2 = t_mtx.shape[0]
				id1 = np.arange(t_id2,t_id2+num2)
				y_valid.extend(y_signal[id1])

			stop = time.time()
			print(t_id1,label_id,local_id,t_id2,num2,stop-start)

		stop1 = time.time()
		print(stop1-start1)

		x_valid, y_valid = np.asarray(x_valid), np.asarray(y_valid)
		print(x_valid.shape,y_valid.shape)

		start2 = time.time()

		for i1 in range(20):
			for i in range(num_sample1):
				start1 = time.time()

				x_train1 = []
				y_train1 = []
				for l in range(select_num):
					t_id1 = i*select_num1+l*interval
					label_id,label_serial,t_filename,local_id = label_ID[t_id1]
					# id1, t_filename = ID
					t_filename1 = '%s/%s'%(file_path1,t_filename)
					start = time.time()
					with h5py.File(t_filename1,'r') as fid:
						t_mtx = fid["vec"][:]
						x_train1.extend(t_mtx)
						if local_id==0:
							local_id = 200
						t_id2 = label_id-local_id+1
						num2 = t_mtx.shape[0]
						id1 = np.arange(t_id2,t_id2+num2)
						y_train1.extend(y_signal[id1])

					stop = time.time()
					print(t_id1,label_id,local_id,t_id2,num2,stop-start)

				stop1 = time.time()
				print(stop1-start1)

				x_train1, y_train1 = np.asarray(x_train1), np.asarray(y_train1)
				print(x_train1.shape,y_train1.shape)
				n_epochs = 1

				x_train, y_train = x_train1, y_train1
				train_num = x_train.shape[0]
				print('x_train, y_train', x_train.shape, y_train.shape)
				print('x_valid, y_valid', x_valid.shape, y_valid.shape)
				
				model.fit(x_train,y_train,epochs = n_epochs, batch_size = BATCH_SIZE, validation_data = [x_valid,y_valid],
									callbacks=[earlystop,checkpointer])
				model_path2 = '%s/model_%d_%d_%d.h5'%(self.path,run_id,type_id2,context_size)
				model.save(model_path2)

				print('loading weights... ', model_path2)
				model.load_weights(model_path2) # load model with the minimum training error

				y_predicted_valid1 = model.predict(x_valid)
				y_predicted_valid = np.ravel(y_predicted_valid1[:,flanking])
				temp1 = score_2a(np.ravel(y_valid[:,flanking]), y_predicted_valid)
				print([i1,i]+list(temp1))

		stop2 = time.time()
		print(stop2-start2)

		print('loading weights... ', MODEL_PATH)
		model.load_weights(MODEL_PATH) # load model with the minimum training error
		y_predicted_valid1 = model.predict(x_valid)
		y_predicted_valid = np.ravel(y_predicted_valid1[:,flanking])
		temp1 = score_2a(np.ravel(y_valid[:,flanking]), y_predicted_valid)
		print(temp1)

		file_path1 = './data1/H1-hESC'

		label_ID = np.arange(len(y_train))
		training_generator = DataGenerator2(label_ID, y_train, train_data=x_train, file_path=file_path1, 
												batch_size=BATCH_SIZE, 
												dim=(context_size,n_step_local,4), 
												shuffle=True)

		earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
		
		return model

	# control function
	# # cross validation
	def control_5(self,path1,file_prefix,n_fold=5,ratio=0.9,est_attention=1):

		print("feature transform")
		start = time.time()
		feature_dim_transform = self.feature_dim_transform
		type_id2 = 0
		if self.load_type==0:
			self.id_cv = self.prep_data_1(path1,file_prefix,type_id2,feature_dim_transform,
								n_fold=n_fold, ratio=ratio)
		else:
			pre_config = self.config['pre_config']
			print('prep_data_sequence_2')
			self.prep_data_sequence_2(pre_config)
		# self.prep_test_data()
		stop = time.time()
		print('prepare',stop-start)
		# return
		self.type_id2 = type_id2
		vec2 = dict()
		
		self.seq_list = dict()
		self.model_vec = dict()
		num_sample = len(self.train_sel_list)
		self.predicted_signal = np.zeros(num_sample,dtype=np.float32)
		self.predicted_attention = np.zeros(num_sample,dtype=np.float32)
		id_fold = np.zeros(num_sample,dtype=np.int8)
		print(self.id_cv.keys())

		score_vec1 = []
		for i in range(n_fold):
			# dict1: keys: n folds: {'train':[],'valid','test'}
			id_train, id_valid, id_test = self.id_cv[i]['train'], self.id_cv[i]['valid'], self.id_cv[i]['test']
			self.seq_list[i] = dict()
			# self.vec[i], self.vec_local[i] = dict(), dict()
			train_vec, serial_vec = dict(), dict()

			t_vec1 = ['train','valid','test']
			t_signal_vec = dict()

			t_signal_vec['test'] = self.signal[id_test]
			# signal scaling using training and validation chromosomes
			if self.config['train_signal_update1']==1:
				print('rescaling for training data',np.max(self.signal_pre),np.min(self.signal_pre))
				num_train, num_valid = len(id_train), len(id_valid)
				id_train_valid = np.asarray(list(id_train)+list(id_valid))
				id1 = mapping_Idx(self.serial,self.train_sel_list[id_train_valid,1])
				t_chrom, t_signal = self.chrom[id1], self.signal_pre[id1]
				train_chromvec = np.unique(self.train_sel_list[id1,0])
				train_signal, id2, signal_vec2 = self.signal_normalize_chrom(t_chrom,t_signal,train_chromvec,self.scale)
				t_signal_vec['train'] = train_signal[0:num_train]
				t_signal_vec['valid'] = train_signal[num_train:]
				# print(np.max(train_signal),np.min(train_signal),np.max(t_signal_vec['test']),np.min(t_signal_vec['test']))
			else:
				t_signal_vec['train'] = self.signal[id_train]
				t_signal_vec['valid'] = self.signal[id_valid]

			print('train',np.max(t_signal_vec['train']),np.min(t_signal_vec['train']),
					'valid',np.max(t_signal_vec['valid']),np.min(t_signal_vec['valid']),
					'test',np.max(t_signal_vec['test']),np.min(t_signal_vec['test']))

			for t_key in t_vec1:

				idx = self.id_cv[i][t_key]
				idx_sel_list = self.train_sel_list[idx]
				# y_signal = self.signal[idx]
				y_signal = t_signal_vec[t_key]

				start = time.time()
				id1 = generate_sequences(idx_sel_list)
				# self.seq_list[i][t_key] = id1
				print(len(id1))
				stop = time.time()
				print('generate_sequences', stop-start)

				start = time.time()
				# x, y, self.vec[i][t_key], self.vec_local[i][t_key] = sample_select2a1(x_train1_trans[idx],y_signal,
				# 										idx_sel_list, id1, self.tol, self.flanking)
				x, y, vec, vec_local = sample_select2a1(self.x_train1_trans[idx],y_signal,
														idx_sel_list, id1, self.tol, self.flanking)
				train_vec[t_key] = (x,y,vec,vec_local)
				# serial_vec[t_key] = (vec,vec_local)
				print(t_key,x.shape,y.shape,vec.shape,vec_local.shape)

				stop = time.time()
				print('sample_select2a1',stop-start)

			train_vec1, valid_vec1, test_vec1 = train_vec['train'], train_vec['valid'], train_vec['test']

			model, vec2 = self.xxxxxkmer_compare_weighted3_3(train_vec1, valid_vec1, test_vec1, id_test, est_attention=est_attention)
			self.model_vec[i] = model

			model_path2 = '%s/model_%d_%d_test%d.h5'%(self.path,self.run_id,type_id2,i)
			model.save(model_path2)

			self.predicted_signal[id_test] = vec2['test1']['pred']
			self.predicted_attention[id_test] = vec2['test1']['attention']
			id_fold[id_test] = i+1

			valid_score1 = vec2['valid']['score']
			test_score1 = vec2['test1']['score']
			score_vec1.append([i+1]+list(valid_score1))
			score_vec1.append([i+1]+list(test_score1))

		columns = ['serial','signal','predicted_signal','predicted_attention','id_fold']
		serial1 = self.train_sel_list[:,1]
		id1 = mapping_Idx(self.serial,serial1)
		signal1 = self.signal[id1]
		value = [serial1,signal1,self.predicted_signal,self.predicted_attention,id_fold]

		if 'pred_filename2' in self.config:
			output_filename = self.config['pred_filename2']
		else:
			output_filename = '%s/feature_transform_%d_%d.1.txt'%(self.path,self.run_id,self.method)

		self.test_result_3_sub1(id1,columns,value,output_filename,sort_flag=True,sort_column='serial')

		if 'pred_filename1' in self.config:
			filename1 = self.config['pred_filename1']
		else:
			filename1 = 'test_vec2_%d_%d_[%d]_%s.1.txt'%(self.run_id,self.method,self.feature_dim_select1,self.cell)
		
		score_vec1 = np.asarray(score_vec1)
		np.savetxt(filename1,score_vec1,fmt='%.7f',delimiter='\t')

		return True

	# context feature
	def kmer_compare_weighted3_1(self,train_vec1, valid_vec1, est_attention=1):

		x_train, y_train = train_vec1
		x_valid, y_valid = valid_vec1
		print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape)

		# load_type: 0, pre-features, 1, features from pretrained cnn
		load_type = self.load_type
		if 'pre_config' in self.config:
			pre_config = self.config['pre_config']
			self.pre_config = pre_config

		if (self.method<10) or (self.method in self.method_vec[2]):
			print('compare_single_1')
			type_id3 = self.method

			model = self.compare_single_1(x_train,y_train,x_valid,y_valid,type_id3)
			self.model = model
				
			# predict and evaluation
			idx = self.idx_list['test']
			print(idx.shape)
			test_sel_list = self.train_sel_list[idx]

			if load_type==0:
				x_test, y_test = self.x_train1_trans[idx], self.y_signal['test']
			else:
				# x_test, y_test = self.x['test'], self.y_signal['test']
				list1 = self.prep_data_sequence_3(pre_config,self.test_chromvec,model=self.load_model1)
				num1 = len(list1)
				feature_dim = x_train.shape[-1]
				num_sample = len(test_sel_list)
				# x_test = np.zeros((num_sample,feature_dim))
				serial_1, x_test = [], []
				# i1 = 0
				for i in range(num1):
					serial1, feature1 = list1[i]
					serial_1.extend(serial1)
					# x_test[i1:(i1+len(serial1))] = feature1
					# i1 = i1+len(serial1)
					x_test.extend(feature1)

				serial_1, x_test = np.asarray(serial_1), np.asarray(x_test)
				id1 = mapping_Idx(serial_1,test_sel_list[:,1])
				b1 = np.where(id1<0)[0]
				if len(b1)>0:
					print('error! loading test samples',len(b1))
					return
				x_test = x_test[id1]

			print("predict",x_valid.shape,x_test.shape)
			y_predicted_valid = model.predict(x_valid).ravel()
			y_predicted_test = model.predict(x_test).ravel()
			print(y_predicted_valid.shape,y_predicted_test.shape)

			score_vec2 = dict()
			y_test = self.y_signal['test']
			for t_chrom in self.test_chromvec:
				b1 = np.where(test_sel_list[:,0]==int(t_chrom))[0]
				print(t_chrom,len(b1))
				if len(b1)>0:
					t_score1 = score_2a(y_test[b1], y_predicted_test[b1])
					print(t_score1)
					score_vec2[t_chrom] = t_score1
				else:
					print('error!',t_chrom)

			aver_score1 = score_2a(y_test, y_predicted_test)
			score_vec2['aver1'] = aver_score1
			print(aver_score1)

			valid_score1 = score_2a(y_valid, y_predicted_valid)

			predicted_attention_train, predicted_attention_valid, predicted_attention_test = [], [], []
			
		else:
			vec_train, vec_valid = self.vec['train'], self.vec['valid']
			vec_train_local, vec_valid_local = self.vec_local['train'], self.vec_local['valid']

			model = self.training_1_1(x_train,x_valid,y_train,y_valid)
			self.model = model

			print("training data")
			if not(self.method in self.attention_vec):
				estimation = 0
				
			if est_attention==1:
				print("predicting attention...")

			start = time.time()
			predict_type = 0
			if self.config['predict_train_attention']==1:
				temp1, predicted_attention_train = self.predict_1(x_train,vec_train_local,est_attention,predict_type,
																attention_type=self.est_attention_type1,vec_test=vec_train)
			else:
				predicted_attention_train = []

			print("valid data")
			predict_type = 1
			est_attention1 = 0
			if ('predict_valid_attention' in self.config) and (self.config['predict_valid_attention']==1):
				est_attention1 = 1

			y_predicted_valid, predicted_attention_valid = self.predict_1(x_valid,vec_valid_local,est_attention1,predict_type,
															attention_type=self.est_attention_type1,vec_test=vec_valid)
			stop = time.time()
			print('train valid', stop-start)

			print("test")
			start = time.time()
			idx = self.idx_list['test']
			print(idx.shape)
			test_sel_list = self.train_sel_list[idx]

			if load_type==0:
				x_test1 = self.x_train1_trans[idx]
			else:
				x_test1 = []
			
			y_predicted_test, predicted_attention_test, score_vec2 = self.test(x_test1, self.y_signal['test'], 
																		test_sel_list,est_attention,predict_type,load_type)
			stop = time.time()
			print('test',stop-start)

			valid_score1 = score_2a(self.y_signal['valid'], y_predicted_valid)

		vec2 = dict()
		vec2['train'] = {'attention':predicted_attention_train}
		# predicted_attention_valid = np.ravel(predicted_attention_valid[:,self.flanking])
		vec2['valid']= {'pred':y_predicted_valid,'score':valid_score1,'attention':predicted_attention_valid}
		vec2['test1'] = {'pred':y_predicted_test,'score':score_vec2,'attention':predicted_attention_test}

		return vec2

	# context feature
	def kmer_compare_weighted3_2(self):

		x_train, y_train = self.x['train'], self.y['train']
		x_valid, y_valid = self.x['valid'], self.y['valid']

		if (self.method<10) or (self.method in self.method_vec[2]):
			print('compare_single_1')
			type_id3 = self.method

			if self.train==0:
				type_id2 = 0
				context_size = 1
				run_id1 = self.config['runid_load']
				model_path1 = 'model_%d_%d_%d.h5'%(run_id1,type_id2,context_size)
			else:
				model_path1 = ''
			model = self.compare_single_1(x_train,y_train,x_valid,y_valid,type_id=type_id3,model_path1=model_path1)
			self.model = model
				
			# predict and evaluation
			idx = self.idx_list['test']
			print(idx.shape)
			test_sel_list = self.train_sel_list[idx]
			chrom1 = test_sel_list[:,0]
			y_test = self.y_signal['test']
			y_predicted_test = np.zeros_like(y_test)

			score_vec2 = dict()
			for t_chrom in self.test_chromvec:

				b1 = np.where(chrom1==int(t_chrom))[0]

				if self.species_id=='mm10':
					filename1 = '%s_%d_chr%s_encoded1.h5'%(self.species_id,self.cell_type1,t_chrom)
				else:
					filename1 = '%s_chr%s_encoded1.h5'%(self.species_id,t_chrom)

				with h5py.File(filename1,'r') as fid:
					serial1 = fid["serial"][:]
					seq1 = fid["vec"][:]
				
				print(serial1.shape, seq1.shape, len(b1))
				
				serial1 = serial1[:,0]
				id1 = mapping_Idx(serial1,test_sel_list[b1,1])
				print(test_sel_list[b1])

				b2 = np.where(id1<0)[0]
				if len(b2)>0:
					print('error!',t_chrom,len(b2))
					return
					
				x_test = seq1[id1]
				y_predicted_test[b1] = model.predict(x_test).ravel()
				print(t_chrom,len(serial1),len(b1))

				if len(b1)>0:
					t_score1 = score_2a(y_test[b1], y_predicted_test[b1])
					print(t_score1)
					score_vec2[t_chrom] = t_score1
				else:
					print('error!',t_chrom)

			aver_score1 = score_2a(y_test, y_predicted_test)
			score_vec2['aver1'] = aver_score1
			print(aver_score1)

			if self.train==1:
				print("predict",x_valid.shape)
				y_predicted_valid = model.predict(x_valid).ravel()
				print(y_predicted_valid.shape)
				valid_score1 = score_2a(y_valid, y_predicted_valid)
			else:
				y_predicted_valid = []
				valid_score1 = np.zeros(len(aver_score1))

			predicted_attention_train, predicted_attention_valid, predicted_attention_test = [], [], []

		else:
			return

		vec2 = dict()
		vec2['train'] = {'attention':predicted_attention_train}
		# predicted_attention_valid = np.ravel(predicted_attention_valid[:,self.flanking])
		vec2['valid']= {'pred':y_predicted_valid,'score':valid_score1,'attention':predicted_attention_valid}
		vec2['test1'] = {'pred':y_predicted_test,'score':score_vec2,'attention':predicted_attention_test}

		return vec2

	# context feature
	def kmer_compare_weighted3_3(self,train_vec1, valid_vec1, test_vec1, test_idx, est_attention=1):

		x_train, y_train, vec_train, vec_train_local = train_vec1
		x_valid, y_valid, vec_valid, vec_valid_local = valid_vec1
		x_test, y_test, vec_test, vec_test_local = test_vec1
		print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape)
		print(x_test.shape,y_test.shape)

		load_type = self.load_type
		if 'pre_config' in self.config:
			pre_config = self.config['pre_config']
			self.pre_config = pre_config

		if (self.method<10) or (self.method in self.method_vec[2]):
			print('compare_single_1')
			type_id3 = self.method

			model = self.compare_single_1(x_train,y_train,x_valid,y_valid,type_id3)
			self.model = model

			idx = test_idx
			print(idx.shape)
			test_sel_list = self.train_sel_list[idx]

			print("predict",x_valid.shape,x_test.shape)
			y_predicted_valid = model.predict(x_valid).ravel()
			y_predicted_test = model.predict(x_test).ravel()
			print(y_predicted_valid.shape,y_predicted_test.shape)

			score_vec2 = dict()
			test_chromvec = np.unique(test_sel_list[:,0])
			for t_chrom in test_chromvec:
				b1 = np.where(test_sel_list[:,0]==int(t_chrom))[0]
				print(t_chrom,len(b1))
				if len(b1)>0:
					t_score1 = score_2a(y_test[b1], y_predicted_test[b1])
					print(t_score1)
					score_vec2[t_chrom] = t_score1
				else:
					print('error!',t_chrom)

			aver_score1 = score_2a(y_test, y_predicted_test)
			score_vec2['aver1'] = aver_score1
			print(aver_score1)

			valid_score1 = score_2a(y_valid, y_predicted_valid)
			test_score1 = score_2a(y_test, y_predicted_test)

			predicted_attention_train, predicted_attention_valid, predicted_attention_test = [], [], []
			
		else:
			model = self.training_1_1(x_train,x_valid,y_train,y_valid)
			self.model = model

			print("training data")
			
			if not(self.method in self.attention_vec):
				est_attention = 0
				
			if est_attention==1:
				print("predicting attention...")

			start = time.time()
			predict_type = 0
			if self.config['predict_train_attention']==1:
				temp1, predicted_attention_train = self.predict_1(x_train,vec_train_local,est_attention,predict_type,
																attention_type=self.est_attention_type1,vec_test=vec_train)
			else:
				predicted_attention_train = []

			print("valid data")
			predict_type = 1
			est_attention1 = 0
			if ('predict_valid_attention' in self.config) and (self.config['predict_valid_attention']==1):
				est_attention1 = est_attention
			# if self.config['predict_valid_attention']==1:
			# 	est_attention = 1
			y_predicted_valid, predicted_attention_valid = self.predict_1(x_valid,vec_valid_local,est_attention1,predict_type,
															attention_type=self.est_attention_type1,vec_test=vec_valid)

			stop = time.time()
			print('train valid', stop-start)

			print("test")
			start = time.time()
			test_idx = np.asarray(test_idx)
			print(len(test_idx))
			test_sel_list = self.train_sel_list[test_idx]
			
			y_predicted_test, predicted_attention_test = self.predict_1(x_test,vec_test_local,est_attention,predict_type,
															attention_type=self.est_attention_type1,vec_test=vec_test)

			stop = time.time()
			print('test',stop-start)

			y_signal_valid = np.ravel(y_valid[:,self.flanking])
			y_signal_test = np.ravel(y_test[:,self.flanking])
			valid_score1 = score_2a(y_signal_valid, y_predicted_valid)
			test_score1 = score_2a(y_signal_test, y_predicted_test)
			print('valid',valid_score1)
			print('test',test_score1)

		vec2 = dict()
		vec2['train'] = {'attention':predicted_attention_train}
		vec2['valid']= {'pred':y_predicted_valid,'score':valid_score1,'attention':predicted_attention_valid}
		vec2['test1'] = {'pred':y_predicted_test,'score':test_score1,'attention':predicted_attention_test}

		return model, vec2

	# context feature
	def kmer_compare_weighted3_test_mode(self,file_path,file_prefix,feature_dim_transform,model_path1,context_size,type_id2=0):

		print(type_id2)
		print("training...")

		test_id = self.idx_list['test']
		x_test, y_test, test_sel_list = self.x_train1_trans[test_id], self.y_signal['test'], self.train_sel_list[test_id]
		feature_dim = x_test.shape[-1]
		model = self.get_model_basic(self.method,context_size,feature_dim)
		model.load_weights(model_path1)

		print("test")
		y_predicted_test, predicted_attetion_test, score_vec2 = self.test(x_test, y_test, test_sel_list)

		vec2 = dict()
		vec2['test1'] = {'pred':y_predicted_test,'score':score_vec2,'attention':predicted_attention_test}
		vec2['test'] = test_sel_list
		np.save('/mnt/yy3/feature_transform_%d_%d_%d.npy'%(self.t_list[0],feature_dim_transform[0],self.run_id),vec2,allow_pickle=True)

		self.output_vec3(vec2,self.t_list)

		return vec1,dict1

	# pretrain
	# def pretrain(self,x_train,x_valid):

	# 	context_size = x_train.shape[1]
	# 	config = self.config
	# 	config['feature_dim'] = x_train.shape[-1]
	# 	BATCH_SIZE = config['batch_size']
	# 	n_epochs = config['n_epochs']
	# 	config['context_size'] = context_size
	# 	MODEL_PATH = self.model_path

	# 	attention = self.attention
	# 	print(self.predict_context, self.attention)
	# 	model = self.get_model_pre(config,context_size)

	def training_1_1(self,x_train,x_valid,y_train,y_valid,model_path1=""):
		
		run_id = self.run_id
		type_id2 = self.type_id2
		print(x_train.shape,x_valid.shape)
		context_size = x_train.shape[1]
		config = self.config
		config['feature_dim'] = x_train.shape[-1]
		BATCH_SIZE = config['batch_size']
		n_epochs = config['n_epochs']
		config['context_size'] = context_size
		MODEL_PATH = self.model_path

		attention = self.attention
		print(self.predict_context, self.attention)
		model = self.get_model(config,context_size)

		if self.train==1:
			print('x_train, y_train', x_train.shape, y_train.shape)
			earlystop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.step, verbose=0, mode='auto')
			checkpointer = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
			model.fit(x_train,y_train,epochs = n_epochs, batch_size = BATCH_SIZE, validation_data = [x_valid,y_valid],
							callbacks=[earlystop,checkpointer])
			model_path2 = '%s/model_%d_%d_%d.h5'%(self.path,run_id,type_id2,context_size)
			model.save(model_path2)
			model_path2 = MODEL_PATH
			print('loading weights... ', model_path2)
			model.load_weights(model_path2) # load model with the minimum training error
		else:
			if model_path1!="":
				MODEL_PATH = model_path1
			print('loading weights... ', MODEL_PATH,model_path1)
			model.load_weights(MODEL_PATH)

		return model
	
	def predict_evaluate_basic(self,model,x,y_true):
		
		vec2 = dict()
		print("predict",x.shape)
		y_predicted = model.predict(x)
		print('y_predicted', y_predicted.shape)
		print('y_true', y_true.shape)

		y_predicted = np.ravel(y_predicted)
		score1 = score_2a(y_true, y_predicted)
		print(score1)

		return y_predicted, score1

	# define model
	def get_model_basic(self,method_type,context_size,feature_dim):
		if method_type==0:
			model = LinearRegression()
		elif method_type<10:
			print("xgboost regression")
			model = xgboost.XGBRegressor(colsample_bytree=1,
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
		else:
			config = self.config
			config['feature_dim'] = feature_dim
			config['lr'] = self.lr
			config['activation'] = self.activation
			BATCH_SIZE = config['batch_size']
			n_epochs = config['n_epochs']
			config['context_size'] = context_size
			attention = self.attention
			print(self.predict_context, self.attention)
			model = self.get_model(config,context_size)

		return model

	def estimate_attention(self,x_test):
		model = self.model

		if self.method==11:
			layer_name = 'attention1'
			intermediate_layer = Model(inputs=model.input,
								 outputs=model.get_layer(layer_name).output)
			feature1, attention = intermediate_layer.predict(x_test)
			print('attention1',attention.shape)
		# elif self.method==17 or self.method==22 or self.method >=32:
		elif self.method in self.attention_vec:
			layer_name = 'logits_T'
			if self.method >=32:
				layer_name = 'logits_T_1'
			intermediate_layer = Model(inputs=model.input,
									outputs=model.get_layer(layer_name).output)
			attention = intermediate_layer.predict(x_test)
			print('logits_T',attention.shape)
		else:
			attention = []

		return attention

	def estimate_attention_1(self,x_test,layer_name=''):
		model = self.model

		if layer_name=='':
			if self.method in [11]:
				layer_name = 'attention2'
			elif self.method in [17,22]:
				layer_name = 'logits_T'
			elif self.method >=32:
				layer_name = 'logits_T_1'

		intermediate_layer = Model(inputs=model.input,
								 outputs=model.get_layer(layer_name).output)
		feature1, attention = intermediate_layer.predict(x_test)
		print('attention1',attention.shape)

		return attention

	def estimate_attention_2(self,x_test,layer_name=''):
		model = self.model

		if self.method==11:
			layer_name = 'attention1'
			intermediate_layer = Model(inputs=model.input,
								 outputs=model.get_layer(layer_name).output)
			feature1, attention1 = intermediate_layer.predict(x_test)
			print('attention1',attention1.shape)
			attention = []
		elif self.method in self.attention_vec:
			layer_name = 'logits_T'
			layer_name1 = 'attention2'
			if self.method >=32:
				layer_name = 'logits_T_1'
			intermediate_layer = Model(inputs=model.input,
									outputs=model.get_layer(layer_name).output)
			attention = intermediate_layer.predict(x_test)
			print('logits_T',attention.shape)

			intermediate_layer1 = Model(inputs=model.input,
								 outputs=model.get_layer(layer_name1).output)
			feature1, attention1 = intermediate_layer.predict(x_test)
			print(layer_name1,feature1.shape,attention1.shape)
		else:
			attention = []
			attention1 = []

		return attention, attention1

	def predict_1(self,x_test,vec_test_local,est_attention=1,predict_type=1,
						y_predicted_test=[], predicted_attention=[],
						attention_type=1,vec_test=[]):

		start = time.time()
		if (self.method in self.attention_vec) and (est_attention==1):
			print("predicting attention...")
			if attention_type==1:
				predicted_attention = self.estimate_attention(x_test)
				predicted_attention = self.process_attention(predicted_attention)
			else:
				predicted_attention1, predicted_attention2 = self.estimate_attention_2(x_test)
				predicted_attention1 = self.process_attention(predicted_attention1)
				if len(vec_test)==0:
					print('error! predicted_attention2')
					return
				predicted_attention2 = self.process_attention_2(predicted_attention2,vec_test)
				predicted_attention = [predicted_attention1, predicted_attention2]
		else:
			predicted_attention = []

		stop = time.time()
		print('predict attention',stop-start)

		if predict_type==0:
			return [], predicted_attention

		start1 = time.time()
		y_predicted_test = self.model.predict(x_test)
		print(len(y_predicted_test))
		stop1 = time.time()
		start2 = stop1
		print('predict 1',stop1-start1)

		flanking1 = self.flanking1

		flag1 = 0
		if self.predict_context==1:
			flag1 = 1
			if ('predict_context_type' in self.config) and (self.config['predict_context_type']==2):
				flag1 = 0

		if flag1==1:
			y_predicted_test = read_predict(y_predicted_test, vec_test_local, [], flanking1, self.predict_type_id)
		else:
			if y_predicted_test.shape[1]>1:
				y_predicted_test = np.ravel(y_predicted_test[:,self.flanking])
			else:
				y_predicted_test = np.ravel(y_predicted_test)

		stop2 = time.time()
		print('predict 2',stop2-start2)

		return y_predicted_test, predicted_attention

	def test(self,x_test1,y_test1,test_sel_list,est_attention=1,predict_type=1,load_type=0):

		tol = self.tol
		L = self.flanking
		run_id = self.run_id
		
		print("test")
		vec2 = dict()
		print(test_sel_list)
		num_test_sample = len(test_sel_list)
		predicted_test1 = np.zeros(num_test_sample)
		predicted_attention1 = dict()
		flag = False
		print('test: tol',tol)

		flag = 1
		if (self.method<10) or (self.method in self.method_vec[2]):
			flag = 0

		test_chromvec = np.unique(test_sel_list[:,0])
		for t_chrom in test_chromvec:
			b1 = np.where(test_sel_list[:,0]==int(t_chrom))[0]
			print(t_chrom,len(b1))
			if len(b1)==0:
				print("error!",t_chrom)
				return

			y_test_ori = y_test1[b1]
			if flag == 0:
				y_predicted_test = self.model.predict(x_test1)
				predicted_attetion = []
			else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
				idx_sel_list = test_sel_list[b1]
				seq_list = generate_sequences(idx_sel_list,region_list=self.region_boundary)
				self.output_generate_sequences(idx_sel_list,seq_list)

				print(len(seq_list))
				if load_type==0:
					x_test1_local = x_test1[b1]
				else:
					list1 = self.prep_data_sequence_3(self.pre_config,[t_chrom],model=self.model_load1)
					x_test1_local = list1[0][1]
					t_serial = list1[0][0]
					id1 = utility_1.search_Idx(t_serial,idx_sel_list[:,1])
					x_test1_local = x_test1_local[id1]
					print('x_test1_local',x_test1_local.shape)

				x_test, y_test, vec_test, vec_test_local = sample_select2a1(x_test1_local, y_test_ori, idx_sel_list, seq_list, tol, L)

				start = time.time()
				y_predicted_test, predicted_attention = self.predict_1(x_test,vec_test_local,
																est_attention,predict_type,
																attention_type=self.est_attention_type1,
																vec_test=vec_test)
				stop = time.time()
				print('test chrom (prev)',t_chrom,stop-start)
				start = time.time()
				predicted_attention1[t_chrom] = dict()
				predicted_attention1[t_chrom]['value'] = predicted_attention
				if len(predicted_attention)>0:
					if self.est_attention_type1==1:
						print(predicted_attention.shape)
					else:
						print(predicted_attention[0].shape)
					
				predicted_attention1[t_chrom]['serial'] = test_sel_list[b1,1]

				stop = time.time()
				print('test chrom',t_chrom,stop-start)

			if len(y_predicted_test)>0:
				flag = True
				vec1 = score_2a(y_test_ori, y_predicted_test)
				print(vec1)
				vec2[t_chrom] = vec1
				predicted_test1[b1] = y_predicted_test

		if flag==True:
			aver_score1 = score_2a(y_test1, predicted_test1)
			vec2['aver1'] = aver_score1
			print(aver_score1)

		return predicted_test1, predicted_attention1, vec2

	def test_1(self,filename,file_prefix,output_filename,feature_dim_transform):

		type_id2 = 0
		path1 = './'
		filename1 = '%s/%s_%d_%d_%d.npy'%(path1,file_prefix,type_id2,feature_dim_transform[0],feature_dim_transform[1])

		if os.path.exists(filename1)==True:
			print("loading data...")
			data1 = np.load(filename1,allow_pickle=True)
			data_1 = data1[()]
			train_sel_list = data_1['idx']
			print('train_sel_list',train_sel_list.shape)
		else:
			print(filename1)
			print("data not found!")
			return

		# training data and test data
		train_id1 = []
		print(self.train_chromvec,self.test_chromvec)
		train_sel_list, test_sel_list, train_id1, test_id1 = self.training_serial(train_sel_list)

		filename2 = 'dict1_train_serial.npy'
		dict_serial = dict()

		vec_test, vec_test_local = self.sample_select2a_1(test_sel_list, self.tol, self.flanking)
		
		data1 = np.load(filename,allow_pickle=True)
		data1 = data1[()]
		data2 = data1[0]
		print(data2.keys())
		data3 = data2['test1']

		dict1 = dict()
		dict1['pred'] = dict()
		for t_chrom in self.test_chromvec:
			b1 = np.where(test_sel_list[:,0]==int(t_chrom))[0]
			n1 = len(b1)
			print(t_chrom,n1)
			if self.method>=10:
				vec_test1 = vec_test[b1]
				chrom_id = str(t_chrom)
				test_1 = data3[chrom_id]
				attention_test = test_1['predicted_attention']
				n_sample = attention_test.shape[0]
				if (n_sample!=len(b1)):
					print('error! %d %d'%(n_sample,n1))
					break
				n2, n3 = attention_test.shape[1], attention_test.shape[2]
				attention_test1 = np.ravel(attention_test)
				vec_test2 = np.ravel(vec_test1)

				dict1[t_chrom] = dict()
				vec1 = np.zeros((n_sample,6))
				for i in range(0,n_sample):
					id1 = test_sel_list[b1[i],1]
					b2 = np.where(vec_test2==id1)[0]
					attention_test2 = attention_test1[b2]
					dict1['pred'][id1] = attention_test2
					vec1[i] = [len(b2),np.max(attention_test2),np.min(attention_test2),np.median(attention_test2),np.mean(attention_test2),np.quantile(attention_test2,0.90)]
					if i%1000==0:
						print(vec1[i])

				dict1[t_chrom]['vec1'] = vec1
				dict1[t_chrom]['serial1'] = test_sel_list[b1,1]

		np.save(output_filename,dict1,allow_pickle=True)

		return dict1

	def test_1_1(self,filename1,output_filename,thresh1=0.95,type_id=1):
		
		data1 = np.load(filename1,allow_pickle=True)
		data1 = data1[()]
		t_vec1 = np.asarray(list(data1.keys()))
		print(t_vec1)
		t_vec2 = np.setdiff1d(t_vec1,'pred')
		print(t_vec2)
		stats_list1 = []
		serial_list1 = []
		
		thresh_vec1 = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]
		for t1 in t_vec2:
			print(t1)
			vec1 = data1[t1]['vec1']
			print(vec1.shape)
			serial_1 = data1[t1]['serial1']
			value_1 = vec1[:,-1]
			value_2 = vec1[:,2]
			threshold = np.quantile(value_1,thresh1)
			thresh_vec2 = np.quantile(value_1,thresh_vec1)
			print(thresh_vec2)
			if type_id==1:
				b1 = np.where(value_1>threshold)[0]
			else:
				b1 = np.where(value_2>threshold)[0]
			t_serial = serial_1[b1]
			temp1 = vec1[b1]
			stats_list1.extend(temp1[:,[0,1,2,3,5]])
			serial_list1.extend(t_serial)

		id1 = mapping_Idx(self.serial, np.asarray(serial_list1))
		chrom1, start1, stop1 = self.chrom[id1], self.start[id1], self.stop[id1]
		signal1 = self.signal[id1]
		fields = ['chrom','start','stop','serial','signal','num','max','min','meidan','0.9-quantile']

		data_1 = pd.DataFrame(columns=fields)
		data_1['chrom'], data_1['start'], data_1['stop'], data_1['serial'] = chrom1, start1, stop1, serial_list1
		data_1['signal'] = signal1

		stats_list1 = np.asarray(stats_list1)
		for i in range(0,5):
			data_1[fields[i+5]] = stats_list1[:,i]

		data_1.to_csv(output_filename,sep='\t',index=False)

		return True

	def test_1_2_sub(self,filename1):
		
		data1 = np.load(filename1,allow_pickle=True)
		data1 = data1[()]
		t_vec1 = np.asarray(list(data1.keys()))
		print(t_vec1)
		t_vec2 = np.setdiff1d(t_vec1,'pred')
		print(t_vec2)
		stats_list1 = []
		serial_list1 = []
		chrom_list1 = []
		dict1 = dict()
		
		thresh_vec1 = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]
		for t1 in t_vec2:
			print(t1)
			vec1 = data1[t1]['vec1']
			print(vec1.shape)
			serial_1 = data1[t1]['serial1']
			
			stats_list1.extend(vec1[:,[0,1,2,5]])
			serial_list1.extend(serial_1)
			chrom_list1.extend([t1]*len(serial_1))

		dict1['stats'] = np.asarray(stats_list1)
		dict1['serial'] = np.asarray(serial_list1)
		dict1['chrom'] = np.asarray(chrom_list1)

		return dict1

	def test_1_2_merge(self,ref_filename,runid_vec,output_filename):
		
		path1 = './'
		file1 = pd.read_csv(ref_filename,header=None,sep='\t')
		colnames = list(file1)		
		col1, col2, col3, col_serial = colnames[0], colnames[1], colnames[2], colnames[3]
		ref_serial = file1[col_serial]

		run_id, run_id1 = runid_vec[0], runid_vec[1]
		filename1 = '%s/attention_%d.npy'%(path1,run_id)
		filename2 = '%s/attention_%d.npy'%(path1,run_id1)
		dict1 = self.test_1_2_sub(filename1)
		dict2 = self.test_1_2_sub(filename2)
		stats1, serial1, chrom1 = dict1['stats'], dict1['serial'], dict1['chrom']
		stats2, serial2, chrom2 = dict2['stats'], dict2['serial'], dict2['chrom']
		t_stats = np.vstack((stats1,stats2))
		t_serial = np.hstack((serial1,serial2))
		t_chrom = np.hstack((chrom1,chrom2))

		id1 = mapping_Idx(self.serial, t_serial)
		t_signal = self.signal[id1]
		
		num1 = len(t_serial)
		quantile_vec = np.zeros((num1,2))
		quantile_vec[:,0] = stats.rankdata(t_stats[:,-1], "average")/num1

		chrom_vec = np.unique(t_chrom)
		for chrom_id in chrom_vec:
			b1 = np.where(t_chrom==chrom_id)[0]
			quantile_vec[b1,1] = stats.rankdata(t_stats[b1,-1], "average")/len(b1)

		t_stats1 = np.hstack((t_stats,quantile_vec))

		id1 = mapping_Idx(ref_serial,t_serial)

		# Q1: whole distribution, 0.9 quantile; Q2: quantile by chromosome
		fields = ['chrom','start','stop','serial','signal','num','max','min','0.9-quantile','Q1','Q2']
		data_1 = pd.DataFrame(columns=fields)
		for i in range(0,4):
			data_1[fields[i]] = file1.loc[id1,colnames[i]]
		data_1['signal'] = t_signal
		t_chrom1 = np.asarray(data_1[fields[0]])

		t_chrom = ['chr%s'%(i1) for i1 in t_chrom]
		b1 = np.where(t_chrom1!=t_chrom)[0]
		if len(b1)>0:
			print('error!',len(b1))

		for i in range(5,11):
			data_1[fields[i]] = t_stats1[:,i-5]
			
		data_2 = data_1.sort_values(by=['serial'])
		data_2.to_csv(output_filename,index=False,sep='\t')

		return True

	def test_1_2(self,x_test1_trans,y_signal_test_ori,test_sel_list):

		tol = self.tol
		L = self.flanking
		run_id = self.run_id
			
		x_test, y_test, vec_test, vec_test_local = sample_select2a(x_test1_trans, y_signal_test_ori, test_sel_list, tol, L)
		print(x_test.shape,y_test.shape)

		x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[-1])
		y_test = y_test[:,L]
		print(x_test.shape,y_test.shape)

		# self.model = model
		y_predicted_test = self.model.predict(x_test)
		print(len(y_predicted_test))

		y_test = np.ravel(y_test)
		vec1 = score_2a(y_test, y_predicted_test)

		dict1 = dict()
		dict1['vec1'] = vec1
		dict1['y_test'], dict1['y_predicted_test'] = y_test, y_predicted_test

		return dict1

def run_initialize_1(file_path,species_id,resolution,run_id,
						generate,chromvec,test_chromvec,
						featureid,type_id,cell,method,ftype,ftrans,tlist,
						flanking,normalize,
						config,
						attention,feature_dim_motif=0,kmer_size=[6,5],
						region_list=[],
						region_list_1=[],
						load_type_id=1):
	
	run_id = int(run_id)
	generate = int(generate)
	ftype = str(ftype)
	tlist = str(tlist)
	normalize = int(normalize)
	flanking = int(flanking)
	attention = int(attention)

	temp1 = chromvec.split(',')
	chrom_vec = [str(chrom_id) for chrom_id in temp1]
	temp1 = test_chromvec.split(',')
	test_chromvec = [str(chrom_id) for chrom_id in temp1]
	temp1 = featureid.split(',')
	feature_idx = [int(f_id) for f_id in temp1]

	temp1 = ftype.split(',')
	ftype = [int(f_id) for f_id in temp1]

	temp1 = tlist.split(',')
	t_list = [int(t_id) for t_id in temp1]

	temp1 = ftrans.split(',')
	ftrans = [int(t_id) for t_id in temp1]

	feature_dim_transform = ftrans
	config['feature_dim_transform'] = feature_dim_transform

	train_chromvec = chrom_vec

	cell = str(cell)
	method = int(method)

	ref_filename = '%s/%s_%s_serial.bed'%(file_path,species_id,resolution)

	print(ref_filename)

	if species_id == 'hg38':
		filename1 = '%s/%s.smooth.sorted.bed'%(file_path,cell)
	else:
		filename1 = ref_filename
	
	print(feature_dim_transform)
	file_prefix_kmer = 'test'
	prefix = 'test'
	t_repli_seq= RepliSeq(file_path, species_id, resolution, run_id, generate,
					chrom_vec,test_chromvec,
					featureid,type_id,cell,method,ftype,ftrans,t_list,
					flanking,normalize,
					config,
					attention,feature_dim_motif,kmer_size)

	print(ref_filename)
	t_repli_seq.load_ref_serial(ref_filename)
	signal_normalize = 1
	if 'signal_normalize' in config:
		signal_normalize = config['signal_normalize']
	t_repli_seq.load_local_serial(filename1,region_list=region_list,type_id2=1,signal_normalize=signal_normalize,region_list_1=region_list_1)

	t_repli_seq.set_species_id(species_id,resolution)
	t_repli_seq.set_featuredim_motif(feature_dim_motif)

	return t_repli_seq

# class RepliSeqHMM(_Base1):

# 	def __init__(self, chromosome,run_id,generate,chromvec,test_chromvec,n_epochs,species_id,
# 					featureid,type_id,cell,method,ftype,ftrans,tlist,flanking,normalize,
# 					hidden_unit,batch_size,lr=0.001,step=5,
# 					activation='relu',min_delta=0.001,
# 					attention=1,fc1=0,fc2=0,units1=[50,50,50,25,50,25,0,0],kmer_size=[6,5],tol=5):
# 		_Base1.__init__(self, chromosome=chromosome,run_id=run_id,generate=generate,
# 							chromvec=chromvec,test_chromvec=test_chromvec,
# 							n_epochs=n_epochs,species_id=species_id,
# 							featureid=featureid,type_id=type_id,cell=cell,method=method,
# 							ftype=ftype,ftrans=ftrans,tlist=tlist,flanking=flanking,normalize=normalize,
# 							hidden_unit=hidden_unit,batch_size=batch_size,lr=lr,step=step,
# 							activation=activation,min_delta=min_delta,
# 							attention=attention,fc1=fc1,fc2=fc2,units1=units1,kmer_size=kmer_size,tol=tol)
		
# 		print('training chromvec',chromvec)
# 		print('test chromvec',test_chromvec)
# 		self.max_iter = self.config['max_iter']
# 		self.n_components = self.config['n_components']

# 		if 'n_features' in self.config:
# 			self.n_features = self.config['n_features']
# 		else:
# 			self.n_features = 1

# 		if 'param_stat' in self.config:
# 			self.param_stat = self.config['param_stat']
# 		else:
# 			self.param_stat = 'm'

# 		self.eps = 1e-07
# 		self.constant1 = np.log(2*3.1415926)
# 		self.stats = dict()
# 		# self.stats['obs_mean'] = dict()
# 		self.eps = 1e-09

# 	def prepare_1(self,x_train1_trans,train_sel_list,idx_train):
		
# 		tol = self.tol
# 		L = self.flanking
# 		run_id = self.run_id
			
# 		seq_list = generate_sequences(train_sel_list[idx_train],region_list=self.region_boundary)
# 		x_train, y_train, vec_train, vec_train_local = sample_select2a1(x_train1_trans[idx_train], y_signal_train1[idx_train], 
# 														train_sel_list[idx_train], seq_list, tol, L)

# 		return x_train, y_train, vec_train, vec_train_local

# 	# control function
# 	def control_1(self,path1,file_prefix):

# 		vec2 = dict()
# 		for type_id2 in self.t_list:
# 			print("feature transform")
# 			start = time.time()
# 			feature_dim_transform = self.feature_dim_transform
# 			self.type_id2 = type_id2
# 			self.prep_data(path1,file_prefix,type_id2,feature_dim_transform)
# 			stop = time.time()
# 			print('prepare',stop-start)
# 			log_likelihood = self.training()
# 			vec2[type_id2] = log_likelihood

# 		train_id, valid_id, test_id = self.idx_list['train'], self.idx_list['valid'], self.idx_list['test']
# 		print('test_sel_list',len(self.train_sel_list[test_id]))
# 		vec2.update({'train':self.train_sel_list[train_id],'valid':self.train_sel_list[valid_id],'test':self.train_sel_list[test_id]})
# 		filename1 = '%s/feature_transform_%d_%d_%d_%d.npy'%(self.path,self.t_list[0],feature_dim_transform[0],self.method,self.run_id,self.method)
# 		np.save(filename1,vec2,allow_pickle=True)

# 		return True



