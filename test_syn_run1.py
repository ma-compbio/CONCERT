
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
tf.keras.backend.set_floatx('float32')

import numpy as np
import utility_1
import pandas as pd
import os.path
from utility_1 import mapping_Idx
import concert_syn as training1
import random
from optparse import OptionParser
from itertools import combinations

chrom_idvec = [1,2,3,4,5]
chrom_idvec = [4,5,6,7,8,9,11,12,13,14,15]
resolution = '5k'

chromosome = '1'
run_id = 1
generate = 0
chromvec = '1,2,3,4,5,16,17'
test_chromvec = '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22'
n_epochs = 100
species_id = 0
featureid = '0,2'
type_id = 0
cell = 'GM12878'
method = 20
ftype = '-5'
ftrans = '50,50'
tlist = '0'
flanking = 60
normalize = 0
unit = 32
batch_size = 512
attention = 1
lr = 0.0005
step = 7
fc1_output_dim = 0
fc2_output_dim = 0
activation='relu'
min_delta = 0.0005
units1 = [32,50]

temp1 = test_chromvec.split(',')
test_chromvec1 = [str(chrom_id) for chrom_id in temp1]
temp1 = ftrans.split(',')
ftrans1 = [int(t_id) for t_id in temp1]
feature_dim_transform = ftrans1
feature_dim_motif = 0

flag = True

def test_1(base1,vec1,vec3,vec3_1,feature_dim_motif,celltype_vec,loss_function_typeid=1,
				local_conv_size=3,kernel_size2=3,flanking=50,gap_thresh=5,seq_len_thresh=5,
				train_mode=1,train_mode2=0,run_id_load=-1,sel_conv_id=1,model_type_id=5):

	cnt1 = 0
	run_id_load_ori = run_id_load

	for value1 in vec1:
		tau1, n_select1, activation3, activation2, activation_self, activation_basic, hidden_unit, regularizer_1 = value1[1:]
		cell_id = value1[0]
		cell = celltype_vec[cell_id]
		print(cell_id,cell)
		print(value1)

		path1 = './'
		vec2 = [52,58,60,55,57,0,1,56,62]
		t1 = [5]
		t2 = [0,0,0,-1]

		t2_1 = [3]
		t2_2 = list(range(0,len(vec3)))
		type_idvec1 = [1]

		regularizer1_vec = [0]
		regularizer2_vec = [0,1e-04,1e-05]

		if activation3=='sigmoid':
			regularizer2_vec = [0]
			regularizer1_vec = [0]
		elif activation3=='relu':
			regularizer2_vec = [1e-04]
			regularizer1_vec = [1e-05]
		else:
			regularizer2_vec = [1e-04]
			regularizer1_vec = [0]

		t2_3 = []
		for t_value3 in type_idvec1:
			if t_value3==1:
				ratio1 = 0.9
			else:
				ratio1 = 0.95

			for regularizer1 in regularizer1_vec:
				for regularizer2 in regularizer2_vec:
					for t_value1 in t2_1:
						for sel1 in t1:
							for t_value2 in t2_2:
								t2_3.append([t_value1,t_value2,t_value3,ratio1,regularizer1,regularizer2,sel1])
		
		for t_value3 in t2_3:
			i,j = t_value3[0], t_value3[1]
			method = vec2[i]
			chromvec = vec3[j]
			test_chromvec = vec3_1[j]
			type_id1, ratio1, regularizer1, regularizer2 = t_value3[2:6]
			sel1 = t_value3[6]

			run_id = base1+cnt1
			cnt1 = cnt1+1

			predict_filename = 'test_%d_%d_score.txt'%(run_id,run_id)
			predict_filename1 = './test_%d_%d_score.txt'%(run_id,run_id)

			train_filename = 'test%d.h5'%(run_id)
			if (os.path.exists(train_filename)==True) and (train_mode==1):
				continue

			activation = 'relu'
			print(cell,run_id,method,sel1,tau1,n_select1,activation,activation2,activation3,activation_self,
					activation_basic,regularizer_1,regularizer1,regularizer2,chromvec,test_chromvec,model_type_id)
			
			units1=[50,50,50,25,50,25,0,0]
			if method==31:
				units1=[50,50,50,25,50,50,0,0]
			seed1 = 0
			random.seed(seed1)
			tf.compat.v1.set_random_seed(seed1)
			np.random.seed(seed1)
			file_path = './'
			species_id = 'hg38'
			resolution = '5k'

			n_epochs = 100
			step = 7
			lr = 0.0005
			min_delta = 0.0001
			kmer_size=[6,5]
			pos = 0
			if activation3 in ['sigmoid','relu','ReLU']:
				typeid_sample = 3
			else:
				typeid_sample = 1

			config={'output_dim':hidden_unit,'fc1_output_dim':fc1_output_dim,'fc2_output_dim':fc2_output_dim,
					'units1':units1[0],'units2':units1[1],
					'feature_dim_vec':units1[2:],
					'tau':tau1,'n_select':n_select1,
					'activation':activation,
					'activation2':activation2,
					'activation3':activation3,
					'activation_self':activation_self,'lr':lr,
					'step':step,'min_delta':min_delta,
					'n_epochs':n_epochs,
					'batch_size':batch_size,
					'pos_code':pos,'tol':5,
					'typeid_sample':typeid_sample,	# sample constructor: typeid_sample: 2, -log(log(p)); 3, -log(p)
					'type_id1':type_id1,	# valid data by each chromosome
					'ratio1':ratio1,
					'train_mode':train_mode,
					'predict_train_attention':0,
					'signal_plot':0,
					'feature_dim_select':sel1,
					'regularizer_1':regularizer_1,
					'regularizer1':regularizer1,
					'regularizer2':regularizer2,
					'local_conv_size':local_conv_size
					}
					
			config['signal_normalize_clip'] = 0
			config['batch_norm2'] = 1
			config['load_type'] = 0
			loss_function_vec = ['mean_squared_error','logcosh','binary_crossentropy']
			config['loss_function'] = loss_function_vec[loss_function_typeid]
			config['activation_basic'] = activation_basic
			if activation_basic=='tanh':
				config['scale'] = [-1,1]
			else:
				config['scale'] = [0,1]

			# predict
			if run_id in [233,234]:
				interval = 5000
			else:
				interval = 5000

			config['interval'] = interval
			config['sel_conv_id'] = sel_conv_id
			config['celltype_id'] = cell_id
			config['dilated_conv_kernel_size2'] = kernel_size2
			config['gap_thresh'] = gap_thresh
			config['seq_len_thresh'] = seq_len_thresh
			config['model_type_id'] = model_type_id
			config['est_attention'] = 1
			predict_mode_list = [[1,0],[0,1],[1,1]]
			predict_valid, predict_test = predict_mode_list[train_mode2]
			config['predict_valid'] = predict_valid
			config['predict_test'] = predict_test

			config['layer_name_est'] = 'logits_T_1'
			base1_1 = 610001
			config['valid_output_filename'] = 'test_valid_%d.1.txt'%(base1_1)
			config['predict_test_filename'] = '2'
			if cell_id==9:
				config['signal_normalize'] = 0
			else:
				config['signal_normalize'] = 1
			print(config['predict_valid'],config['predict_test'],config['valid_output_filename'])
			print(config['model_type_id'],config['dilated_conv_kernel_size2'],config['gap_thresh'],config['seq_len_thresh'])

			if train_mode==2:
				type_id2 = 2
				context_size = 2*flanking+1
				epoch_id = 1
				local_id = 3
				config['model_path1'] = '%s/model_%d_%d_%d_%d_%d.h5'%(file_path,run_id,type_id2,context_size,epoch_id,local_id)
				config['train_pre_epoch'] = [epoch_id,local_id+1]

			config['pred_filename1'] = 'test_vec2_%d_%d_[%d]_%s.1.txt'%(run_id,method,sel1,cell)
			config['pred_filename2'] = '%s/feature_transform_%d_%d.1.txt'%(file_path,run_id,method)

			optimizer_vec1 = ['SGD','RMSprop','Adadelta','Adagrad','Nadam','Adam']
			lr_schedule = 0
			init_lr, decay_rate1 = 0.01, 0.9
			optimizer_id = 5
			optimizer1 = optimizer_vec1[optimizer_id]
			config.update({'optimizer':optimizer1,'init_lr':init_lr,'decay_rate1':decay_rate1,'lr_schedule':lr_schedule})

			if run_id_load_ori<0:
				run_id_load = run_id
			else:
				run_id_load = run_id_load_ori

			model_path1 = ''
			if train_mode==0:
				model_path1 = 'test%d.h5'%(run_id_load)
				print(run_id,run_id_load,model_path1)
				if os.path.exists(model_path1)==False:
					config['train_mode'] = 1
					model_path1 = ''
					print('file does not exist',model_path1)
					break

			t_celltype_id = 'H1-hESC'
			filename_list1 = ['%s_loop_2Ends.1.bed'%(t_celltype_id),
							'%s_loop_0Ends.1.bed'%(t_celltype_id),
							'%s_noloop_0Ends.1.bed'%(t_celltype_id)]

			annot_vec1 = ['deletion','duplication','deletion','duplication']
			annot_vec2 = [0,0,1,1]

			celltype_vec = ['GM12878','K562','H1-hESC','H9','HCT116','HEK293','RPE-hTERT','U2OS','IMR-90.2']
			celltype_vec = np.asarray(celltype_vec)

			path_1 = 'pred'
			list1 = []

			celltype_id = cell
			filename1 = './%s_%d.label.sorted.bed'%(celltype_id,run_id)

			filename1 = 'test_data_pred1.1.txt'
			filename_list1 = [filename1]
			list1 = ['tol1']
			annot_vec2 = list1

			id2 = 0
			tol_1 = 1
			for t_filename1 in filename_list1:
				data1_ori = pd.read_csv(t_filename1,sep='\t')
				data1 = data1_ori
				colnames = list(data1)

				if data1.shape[1]>=5:
					data1 = data1.loc[:,colnames[0:4]]
				data1.columns = ['chrom','start','stop','serial']

				chrom_1 = np.asarray(data1['chrom'])
				id1 = np.where((chrom_1!='chrM')&(chrom_1!='chrX')&(chrom_1!='chrY'))[0]
				data1 = data1.loc[id1,:]
				data1.reset_index(drop=True,inplace=True)
				print(data1.shape)

				chrom_1 = np.asarray(data1['chrom'])
				position1 = data1.loc[:,['start','stop']]
				position1 = np.asarray(position1)
				
				chrom1 = [int(t_chrom[3:]) for t_chrom in chrom_1]
				chrom1 = np.asarray(chrom1)
				region_num1 = data1.shape[0]
				print('region_num',len(chrom1),chrom1[0],position1[0],position1[-1])

				if data1.shape[0]==1:
					region_list = [chrom1[0]]+list(position1)
				else:
					region_list = np.hstack((chrom1[:,np.newaxis],position1))

				num1 = len(region_list)
				bin_size = 5000
				tol = tol_1*bin_size
				for i in range(num1):
					region_list[i,1:3] = region_list[i,1]-tol, region_list[i,2]+tol

				print(t_filename1,num1,len(region_list),region_list[0:10])

				id2 += 1
				extend1 = 1000
				bin_size = 5000
				chromvec = '1'
				test_id1=2
				test_id2=0
				# predict for each locus
				if test_id1==1: 
					for i in range(num1):
						t_region = list(region_list[i])
						start_extend = t_region[1]-bin_size*extend1
						stop_extend = t_region[2]+bin_size*extend1
						region_list1 = [t_region]
						region_list_1 = [[t_region[0],start_extend,stop_extend]]

						config['tol_region_search'] = 0
						config['region_list_test'] = region_list1
						t_chrom1 = t_region[0]
						test_chromvec_pre1 = '%s'%(t_chrom1)
						test_chromvec1 = test_chromvec_pre1.split(',')
						chromvec = str(t_chrom1)
						test_chromvec = str(t_chrom1)
						print(region_list1,region_list_1,test_chromvec1)
						config['filename_prefix_predict'] = '1_%s_region_tol%s.local2.%d_%d.repeat.pre2.ori'%(test_chromvec1[0],tol_1,i+1,t_region[1])
						
						t_repli_seq = training1.run_initialize_1(file_path,species_id,resolution,run_id,
												generate,chromvec,test_chromvec,
												featureid,type_id,cell,method,ftype,ftrans,tlist,
												flanking,normalize,
												config,
												attention,feature_dim_motif,kmer_size,
												region_list = region_list1,
												region_list_1 = region_list_1
												)
						
						file_prefix = 'chr1_chr22'

						file_path = './'
						estimation = 1
						t_repli_seq.control_pre_test3_predict(file_path,file_prefix,
																model_path1=model_path1,
																type_id=-1,
																est_attention=estimation)

				# predict for loci in the list
				elif test_id1==2:
					
					estimation = 1
					t_chrom1 = np.sort(np.unique(region_list[:,0]))
					region_list_1 = []
					chrom = region_list[:,0]
					region_list1 = region_list

					for t_chrom_id in t_chrom1:
						b1 = np.where(chrom==t_chrom_id)[0]
						t_region_list = region_list[b1,:]
						t_region = t_region_list[0]
						chrom_id1 = t_region[0]
						start_extend = np.max([t_region_list[0][1]-bin_size*extend1,0])
						stop_extend = t_region_list[-1][2]+bin_size*extend1
						region_list_1.append([chrom_id1,start_extend,stop_extend])
						print(chrom_id1,start_extend,stop_extend)

					config['tol_region_search'] = 0
					config['region_list_test'] = region_list1
					str1 = [str(t_chrom_id) for t_chrom_id in t_chrom1]
					test_chromvec_pre1 = ','.join(str1)
					test_chromvec1 = test_chromvec_pre1.split(',')

					chromvec = test_chromvec_pre1
					test_chromvec = chromvec
					print(region_list1,region_list_1,test_chromvec1)

					config['filename_prefix_predict'] = '1_%s_region_tol%d.local.%d'%(test_chromvec1[0],tol_1,run_id)

					t_repli_seq = training1.run_initialize_1(file_path,species_id,resolution,run_id,
												generate,chromvec,test_chromvec,
												featureid,type_id,cell,method,ftype,ftrans,tlist,
												flanking,normalize,
												config,
												attention,feature_dim_motif,kmer_size,
												region_list = region_list1,
												region_list_1 = region_list_1
												)
						
					file_prefix = 'chr1_chr22'

					t_repli_seq.control_pre_test3_predict(file_path,file_prefix,
																model_path1=model_path1,
																type_id=-1,
																est_attention=estimation)

				tol_1 = 0
				extend1 = 1000

				if test_id2==0:
					return

				for combination_size in [2]:
					serial_vec_local = [1,2]
					
					serial = np.asarray(data1.loc[:,'serial'])
					b1 = np.where((serial>=serial_vec_local[0])&(serial<=serial_vec_local[1]))[0]
					data2 = data1.loc[b1,:]
					data2.reset_index(drop=True,inplace=True)
					region_num = data2.shape[0]
					serial_local = np.asarray(data2['serial'])

					region_list_pre1 = np.asarray(data2.loc[:,['chrom','start','stop','serial']])
					region_list_pre2 = region_list_pre1.copy()
					num2 = region_list_pre1.shape[0]
					tol = tol_1*bin_size
					extend_size1 = bin_size*extend1
					for l in range(num2):
						region_list_pre1[l,1:3] = region_list_pre1[l,1]-bin_size, region_list_pre1[l,2]
						region_list_pre2[l,1:3] = region_list_pre2[l,1]+bin_size, region_list_pre2[l,2]+bin_size

					list1 = list(combinations(range(region_num),combination_size))
					sel_num = len(list1)
					print(region_num,serial_local,combination_size,sel_num)
					print(region_list_pre1)

					list1 = [[0,4]]
					sel_num = 1

					for i1 in range(1):
						for i2 in range(sel_num):
							sel_id1 = list(list1[i2])
							region_list1 = [region_list_pre2[sel_id1[0],:], region_list_pre1[sel_id1[1],:]]
							print(region_list1,sel_id1)

							config['tol_region_search'] = 0
							config['region_list_test'] = region_list1
							t_chrom1 = region_list1[0][0]
							chromvec = str(t_chrom1)
							test_chromvec = str(t_chrom1)

							start_extend = region_list1[0][1]-extend_size1
							stop_extend = region_list1[-1][2]+extend_size1
							region_list_1 = [[t_chrom1,start_extend,stop_extend]]

							print('region_list1')
							print(region_list1,region_list_1,test_chromvec)
							
							config['filename_prefix_predict'] = '1_%s_region_tol%s.local2_1.%d_%d_%d_%d.%d.repeat.2_1.3'%(t_chrom1,tol_1,sel_id1[0],sel_id1[-1],region_list1[0][1],region_list1[-1][1],combination_size)
							
							t_repli_seq = training1.run_initialize_1(file_path,species_id,resolution,run_id,
													generate,chromvec,test_chromvec,
													featureid,type_id,cell,method,ftype,ftrans,tlist,
													flanking,normalize,
													config,
													attention,feature_dim_motif,kmer_size,
													region_list = region_list1,
													region_list_1 = region_list_1
													)
							
							file_prefix = 'chr1_chr22'
							file_path = './'
							estimation = 1
							t_repli_seq.control_pre_test3_predict(file_path,file_prefix,
																model_path1=model_path1,
																type_id=-1,
																est_attention=estimation)


def run(chromosome,run_id,generate,chromvec,test_chromvec,n_epochs,species_id,
		featureid,cell,method,ftype,ftrans,tlist,flanking,
		normalize,unit,batch,fc1,fc2,
		feature_dim_motif,loss_function_typeid,activation3_typeid,
		local_conv_size,kernel_size2,gap_thresh,seq_len_thresh,
		train_mode,train_mode2,
		run_id_load,device_id,
		sel_conv_id,model_type_id,chrom_vec_id,
		activation2_typeid,activation_self_typeid,activation_basic_typeid):	
	
	cnt1 = 0
	base1 = int(run_id)
	feature_dim_motif = int(feature_dim_motif)
	loss_function_typeid = int(loss_function_typeid)
	local_conv_size = int(local_conv_size)
	kernel_size2 = int(kernel_size2)
	train_mode = int(train_mode)
	train_mode2 = int(train_mode2)
	device_id = int(device_id)
	type_id3 = str(activation3_typeid)
	type_id3 = type_id3.split(',')
	activation3_sel_id = [int(i) for i in type_id3]
	flanking = int(flanking)
	run_id_load = int(run_id_load)
	sel_conv_id = int(sel_conv_id)
	gap_thresh = int(gap_thresh)
	seq_len_thresh = int(seq_len_thresh)
	model_type_id = int(model_type_id)
	activation2_typeid = str(activation2_typeid)
	activation2_typeid = activation2_typeid.split(',')
	activation2_sel_id = [int(i) for i in activation2_typeid]

	activation_self_typeid = str(activation_self_typeid)
	activation_self_typeid = activation_self_typeid.split(',')
	activation_self_id = [int(i) for i in activation_self_typeid]

	activation_basic_typeid = str(activation_basic_typeid)
	activation_basic_typeid = activation_basic_typeid.split(',')
	activation_basic_id = [int(i) for i in activation_basic_typeid]

	cell_id_ori = int(cell)
	chrom_vec_id = int(chrom_vec_id)

	print(base1,feature_dim_motif,loss_function_typeid,local_conv_size,train_mode,device_id)

	chrom_vec = list(range(1,23))
	vec3_pre = [[1,3,5,7,9,11,13,15],[2,4,6,8,10,12,14,16]]
	
	flag = False
	vec3, vec3_1 = [], []
	for t_value1 in vec3_pre:
		str1 = [str(t1) for t1 in t_value1]
		vec3.append(','.join(str1))
		t_value2 = np.sort(list(set(chrom_vec)-set(t_value1)))
		str2 = [str(t2) for t2 in t_value2]
		vec3_1.append(','.join(str2))

	vec3 = ['11']
	vec3_1 = [str(t_chrom) for t_chrom in range(1,23)]
	list_1 = [[str(t_chrom) for t_chrom in range(1,23)]]

	list_2 = []
	for t_list1 in list_1:
		list_2.append([','.join(t_list1)])

	vec3_1 = list_2[chrom_vec_id]

	t1 = list(range(0,8))+[-1]
	t2 = [0,0,0,1]
	t1_1 = [0.05,0.1,0.25,0.025,0.01,0.5]
	t1_2 = [10,20,5]
	activation_vec = np.asarray(['sigmoid','relu','linear','tanh','softsign','ReLU'])
	activation2_vec = np.asarray(['tanh','softsign','sigmoid','relu','linear'])
	activation_self_vec = np.asarray(['tanh','sigmoid','relu'])

	hidden_unit_vec = [32]
	regularizer_1 = 1
	t_value1 = 0.1
	t_value2 = 10
	celltype_vec = ['GM12878','K562','H1-hESC','H9','HCT116','HEK293','RPE-hTERT','U2OS','IMR-90.2']
	celltype_vec = ['GM12878','K562','H1-hESC','H9','HCT116','HEK293','RPE-hTERT','U2OS','IMR-90.2','WTC11']
	num2 = len(celltype_vec)
	cell_vec1 = [0,1,2,7,8]
	cell_vec2 = [3,5,4,6]
	cell_vec3 = [3,5,0,4,6,1,2,7,8]
	if cell_id_ori>=0:
		cell_vec3 = [cell_id_ori]
	elif cell_id_ori==-2:
		cell_vec3 = [3,0]
	else:
		cell_vec3 = [3,5,0,4,6,1,2,7,8]

	t1_3 = []
	for t_value1 in [0.1]:
		for t_value2 in [10]:
			for t_activation_basic in activation_vec[activation_basic_id]:
				for t_activation_self in activation_self_vec[activation_self_id]:
					for t_activation2 in activation2_vec[activation2_sel_id]:
						for t_activation3 in activation_vec[activation3_sel_id]:
							for hidden_unit1 in hidden_unit_vec:
								for cell_id in cell_vec3:
									t1_3.append([cell_id,t_value1,t_value2,t_activation3,t_activation2,
													t_activation_self,t_activation_basic,
													hidden_unit1,regularizer_1])
	vec1 = t1_3

	test_1(base1,vec1,vec3,vec3_1,feature_dim_motif,celltype_vec,loss_function_typeid,
				local_conv_size,kernel_size2,flanking,gap_thresh,seq_len_thresh,
				train_mode,train_mode2,run_id_load,sel_conv_id,model_type_id)

	return True

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="100", help="experiment id")
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
	parser.add_option("-p","--ftrans",default="300",help="transform feature dimension")
	parser.add_option("-q","--tlist",default="3,1,2",help="feature transform method")
	parser.add_option("-v","--normalize",default="0",help="normalize feature")
	parser.add_option("-w","--flanking",default="50",help="flanking region")
	parser.add_option("--bc",default="128",help="batch size")
	parser.add_option("-h","--unit",default="32",help="hidden unit")
	parser.add_option("--fc1",default="0",help="fc1 output dim")
	parser.add_option("--fc2",default="0",help="fc2 output dim")
	parser.add_option("--typeid1",default="0",help="feature motif")
	parser.add_option("--typeid2",default="1",help="loss function type id")
	parser.add_option("--typeid3",default="1",help="activation3 type id")
	parser.add_option("--typeid5",default="0",help="kmer size")
	parser.add_option("--typeid6",default="0",help="train type id")
	parser.add_option("--typeid7",default="1",help="activation2 type id")
	parser.add_option("--typeid8",default="0",help="activation self type id")
	parser.add_option("--typeid9",default="0",help="activation basic type id")
	parser.add_option("--sel2",default="50,50",help="feature dim")
	parser.add_option("--l1",default="3",help="local conv size")
	parser.add_option("--l2",default="3",help="conv size 2")
	parser.add_option("--l3",default="10",help="gap thresh")
	parser.add_option("--l5",default="2",help="seq len thresh")
	parser.add_option("--mode1",default="1",help="train mode")
	parser.add_option("--sel",default="0",help="sel id")
	parser.add_option("--sel1",default="5",help="feature select id")
	parser.add_option("--mode2",default="0",help="shuffle local")
	parser.add_option("--mode3",default="0",help="train mode2")
	parser.add_option("--id2",default="-1",help="run id load")
	parser.add_option("--id1",default="1",help="device id")
	parser.add_option("--id3",default="1",help="convolution id")
	parser.add_option("--id5",default="5",help="model type id")
	parser.add_option("--id6",default="0",help="chrom vec id")

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':

	opts = parse_args()
	run(opts.chromosome,opts.run_id,opts.generate,opts.chromvec,opts.testchromvec,
		opts.n_epochs,opts.species,opts.featureid,opts.cell,
		opts.method,opts.ftype,opts.ftrans,opts.tlist,opts.flanking,opts.normalize,opts.unit,opts.bc,
		opts.fc1,opts.fc2,opts.typeid1,opts.typeid2,opts.typeid3,opts.l1,opts.l2,opts.l3,opts.l5,
		opts.mode1,opts.mode3,
		opts.id2,opts.id1,opts.id3,opts.id5,opts.id6,
		opts.typeid7,opts.typeid8,opts.typeid9)


