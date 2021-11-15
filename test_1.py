import numpy as np
import concert
import utility_1

resolution = '5k'
chromosome = '1'
run_id = 2
generate = 0
chromvec = '1,2,3,4,5'
test_chromvec = '10'
n_epochs = 100
species_id = 0
featureid = '0,2'
type_id = 0
cell = 'GM12878'
method = 11
ftype = '-5'
ftrans = '50,50'
tlist = '0'
flanking = 50
normalize = 0
unit = 64
batch_size = 512
attention = 1
lr = 0.0005
step = 5
fc1_output_dim = 0
fc2_output_dim = 0
activation='relu'
delta = 0.001

celltype_vec = ['GM12878','IMR-90','K562','H1-hESC','H9','HCT116','HEK293','RPE-hTERT','U2OS']
num1 = len(celltype_vec)

for i in range(0,num1):
	cell = celltype_vec[i]
	run_id = i
	print(cell)
	feature_dim_motif = 769
	training2_2.run(chromosome,run_id,generate,chromvec,test_chromvec,n_epochs,species_id,
		 featureid,type_id,cell,method,ftype,ftrans,tlist,flanking,normalize,unit,batch_size,
		 lr,step,activation,delta,attention,fc1_output_dim,fc2_output_dim,feature_dim_motif)
	
