# -*- coding: utf-8 -*-
import numpy as np
import copy


def transfer_labels(labels):
	#some labels are [1,2,4,11,13] and is transfer to standard label format [0,1,2,3,4]
	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = labels.shape[0]

	for i in range(num_samples):
		new_label = np.argwhere( labels[i] == indexes )[0][0]
		labels[i] = new_label
	return labels, num_classes

def load_data(filename):
	data_label = np.loadtxt(filename,delimiter=',')
	data = data_label[:,1:]
	label = data_label[:,0].astype(np.int32)
	return data, label

def convertToOneHot(vector, num_classes=None):
	#convert label to one_hot format
    vector = np.array(vector,dtype = int)
    if 0 not in np.unique(vector):
        vector = vector - 1
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    assert num_classes is not None

    assert num_classes > 0
    vector = vector % num_classes

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(np.int32)

def next_batch(batch_size, data, label, end_to_end,input_dimension_size,num_step, Trainable):
	if end_to_end:
		data [ np.where(np.isnan(data))] = 128
	need_label = copy.deepcopy(label)
	label = convertToOneHot(label, num_classes = len(np.unique(label)))
	assert data.shape[0] == label.shape[0]
	assert data.shape[0] >= batch_size
	row = data.shape[0]
	batch_len = int( row / batch_size )
	left_row = row - batch_len * batch_size

	#shuffle data for train
	if Trainable:
		indices = np.random.permutation(data.shape[0])
		rand_data = data[indices]
		rand_label = label[indices]
		need_rand_label = need_label[indices]
	else:
		rand_data = data
		rand_label = label
		need_rand_label = need_label

	for i in range(batch_len):
		batch_input = rand_data[ i*batch_size : (i+1)*batch_size, :]
		batch_prediction_target = rand_data[ i*batch_size : (i+1)*batch_size, input_dimension_size:]
		mask = np.ones_like(batch_prediction_target)
		mask [ np.where( batch_prediction_target == 128 ) ] = 0
		batch_label = rand_label[ i*batch_size : (i+1)*batch_size, : ]
		batch_need_label = need_rand_label[i*batch_size : (i+1)*batch_size]
		yield (batch_input.reshape(-1, num_step, input_dimension_size), batch_prediction_target.reshape(-1, num_step - 1, input_dimension_size), mask.reshape(-1, num_step - 1, input_dimension_size), batch_label, batch_size, batch_need_label)

	# padding data for equal batch_size
	if left_row != 0:
		need_more = batch_size - left_row
		need_more = np.random.choice( np.arange(row), size = need_more )
		batch_input = np.concatenate((rand_data[ -left_row: , : ], rand_data[need_more]), axis=0)
		batch_prediction_target = np.concatenate((rand_data[ -left_row: , : ], rand_data[need_more]), axis=0)[:, input_dimension_size:]
		assert batch_input.shape[0] == batch_prediction_target.shape[0]
		assert batch_input.shape[1] - input_dimension_size == batch_prediction_target.shape[1]
		mask = np.ones_like(batch_prediction_target)
		mask [ np.where( batch_prediction_target == 128 ) ] = 0
		batch_label = np.concatenate( (rand_label[ -left_row: , : ], rand_label[ need_more ]),axis=0)
		batch_need_label =  np.concatenate( (need_rand_label[-left_row:], need_rand_label[ need_more ]), axis=0)
		yield (batch_input.reshape(-1, num_step, input_dimension_size), batch_prediction_target.reshape(-1, num_step - 1, input_dimension_size), mask.reshape(-1, num_step - 1, input_dimension_size), batch_label, left_row, batch_need_label)
