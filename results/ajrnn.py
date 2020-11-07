
import numpy as np
import copy
import tensorflow as tf
import argparse

'''load saved model'''
def load_model_N_inference(dataset,missing_ratio):
    Length_dict = {
        '50words':270,
        'Adiac':176,
        'ChlorineConcentration':166,
        'ArrowHead':251,
        'CBF':128,
        'Computers':720,
        'Cricket_X':300,
        'Cricket_Y':300,
        'Cricket_Z':300,
        'FISH':463,
        'Ham':431,
        'LargeKitchenAppliances':720,
        'Plane':144,
        'RefrigerationDevices':720,
        'ScreenType':720,
        'synthetic_control':60,
        'Two_Patterns':128
    }

    model_dir = dataset + '_' + str(missing_ratio)
    
    
    test_data_filename = './data/'	+dataset +'/'+dataset +'_TEST_' +str(missing_ratio) +'.csv'	
    test_data, test_labels = load_data(test_data_filename)
    test_label, test_classes = transfer_labels(test_labels)
    
    batch_size = 20
    
    #for univariate
    input_dimension_size = 1
    num_steps = test_data.shape[1]
    
    saver = tf.train.import_meta_graph("./model/"+model_dir+"/model.ckpt.meta")
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        
        
        saver.restore(sess, "./model/"+model_dir+"/model.ckpt")
        
        accuracy = tf.get_default_graph().get_tensor_by_name('accuracy:0')
        prediction = tf.get_default_graph().get_tensor_by_name('Generator_LSTM/prediction:0')
        Label_predict = tf.get_default_graph().get_tensor_by_name('test_probab:0')
        Last_hidden_output = tf.get_default_graph().get_tensor_by_name('Generator_LSTM/RNN/RNN/multi_rnn_cell/cell_0/gru_cell/add_'+str(Length_dict[dataset]-1) +':0')
        input_tensors_input = tf.get_default_graph().get_tensor_by_name("inputs:0")
        input_tensors_prediction_target = tf.get_default_graph().get_tensor_by_name("prediction_target:0")
        input_tensors_mask = tf.get_default_graph().get_tensor_by_name("mask:0")
        input_tensors_label_target = tf.get_default_graph().get_tensor_by_name("label_target:0")
        input_tensors_lstm_keep_prob = tf.get_default_graph().get_tensor_by_name("lstm_keep_prob:0")
        input_classification_keep_prob = tf.get_default_graph().get_tensor_by_name("classification_keep_prob:0")
    
        
        
        total_test_accuracy = []
        total_sample_num = 0
        total_Pre = []
        test_label_predict = []                
        '''test'''
        for input, prediction_target, mask, label_target, batch_size, batch_need_label in next_batch(batch_size, test_data, test_label, True,input_dimension_size,num_steps, Trainable = False):
            total_sample_num += batch_size
            batch_accuracy, batch_Pre, batch_Label_predict, batch_test_hidden = sess.run([accuracy, prediction, Label_predict, Last_hidden_output], feed_dict={input_tensors_input: input, input_tensors_prediction_target: prediction_target,
																		input_tensors_mask: mask, input_tensors_label_target: label_target,
																		input_tensors_lstm_keep_prob: 1.0, input_classification_keep_prob: 1.0})
            total_test_accuracy.append(batch_accuracy)					
            total_Pre.append(batch_Pre)
            
            #---------
            batch_Label_predicts = np.argmax(batch_Label_predict,1)
            test_label_predict.append(batch_Label_predicts)
	
        assert total_sample_num == test_data.shape[0]
        Test_acc = np.mean(np.array(total_test_accuracy).reshape(-1)[:total_sample_num])
        #----------
        test_label_predict = np.reshape(test_label_predict,[-1,1])[:total_sample_num]
        
    return Test_acc

def load_data(filename):
	data_label = np.loadtxt(filename,delimiter=',')
	data = data_label[:,1:]
	label = data_label[:,0].astype(np.int32)
	return data, label

def transfer_labels(labels):
	#some labels are [1,2,4,11,13] and is transfer to standard label format [0,1,2,3,4]
	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = labels.shape[0]

	for i in range(num_samples):
		new_label = np.argwhere( labels[i] == indexes )[0][0]
		labels[i] = new_label
	return labels, num_classes

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
			   	
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name',type=str,required=True)
    parser.add_argument('--missing_ratio',type=int,required=True)
    
    config = parser.parse_args()
    ACC = load_model_N_inference(config.dataset_name,config.missing_ratio)
    
    print('Test ACC on %s_%d : %f' % (config.dataset_name,config.missing_ratio,ACC))