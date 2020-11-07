# -*- coding: utf-8 -*-
import tensorflow as tf
import utils
import os
import numpy as np
import argparse
Missing_value = 128.0


class Config(object):
    layer_num = 1
    hidden_size = 100
    learning_rate = 1e-3
    cell_type = 'GRU'
    lamda = 1
    D_epoch = 1
    GPU = '0'
    '''User defined'''
    batch_size = None   #batch_size for train
    epoch = None    #epoch for train
    lamda_D = None  #epoch for training of Discriminator
    G_epoch = None  #epoch for training of Generator
    train_data_filename = None
    test_data_filename = None


def RNN_cell(type, hidden_size, keep_prob):
	if type == 'LSTM':
		cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
	elif type == 'GRU':
		cell = tf.contrib.rnn.GRUCell(hidden_size)
	return cell


class Generator(object):

	def __init__(self, config):
		self.batch_size = config.batch_size
		self.hidden_size = config.hidden_size
		self.num_steps = config.num_steps
		self.input_dimension_size = config.input_dimension_size
		self.cell_type = config.cell_type
		self.lamda = config.lamda
		self.class_num = config.class_num
		self.layer_num = config.layer_num
		self.name = 'Generator_LSTM'

	def build_model(self):

		# input has shape (batch_size, n_steps, embedding_size)
		input = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, self.input_dimension_size], name = 'inputs') # input

		# prediction_target has shape (batch_size, n_steps-1, embedding_size)
		prediction_target = tf.placeholder(tf.float32, [self.batch_size, self.num_steps - 1, self.input_dimension_size], name = 'prediction_target')
		mask = tf.placeholder(tf.float32, [self.batch_size, self.num_steps - 1, self.input_dimension_size], name = 'mask')


		# label_target has shape (batch_size, self.class_num) # likes one-hot
		label_target = tf.placeholder(tf.float32, [self.batch_size, self.class_num], name = 'label_target')

		# dropout for rnn
		lstm_keep_prob = tf.placeholder(tf.float32, [],name='lstm_keep_prob')
		classfication_keep_prob = tf.placeholder(tf.float32, [],name='classification_keep_prob')
		with tf.variable_scope(self.name):
			# project layer weight W and bias
			W = tf.Variable(tf.truncated_normal( [self.hidden_size, self.input_dimension_size], stddev = 0.1 ), dtype = tf.float32, name= 'Project_W')
			bias = tf.Variable(tf.constant(0.1,shape = [self.input_dimension_size]), dtype = tf.float32, name= 'Project_bias')

			# construct cells with the specific layer_num
			mulrnn_cell = tf.contrib.rnn.MultiRNNCell([RNN_cell(type = self.cell_type, hidden_size = self.hidden_size, keep_prob = lstm_keep_prob)	for _ in range(self.layer_num)])

			# initialize state to zero
			init_state = mulrnn_cell.zero_state(self.batch_size, dtype=tf.float32)
			state = init_state

			outputs = list()

			# makes cell run
			# outputs has list of 'num_steps' with each element's shape (batch_size, hidden_size)
			with tf.variable_scope("RNN"):
				for time_step in range(self.num_steps):
					if time_step > 0 : tf.get_variable_scope().reuse_variables()
					if time_step == 0 :
						(cell_output, state) = mulrnn_cell(input[:, time_step, :],state)
						outputs.append(cell_output)
					else:
						# comparison has shape (batch_size, self.input_dimension_size) with elements 1 (means missing) when equal or 0 (not missing) otherwise
						comparison = tf.equal( input[:, time_step, :], tf.constant( Missing_value ) )
						current_prediction_output = tf.matmul(outputs[time_step - 1], W) + bias
						#change the current_input, select current_prediction_output when 1 (missing) or use input when 0 (not missing)
						current_input = tf.where(comparison, current_prediction_output, input[:,time_step,:])
						(cell_output, state) = mulrnn_cell(current_input, state)
						outputs.append(cell_output)

			# label_target_hidden_output has the last_time_step of shape (batch_size, hidden_size)
			label_target_hidden_output = outputs[-1]

			# prediction_target_hidden_output has list of 'num_steps - 1' with each element's shape (batch_size, hidden_size)
			prediction_target_hidden_output = outputs[:-1]

			#unfolded outputs into the [batch, hidden_size * (numsteps-1)], and then reshape it into [batch * (numsteps-1), hidden_size]
			prediction_hidden_output = tf.reshape( tensor = tf.concat(values = prediction_target_hidden_output, axis = 1), shape = [-1, self.hidden_size] )

			# prediction has shape (batch * (numsteps - 1), self.input_dimension_size)
			prediction = tf.add(tf.matmul(prediction_hidden_output, W),bias,name='prediction')

			# reshape prediction_target and corresponding mask  into [batch * (numsteps-1), hidden_size]
			prediction_targets = tf.reshape(prediction_target,[-1, self.input_dimension_size])
			masks = tf.reshape( mask,[-1, self.input_dimension_size] )

			#  softmax for the label_prediction, label_logits has shape (batch_size, self.class_num)
			with tf.variable_scope('Softmax_layer'):
				label_logits = tf.contrib.layers.legacy_fully_connected(x = label_target_hidden_output , num_output_units = self.class_num)
				loss_classficiation = tf.nn.softmax_cross_entropy_with_logits(labels = label_target, logits = label_logits, name = 'loss_classficiation')

		# use mask to use the observer values for the loss_prediction
		with tf.name_scope("loss_prediction"):
			loss_prediction = tf.reduce_mean(tf.square( (prediction_targets - prediction) * masks )) / (self.batch_size)

		regularization_loss = 0.0
		for i in self.vars:
			regularization_loss += tf.nn.l2_loss(i)

		with tf.name_scope("loss_total"):
			loss =  loss_classficiation + self.lamda * loss_prediction + 1e-4 * regularization_loss

		# for get the classfication accuracy, label_predict has shape (batch_size, self.class_num)
		label_predict = tf.nn.softmax(label_logits, name='test_probab')
		correct_predictions = tf.equal(tf.argmax(label_predict,1), tf.argmax(label_target,1))
		accuracy = tf.cast(correct_predictions, tf.float32,name='accuracy')

		input_tensors = {
			'input' : input,
			'prediction_target' : prediction_target,
			'mask' : mask,
			'label_target' : label_target,
			'lstm_keep_prob' : lstm_keep_prob,
			'classfication_keep_prob' : classfication_keep_prob
		}

		loss_tensors = {
			'loss_prediction' : loss_prediction,
			'loss_classficiation' : loss_classficiation,
			'regularization_loss' : regularization_loss,
			'loss' : loss
		}


		return input_tensors, loss_tensors, accuracy, tf.reshape( prediction, [-1, (self.num_steps - 1)*self.input_dimension_size] ), tf.reshape( mask, [-1, (self.num_steps - 1)*self.input_dimension_size]), label_predict, tf.reshape( prediction_targets, [-1, (self.num_steps - 1)*self.input_dimension_size] ), label_target_hidden_output

	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]


class Discriminator(object):
    def __init__(self,config):
        self.name = "Discriminator"

    def __call__(self, x):
        with tf.variable_scope(self.name) as vs:
            x1 = tf.contrib.layers.legacy_fully_connected(x = x, num_output_units = x.shape[1], activation_fn= tf.nn.tanh)
            x2 = tf.contrib.layers.legacy_fully_connected(x = x1, num_output_units = int(x.shape[1])//2, activation_fn = tf.nn.tanh)
            predict_mask = tf.contrib.layers.legacy_fully_connected(x = x2, num_output_units = x.shape[1], activation_fn = tf.nn.sigmoid)
        return predict_mask

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES']=config.GPU
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True


    print ('Loading data && Transform data--------------------')
    print (config.train_data_filename)
    train_data, train_label = utils.load_data(config.train_data_filename)

    #For univariate
    config.num_steps = train_data.shape[1]
    config.input_dimension_size = 1

    train_label, num_classes = utils.transfer_labels(train_label)
    config.class_num = num_classes


    print ('Train Label:', np.unique(train_label))
    print ('Train data completed-------------')

    test_data, test_labels = utils.load_data(config.test_data_filename)

    test_label, test_classes = utils.transfer_labels(test_labels)
    print ('Test data completed-------------')

    with tf.Session(config = gpu_config) as sess:

        G = Generator(config = config)
        input_tensors, loss_tensors, accuracy, prediction, M, Label_predict, prediction_target, Last_hidden_output  = G.build_model()

        real_pre = prediction * (1 - M) + prediction_target * M
        real_pre = tf.reshape(real_pre,[config.batch_size,(config.num_steps-1)*config.input_dimension_size])
        D = Discriminator(config)
        predict_M = D (real_pre)

        predict_M = tf.reshape(predict_M,[-1,(config.num_steps-1)*config.input_dimension_size])

        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = predict_M, labels = M))
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = predict_M, labels = 1 - M) * (1-M))

        total_G_loss = loss_tensors['loss'] + config.lamda_D * G_loss

		#D_solver
        D_solver = tf.train.AdamOptimizer(config.learning_rate).minimize(D_loss, var_list = D.vars)

		#G_solver
        G_solver = tf.train.AdamOptimizer(config.learning_rate).minimize(total_G_loss, var_list = G.vars)

		#global_variables_initializer
        sess.run(tf.global_variables_initializer())

#------------------------------------------------train---------------------------------------------------
        Epoch = config.epoch

        for i in range(Epoch):
            total_loss = []
            total_batch_d_loss = []
            total_batch_g_loss = []
            total_train_accuracy = []


            print ('----------Epoch %d----------'%i)

            '''train'''
            for input, prediction_target, mask, label_target, _, batch_need_label in utils.next_batch(config.batch_size, train_data, train_label, True,config.input_dimension_size,config.num_steps, Trainable = True):
                for _ in range(config.D_epoch):
                    _ , batch_d_loss, p_M, real_M = sess.run([D_solver, D_loss, predict_M, M], feed_dict={input_tensors['input']: input, input_tensors['prediction_target']: prediction_target,
																														input_tensors['mask']: mask, input_tensors['label_target']: label_target,
																														input_tensors['lstm_keep_prob']: 1.0, input_tensors['classfication_keep_prob']: 1.0})
                total_batch_d_loss.append(batch_d_loss)
                for _ in range(config.G_epoch):
                    batch_loss, batch_g_loss, batch_accuracy, _, batch_train_Pre, batch_train_hidden = sess.run([loss_tensors['loss'], G_loss, accuracy, G_solver, prediction, Last_hidden_output], feed_dict={input_tensors['input']: input, input_tensors['prediction_target']: prediction_target,
																														input_tensors['mask']: mask, input_tensors['label_target']: label_target,
																														input_tensors['lstm_keep_prob']: 1.0, input_tensors['classfication_keep_prob']: 1.0})
                total_loss.append(batch_loss)
                total_batch_g_loss.append(batch_g_loss)
                total_train_accuracy.append(batch_accuracy)

            print ("Loss:",np.mean(total_loss),"Train acc:",np.mean(np.array(total_train_accuracy).reshape(-1)))



        '''test'''
        total_test_accuracy = []
        total_sample_num = 0
        total_Pre = []
        for input, prediction_target, mask, label_target, batch_size, batch_need_label in utils.next_batch(config.batch_size, test_data, test_label, True,config.input_dimension_size,config.num_steps, Trainable = False):
            total_sample_num += batch_size
            batch_accuracy, batch_Pre, batch_Label_predict, batch_test_hidden = sess.run([accuracy, prediction, Label_predict, Last_hidden_output], feed_dict={input_tensors['input']: input, input_tensors['prediction_target']: prediction_target,
																		input_tensors['mask']: mask, input_tensors['label_target']: label_target,
																		input_tensors['lstm_keep_prob']: 1.0, input_tensors['classfication_keep_prob']: 1.0})
            total_test_accuracy.append(batch_accuracy)
            total_Pre.append(batch_Pre)

        assert total_sample_num == test_data.shape[0]
        Test_acc = np.mean(np.array(total_test_accuracy).reshape(-1)[:total_sample_num])

        print('Test acc:',Test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',type=int,required=True)
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--lamda_D',type=float,required=True,help='coefficient that adjusts gradients propagated from discriminator')
    parser.add_argument('--G_epoch',type=int,required=True,help='frequency of updating AJRNN in an adversarial training epoch')
    parser.add_argument('--train_data_filename',type=str,required=True)
    parser.add_argument('--test_data_filename',type=str,required=True)

    parser.add_argument('--layer_num',type=int,required=False,default=1,help='number of layers of AJRNN')
    parser.add_argument('--hidden_size',type=int,required=False,default=100,help='number of hidden units of AJRNN')
    parser.add_argument('--learning_rate',type=float,required=False,default=1e-3)
    parser.add_argument('--cell_type',type=str,required=False,default='GRU',help='should be "GRU" or "LSTM" ')
    parser.add_argument('--lamda',type=float,required=False,default=1,help='coefficient that balances the prediction loss')
    parser.add_argument('--D_epoch',type=int,required=False,default=1,help='frequency of updating dicriminator in an adversarial training epoch')
    parser.add_argument('--GPU',type=str,required=False,default='0',help='GPU to use')

    config = parser.parse_args()
    main(config)
