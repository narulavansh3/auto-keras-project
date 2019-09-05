import numpy as np
#from autokeras.metric import Accuracy
from autokeras.supervised import Supervised
import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.optim as optim
import spacy
from tqdm import tqdm
import os
import csv

import time


#this one works using pickleload
def tsv_transform(documents,labels,temporary_tsv_file):
	"""
	receives two numpy n dimentional numpy lists that have the same size,
	one is made out of strings and the other made out of labels_placeholder
	it receives a string temporary_tsv_file
	it transforms the array into a tsv file with each line in the format [sentiment]	[sentence]
	"""
	myFile = open(temporary_tsv_file, 'w')
	with myFile:
		myFile.write('sentiment'+'\t'+ 'sentence\n')
		for i in range(len(documents)):
			myFile.write(str(labels[i])+'\t'+ str(documents[i]))
			if (i == len(documents)-1):
				print(" saved %s documents into %s" %(str(i+1), temporary_tsv_file))


def tsv_remover(temporary_tsv_file):
	"""
	remove the tsv file in the directory of the string passed
	"""
	os.remove(temporary_tsv_file)

class Model(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
		"""
		We initialize RNN model with the sizes givenself.
		"""
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		self.fc = nn.Linear(hidden_dim*2, output_dim)
		self.dropout = nn.Dropout(dropout)


	def forward(self, x):
		"""
		given an input x, we will do a forward pass with our function and return the final prediction.
		"""
		#x = [sent len, batch size]
		embedded = self.dropout(self.embedding(x))

		#embedded = [sent len, batch size, emb dim]
		output, (hidden, cell) = self.rnn(embedded)
		hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

		return self.fc(hidden.squeeze(0))


class TextClassifier(Supervised):
	def __init__(self):
		"""
		Text Classifier is a supervised autokeras class
		it gets initialized and with no hyperparameters
		self.final_hyperparams contains a list of the final hyperparameters we decide to go with

		self.models will keep a list of models trained with their respective statistics
		we will fill this variables after fit is called.autokeras
		"""
		super().__init__(verbose=False)
		self.labels = None
		#hyperparameters
		self.batch_size = None
		self.num_epochs = None
		self.final_hyperparams = None

		#word embedding from torchtext
		self.sentence = data.Field(tokenize='spacy')
		self.sentiment = data.LabelField(dtype=torch.float)
		self.fields = [('sentiment',self.sentiment),('sentence',self.sentence)]

		#model and collection of models
		self.model = None
		self.optimizer = None
		self.test_model = None #this model is just for running predictions
		self.training_predict = False
		self.loss_function = None
		self.models = []
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.final_test_acc = None


		#if 1, do grid search
		self.search_mode = 2

	def evaluate(self, x_test, y_test):
		"""
		evaluate is called assuming fit and final fi had already been called
		self.model will contain our best model
		if defined, will return the accuracy of our best model

		will also set self.final_test_acc to the accuracy on this dataset
		will transform the input array of data into tsv format and then call test on the function
		"""
		self.model.eval()

		temporary_tsv_file = 'sample/corpus.tsv'
		tsv_transform(x_test,y_test,temporary_tsv_file)

		test_data = data.TabularDataset(path = temporary_tsv_file, format = 'tsv',fields = self.fields,skip_header = True)
		#self.sentence.build_vocab(test_data, max_size=self.final_hyperparams[3], vectors="glove.6B.100d")
		#self.sentiment.build_vocab(test_data)
		tsv_remover(temporary_tsv_file)
		test_iterator = data.BucketIterator(test_data, sort_key=lambda x: len(x.sentence),batch_size=self.final_hyperparams[1],device=self.device)
		#test_iterar, valid_iterator = data.BucketIterator.splits((train_data, valid_data), sort_key=lambda x: len(x.sentence),batch_size=batch_size,device=self.device)
		print("starting preds")
		test_acc = self.test(self.model,test_iterator)
		#print("Test accuracy is : ", test_acc)
		self.final_test_acc = test_acc
		return test_acc

	def test(self, model, iterator):#,sentence,sentiment):
		"""
		test receives a model and iterator assuming that the data inside iterator is consistentwith the vocabulary created for our sentences
		will iterate through as much data as possible in batches
		retuns final accuracy
		"""
		final_acc = 0
		counter =0
		with torch.no_grad():
			#print(type(iterator),len(iterator))
			for batch in iterator:
				#print(batch)
				counter +=1
				if counter%3000 == 0 : print("3K over")
				#run before calling predict
				self.test_model = model
				self.training_predict = True

				prediction = self.predict(batch.sentence)
				self.test_model = None
				self.training_predict = False


				acc = self.acc_metric(prediction, batch.sentiment)
				final_acc += acc.item()
			final_acc = final_acc/len(iterator)
		return final_acc


	def predict(self, x_test):
		"""
		returns a prediction given a models
		can only be called given that x_test is compatible with the vocab given

		CANNOT RETURN A PREDICTION IF RETRAIN IS NOT RUN, vocabulary is not saved for each model due to RAM constrains
		"""
		if self.training_predict:
			model = self.test_model
		else:
			model = self.model
		prediction = model(x_test).squeeze(-1)
		return prediction

	def train_model(self, train_data = None, valid_data = None, m_hidden_dim = 256, batch_size = 4, num_epochs = 2,vocab_max_size = 25000, m_n_layers = 2,  m_dropout = 0.5  ):
		"""
		receives a test and train data in torchtext dataset format. Also receives a set of hyperparameters
		creates a vocabulary and saves it in self.sentence.vocab_size

		trains a model for a number of epochs
		will return a model and its accuracy given some hyperparameters

		"""
		m_embedding_dim = 100
		m_output_dim = 1
		m_bidirectional = True



		self.sentence.build_vocab(train_data, max_size=vocab_max_size, vectors="glove.6B.100d")
		self.sentiment.build_vocab(train_data)
		print("vocab built")

		model = Model(len(self.sentence.vocab),
						embedding_dim = m_embedding_dim,
						hidden_dim = m_hidden_dim,
						output_dim = m_output_dim,
						n_layers = m_n_layers,
						bidirectional = m_bidirectional,
						dropout = m_dropout)

		#self.model = model
		optimizer = optim.Adam(model.parameters())
		loss_function = nn.BCEWithLogitsLoss()
		loss_function.to(self.device)

		statistic = {"training":[],"validation":[]}
		model.to(self.device)
		print("trainining model with device: ", str(self.device) )
		train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data), sort_key=lambda x: len(x.sentence),batch_size=batch_size,device=self.device)
		for epoch in range(num_epochs):
			print("Epoch ", epoch+1)
			print("training")
			train_loss , train_acc = self.train(model,train_iterator,optimizer,loss_function)
			statistic["training"].append((train_loss , train_acc))
			print("train loss: ", str(train_loss)," train accuracy: ", str(train_acc))
			print("testing")
			valid_acc = self.test(model,valid_iterator)#,sentence,sentiment)
			print("validation accuracy: ", str(valid_acc))
			statistic["validation"].append(valid_acc)

		return model, valid_acc*100


	def fit(self, x_train=None, y_train=None, time_limit=None):
		"""
		fit implements our NAS
		inputs: list of sentences in numpy  format with their arrays: xtrain and y trains

		creates vocab,
		creates several array of searchable hyperparameters

		implements either grid, random or greedy search dependent on the status of self.search mode (0,1,2 respectively)
		each search will append a list of accuracies + hyperparameters to self.models[]


		after search is dome, the best values are stored to self.final_hyperparams for future refernece by final_fit
		outputs: none
		"""
		initial_time = time.clock()


		SEED = 1234

		tsv_dict_path = 'sample/corpus.tsv'
		tsv_transform(x_train,y_train,tsv_dict_path)
		train_data = data.TabularDataset(path = tsv_dict_path, format = 'tsv',fields = self.fields,skip_header = True)
		tsv_remover(tsv_dict_path)
		train_data, valid_data = train_data.split(random_state=random.seed(SEED))


		def print_stats(stats_dict):
			return str(stats_dict["valid_acc"]) , str(stats_dict["hidden_dimensions"]) ,str(stats_dict["batch_size"]),str(stats_dict["num_epochs"]),str(stats_dict["vocab_size"]),str(stats_dict["layers"]),str(stats_dict["dropout"]),str(stats_dict["time"])


		def get_optimal_values():
			if (len(self.models) <1):
				print("you have no saved models")
				return

			max_acc = 0
			best_model = 0
			for i in range(len(self.models)):
				acc= self.models[i]["valid_acc"]
				if (acc > max_acc):
					max_acc = acc
					best_model = i
				#else
					#self.models.remove(i) #pop it if its minimal
			return best_model #return the index of the best models

		p_hidden_dim = [64,128,256]
		p_batch_size = [16,32,64	]
		p_num_epoch = [2, 3, 5]###############change to 3 4 5
		p_vocab_size = [20000,25000]#,25000,30000]
		p_m_n_layers = [2,3,4]#,2,3]
		p_m_dropout=[0.4,0.5, 0.6]#,0.5,0.6]
		possibilities = len(p_hidden_dim)*len( p_batch_size) *len( p_num_epoch)*len( p_vocab_size) *len(p_m_n_layers ) * len (p_m_dropout)

		#----------------grid search----------------------------------------------
		if(self.search_mode==0):
			class BreakIt(Exception): pass

			print("doing grid search")
			try:
				for hid in p_hidden_dim:
					for bat in p_batch_size:
						for epc in p_num_epoch:
							for voc in p_vocab_size:
								for lay in p_m_n_layers:
									for drp in p_m_dropout:
										time_elapsed =(time.clock()-initial_time)
										if(time_elapsed > time_limit):
											print("exceeded time, computing optimal")
											raise BreakIt
										print("time elapsed: ",time_elapsed)
										time_model_start = time.clock()
										model, valid_acc = self.train_model( train_data, valid_data, hid, bat,epc, voc, lay, drp)
										search_results = {"model":model, "valid_acc":valid_acc,"hidden_dimensions":hid,"batch_size":bat,"num_epochs": epc,"vocab_size":voc,"layers": lay,"dropout":drp,"time":(time.clock()- time_model_start)}
										self.models.append(search_results)
										print( "Obtained accuracy %s, for model with hidden dimensions %s, batchsize %s, epochs %s, vocab size %s, layers %s  dropout %s, time %s"%(print_stats(search_results)))
										#print("This model took %s" %str(time.clock()- time_epoch))
			except BreakIt:
				pass

		#-------------------------------------------grid search----------------------------------------------

		def get_ramdom_params():
			hid = p_hidden_dim[random.randint(0,(len(p_hidden_dim) - 1)) ]
			bat = p_batch_size[random.randint(0,(len(p_batch_size) - 1)) ]
			epc = p_num_epoch[random.randint(0,(len(p_num_epoch) - 1))]
			voc = p_vocab_size[random.randint(0,(len(p_vocab_size) - 1))]
			lay = p_m_n_layers[random.randint(0,(len(p_m_n_layers) - 1))]
			drp  = p_m_dropout[random.randint(0,(len(p_m_dropout) - 1))]
			return hid, bat, epc, voc, lay, drp

		#------------------------random search--------------------------------------------------------------------
		if(self.search_mode==1):
			print("doing random search")
			time_elapsed = time.clock() - initial_time
			first_model = True
			model_counter = 0

			while (time.clock() - initial_time< time_limit):
				#in the case we exhaust our entire search space randomly
				if (model_counter >= possibilities):
					break
				model_counter = model_counter + 1
				hid, bat, epc, voc, lay, drp = get_ramdom_params()
				#check if we have already selected these parameters
				if not first_model:
					computed = True
					while (computed):
						for s in self.models:
							if (hid is s["hidden_dimensions"]) and (bat is s["batch_size"]) and ( epc is s["num_epochs"]) and ( voc is s["vocab_size"]) and ( lay is s["layers"]) and ( drp is s["dropout"]):
								computed = True
								hid, bat, epc, voc, lay, drp = get_ramdom_params()
								break
							else:
								 computed = False
				first_model = False
				#time_model = time.clock() - initial_time
				print("time elapsed: ",time.clock() - initial_time)
				time_model = time.clock()
				print ("Started new train  for hid %s, bat %s, epc %s, voc %s, lay %s, drp %s"%(str(hid), str(bat), str( epc), str( voc), str( lay), str( drp )))
				model, valid_acc = self.train_model( train_data, valid_data, hid, bat,epc, voc, lay, drp)
				search_results = {"model":model, "valid_acc":valid_acc,"hidden_dimensions":hid,"batch_size":bat,"num_epochs": epc,"vocab_size":voc,"layers": lay,"dropout":drp,"time":(time.clock()- time_model)}
				self.models.append(search_results)
				print( "Obtained accuracy %s, for model with hidden dimensions %s, batchsize %s, epochs %s, vocab size %s, layers %s  dropout %s, time %s"%(print_stats(search_results)))
		#------------------------random search--------------------------------------------------------------------

		#-------------------------------------------greedy search----------------------------------------------
		if(self.search_mode==2):
			print("doing Greedy search")
			class BreakIt(Exception): pass
			try:
				def get_pos (array):
					if len(array)>1:
						return len(array)
					else :
						return 0


				possibilities =  get_pos(p_hidden_dim) +get_pos( p_batch_size) + get_pos( p_num_epoch)+get_pos( p_vocab_size) + get_pos(p_m_n_layers ) + get_pos (p_m_dropout)
				#initialize hyperparams at a random startin point
				hid, bat, epc, voc, lay, drp = get_ramdom_params()
				#params_dict = [p_hidden_dim , p_batch_size , p_num_epoch , p_vocab_size , p_m_n_layers,	p_m_dropout]#,0.5,0.6]}


				if len(p_m_n_layers) >1:
					for layers  in p_m_n_layers:
						time_elapsed =(time.clock()-initial_time)
						if(time_elapsed > time_limit):
							print("exceeded time, computing optimal")
							raise BreakIt
						time_model = time.clock() - initial_time
						print("time elapsed: ",time.clock()-initial_time)
						time_model_start = time.clock()
						print ("Started new train  for hid %s, bat %s, epc %s, voc %s, lay %s, drp %s"%(str(hid), str(bat), str( epc), str( voc), str( layers), str( drp )))
						model, valid_acc = self.train_model( train_data, valid_data, hid, bat, epc, voc, layers, drp)
						search_results = {"model":model, "valid_acc":valid_acc,"hidden_dimensions":hid,"batch_size":bat,"num_epochs": epc,"vocab_size":voc,"layers": layers,"dropout":drp,"time":(time.clock()- time_model)}
						self.models.append(search_results)
						print( "Obtained accuracy %s, for model with hidden dimensions %s, batchsize %s, epochs %s, vocab size %s, layers %s  dropout %s, time %s"%(print_stats(search_results)))
					max_acc = 0
					for i in self.models[-len(p_m_n_layers):]:
						if i["valid_acc"] > max_acc:
							max_acc = i["valid_acc"]
							lay = i["layers"]
#------------------------------------------------------------------------------------------------------------------------------------
				if len(p_hidden_dim) >1:
					for hidden in p_hidden_dim:
						time_elapsed =(time.clock()-initial_time)
						if(time_elapsed > time_limit):
							print("exceeded time, computing optimal")
							raise BreakIt
						time_model = time.clock() - initial_time
						print("time elapsed: ",time.clock()-initial_time)
						time_model_start = time.clock()
						print ("Started new train  for hid %s, bat %s, epc %s, voc %s, lay %s, drp %s"%(str(hidden), str(bat), str( epc), str( voc), str( lay), str( drp )))
						model, valid_acc = self.train_model( train_data, valid_data, hidden, bat,epc, voc, lay, drp)
						search_results = {"model":model, "valid_acc":valid_acc,"hidden_dimensions":hidden,"batch_size":bat,"num_epochs": epc,"vocab_size":voc,"layers": lay,"dropout":drp,"time":(time.clock()- time_model)}
						self.models.append(search_results)
						print( "Obtained accuracy %s, for model with hidden dimensions %s, batchsize %s, epochs %s, vocab size %s, layers %s  dropout %s, time %s"%(print_stats(search_results)))
					max_acc = 0
					for i in self.models[-len(p_hidden_dim):]:
						if i["valid_acc"] > max_acc:
							max_acc = i["valid_acc"]
							hid = i["hidden_dimensions"]
#------------------------------------------------------------------------------------------------------------------------------------
				if len(p_vocab_size) >1:
					for vocab  in p_vocab_size:
						time_elapsed =(time.clock()-initial_time)
						if(time_elapsed > time_limit):
							print("exceeded time, computing optimal")
							raise BreakIt
						time_model = time.clock() - initial_time
						print("time elapsed: ",time.clock()-initial_time)
						time_model_start = time.clock()
						print ("Started new train  for hid %s, bat %s, epc %s, voc %s, lay %s, drp %s"%(str(hid), str(bat), str( epc), str( vocab), str( lay), str( drp )))
						model, valid_acc = self.train_model( train_data, valid_data, hid, bat, epc, vocab, lay, drp)
						search_results = {"model":model, "valid_acc":valid_acc,"hidden_dimensions":hid,"batch_size":bat,"num_epochs": epc,"vocab_size":vocab,"layers": lay,"dropout":drp,"time":(time.clock()- time_model)}
						self.models.append(search_results)
						print( "Obtained accuracy %s, for model with hidden dimensions %s, batchsize %s, epochs %s, vocab size %s, layers %s  dropout %s, time %s"%(print_stats(search_results)))
					max_acc = 0
					for i in self.models[-len(p_vocab_size):]:
						if i["valid_acc"] > max_acc:
							max_acc = i["valid_acc"]
							voc = i["vocab_size"]
#------------------------------------------------------------------------------------------------------------------------------------
				if len(p_batch_size) >1:
					for batch in  p_batch_size:
						time_elapsed =(time.clock()-initial_time)
						if(time_elapsed > time_limit):
							print("exceeded time, computing optimal")
							raise BreakIt
						#time_model = time.clock() - initial_time
						print("time elapsed: ",time_elapsed)
						time_model = time.clock()
						print ("Started new train  for hid %s, bat %s, epc %s, voc %s, lay %s, drp %s"%(str(hid), str(batch), str( epc), str( voc), str( lay), str( drp )))
						model, valid_acc = self.train_model( train_data, valid_data, hid, batch, epc, voc, lay, drp)
						search_results = {"model":model, "valid_acc":valid_acc,"hidden_dimensions":hid,"batch_size":batch,"num_epochs": epc,"vocab_size":voc,"layers": lay,"dropout":drp,"time":(time.clock()- time_model)}
						self.models.append(search_results)
						print( "Obtained accuracy %s, for model with hidden dimensions %s, batchsize %s, epochs %s, vocab size %s, layers %s  dropout %s, time %s"%(print_stats(search_results)))
					max_acc = 0
					for i in self.models[-len(p_batch_size):]:
						if i["valid_acc"] > max_acc:
							max_acc = i["valid_acc"]
							bat = i["batch_size"]
#------------------------------------------------------------------------------------------------------------------------------------
				if len(p_num_epoch) >1:
					for epoch in  p_num_epoch:
						time_elapsed =(time.clock()-initial_time)
						if(time_elapsed > time_limit):
							print("exceeded time, computing optimal")
							raise BreakIt
						time_model = time.clock() - initial_time
						print("time elapsed: ",time.clock()-initial_time)
						time_model_start = time.clock()
						print ("Started new train  for hid %s, bat %s, epc %s, voc %s, lay %s, drp %s"%(str(hid), str(bat), str( epoch), str( voc), str( lay), str( drp )))
						model, valid_acc = self.train_model( train_data, valid_data, hid, bat, epoch, voc, lay, drp)
						search_results = {"model":model, "valid_acc":valid_acc,"hidden_dimensions":hid,"batch_size":bat,"num_epochs": epoch,"vocab_size":voc,"layers": lay,"dropout":drp,"time":(time.clock()- time_model)}
						self.models.append(search_results)
						print( "Obtained accuracy %s, for model with hidden dimensions %s, batchsize %s, epochs %s, vocab size %s, layers %s  dropout %s, time %s"%(print_stats(search_results)))
					max_acc = 0
					for i in self.models[-len(p_num_epoch):]:
						if i["valid_acc"] > max_acc:
							max_acc = i["valid_acc"]
							epc = i["num_epochs"]



#------------------------------------------------------------------------------------------------------------------------------------
				if len(p_m_dropout) >1:
					for dropout  in p_m_dropout:
						time_elapsed =(time.clock()-initial_time)
						if(time_elapsed > time_limit):
							print("exceeded time, computing optimal")
							raise BreakIt
						time_model = time.clock() - initial_time
						print("time elapsed: ",time.clock()-initial_time)
						time_model_start = time.clock()
						print ("Started new train  for hid %s, bat %s, epc %s, voc %s, lay %s, drp %s"%(str(hid), str(bat), str( epc), str( voc), str( lay), str( dropout )))
						model, valid_acc = self.train_model( train_data, valid_data, hid, bat, epc, voc, lay, dropout)
						search_results = {"model":model, "valid_acc":valid_acc,"hidden_dimensions":hid,"batch_size":bat,"num_epochs": epc,"vocab_size":voc,"layers": lay,"dropout":dropout,"time":(time.clock()- time_model)}
						self.models.append(search_results)
						print( "Obtained accuracy %s, for model with hidden dimensions %s, batchsize %s, epochs %s, vocab size %s, layers %s  dropout %s, time %s"%(print_stats(search_results)))
					max_acc = 0
					for i in self.models[-len(p_m_dropout):]:
						if i["valid_acc"] > max_acc:
							max_acc = i["valid_acc"]
							drp = i["dropout"]


			except BreakIt:
				pass
		#-------------------------------------------greedy search----------------------------------------------

		print ("exhausted %d/%d possibilities at time %f" %(len(self.models), possibilities,(time.clock()-initial_time)))
		results_file = open("results.csv", 'w')
		with results_file:
			results_file.write(" accuracy , hidden dimensions , batchsize , epochs , vocab size , layers ,  dropout, time \n")
			for stats in self.models:
				results_file.write(" %s, %s, %s, %s, %s, %s, %s, %s\n"%print_stats(stats))
		self.model =self.models[get_optimal_values()]["model"]

		opt_hid = self.models[get_optimal_values()]["hidden_dimensions"]
		opt_bat = self.models[get_optimal_values()]["batch_size"]
		opt_epc = self.models[get_optimal_values()]["num_epochs"]
		opt_voc= self.models[get_optimal_values()]["vocab_size"]
		opt_lay= self.models[get_optimal_values()]["layers"]
		opt_drp = self.models[get_optimal_values()]["dropout"]
		self.final_hyperparams = (opt_hid,opt_bat,opt_epc,opt_voc,opt_lay,opt_drp)

		print("------------------------------------------------------------------------------------------------------------------------")
		print ( "Best model found with, hidden dimensions %s, batchsize %s, epochs %s, vocab size %s, layers %s, dropout %s"%(str(opt_hid),str(opt_bat),str(opt_epc),str(opt_voc), str(opt_lay),str(opt_drp)))
		print("------------------------------------------------------------------------------------------------------------------------")


	@property
	def metric(self):

		return self.final_test_acc

	def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=True):
		"""
		receives a train and test set, each defined as 2 numpy list in the format of sentence, labels
		if retrain: use our best hyperparameters (found by fit)

		retrain with the entire dataset (using ytest only for validation)

		"""
		if (retrain):

			#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			SEED = 1234

			tsv_train = 'sample/train.tsv'
			tsv_test = 'sample/test.tsv'
			tsv_transform(x_train,y_train,tsv_train)
			tsv_transform(x_test,y_test,tsv_test)

			train_data = data.TabularDataset(path = tsv_train, format = 'tsv',fields = self.fields,skip_header = True)
			valid_data = data.TabularDataset(path = tsv_test, format = 'tsv',fields = self.fields,skip_header = True)

			tsv_remover(tsv_train)
			tsv_remover(tsv_test)
			param = self.final_hyperparams
			self.model, accuracy = self.train_model( train_data, valid_data, param[0], param[1],param[2], param[3], param[4], param[5])
			print("Model saved. The accuracy of our newly trained model is ", accuracy)
		else:
			print("RUN WITH RETRAIN ENABLED")
		pass

	def acc_metric(self, prediction,ground_truth):
		"""
		takes output of the model and labels and and it defines an accuracy
		"""
		#round predictions to the closest integer
		rounded_preds = torch.round(torch.sigmoid(prediction))
		correct = (rounded_preds == ground_truth).float() #convert into float for division
		acc = correct.sum()/len(correct)
		return acc

	def train(self, model, iterator, optimizer, criterion):
		"""
		takes model specific hyperparameters and trains
		"""
		epoch_loss = 0
		epoch_acc = 0
		model.train()
		counter = 0

		for batch in tqdm(iterator):
			optimizer.zero_grad()
			counter+=1
		#	if counter%1000 == 0 : print("1k over")
			predictions = model(batch.sentence).squeeze(1)
			loss = criterion(predictions, batch.sentiment)
			acc = self.acc_metric(predictions, batch.sentiment)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			epoch_acc += acc.item()

		return epoch_loss / len(iterator), epoch_acc / len(iterator)
