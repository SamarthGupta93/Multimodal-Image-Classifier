from keras.layers import Input, Dense, Flatten, Conv3D, Dropout, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
import json
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import keras.backend as K
from itertools import product


class Multimodal():

	def __init__(self, n_classes, nucleus_features, alpha=0.001):
		# nucleus_features - Non-image features dimensions
		self.n_classes = n_classes
		self.nucleus_features = nucleus_features
		self.lr = alpha
		self.callbacks = None

		# Define weights array between classes. This is used in weighted cross entropy function. Use it if necessary
		self.w_array = self.get_w_array()

	def save_weights(self,path):
		self.model.save_weights(path)

	def load_weights(self,path):
		self.model.load_weights(path)

	def fit_generator(self, train_generator, dev_generator, steps_per_epoch, validation_steps, epochs=10):
		# Fit the model with the generator
		history = self.model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, 
			callbacks=self.callbacks, validation_data=dev_generator, validation_steps=validation_steps, verbose=1)

		self.plot_training_graphs(history)	
		return history


	def fit_data(self, x_train, y_train):
		self.model.fit(x_train, y_train)


	def create_model(self):
		"""Create CNN Classifier"""

		input_volume = Input(shape=(12, 12, 12, 1), name='input_volume')
		# Did not normalize during preprocessing to save memory (floats take more space than uint8 datatype)
		#state_input_normalized = Lambda(lambda x: x / 255.0)(input_volume)
		X = Conv3D(4, kernel_size=(3, 3, 3), activation='relu', name='conv_layer1')(input_volume)
		X = BatchNormalization(name='BatchNorm_layer_1')(X)

		X = Conv3D(4, kernel_size=(3, 3, 3), activation='relu', name='conv_layer2')(X)
		X = BatchNormalization(name='BatchNorm_layer_2')(X)

		X = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', name='conv_layer3')(X)
		X = BatchNormalization(name='BatchNorm_layer_3')(X)

		X = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', name='conv_layer4')(X)
		X = BatchNormalization(name='BatchNorm_layer_4')(X)

		# Flatten the convolutional layer to connect to the subsequent fully connected layer
		X = Flatten(name='flatten_layer')(X)


		X = Dense(64, activation='relu', name='dense_layer1')(X)
		X = Dropout(0.2, name="dropout_layer_1")(X)

		image_features = Dense(7, activation='relu', name='image_features')(X)

		# Add nucleus fetures
		nucleus_input = Input(shape=(self.nucleus_features,), name="nucleus_input")
		multimodal_input = concatenate([image_features, nucleus_input])

		# Create a MLP for classification
		X = Dense(128, activation='relu', name='mlp_layer_1')(multimodal_input)
		X = Dropout(0.3, name="dropout_layer_2")(X)
		X = Dense(64, activation='relu', name='mlp_layer_2')(X)
		X = Dropout(0.2, name="dropout_layer_3")(X)

		y_hat = Dense(self.n_classes, activation='softmax', name='output_layer')(X)


		self.model = Model([input_volume,nucleus_input], y_hat)

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	def create_model_2(self):
		"""Create CNN Classifier"""

		input_volume = Input(shape=(12, 12, 12, 1), name='input_volume')

		X = Conv3D(4, kernel_size=(3, 3, 3), activation='relu', name='conv_layer1')(input_volume)
		X = BatchNormalization(name='BatchNorm_layer_1')(X)

		X = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', name='conv_layer2')(X)
		X = BatchNormalization(name='BatchNorm_layer_2')(X)

		X = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', name='conv_layer3')(X)
		X = BatchNormalization(name='BatchNorm_layer_3')(X)

		X = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', name='conv_layer4')(X)
		X = BatchNormalization(name='BatchNorm_layer_4')(X)

		# Flatten the convolutional layer to connect to the subsequent fully connected layer
		X = Flatten(name='flatten_layer')(X)

		X = Dense(20, activation='relu', name='fc1')(X)
		image_features = Dense(7, activation='relu', name='image_features')(X)

		# Add nucleus fetures
		nucleus_input = Input(shape=(self.nucleus_features,), name="nucleus_input")
		multimodal_input = concatenate([image_features, nucleus_input])

		# Create a MLP for classification
		X = Dense(64, activation='relu', name='mlp_layer_1')(multimodal_input)
		X = Dropout(0.3, name="dropout_layer_2")(X)

		y_hat = Dense(self.n_classes, activation='softmax', name='output_layer')(X)

		self.model = Model([input_volume,nucleus_input], y_hat)

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


	def create_model_3(self):
		"""Create CNN Classifier"""

		cnn_features = Input(shape=(20+self.nucleus_features,), name='input_volume')

		#image_features = Dense(7, activation='relu', name='image_features')(cnn_features)

		# Add nucleus fetures
		#nucleus_input = Input(shape=(self.nucleus_features,), name="nucleus_input")
		#multimodal_input = concatenate([image_features, nucleus_input])

		# Create a MLP for classification
		X = Dense(256, activation='relu', name='mlp_layer_1')(cnn_features)
		X = Dropout(0.4, name="dropout_layer_1")(X)

		#X = Dense(64, activation='relu', name='mlp_layer_2')(X)
		#X = Dropout(0.2, name="dropout_layer_2")(X)

		y_hat = Dense(self.n_classes, activation='softmax', name='output_layer')(X)

		self.model = Model(cnn_features, y_hat)
		optimizer = Adam(lr=self.lr)
		self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


	def create_model_4(self):
		"""Create CNN Classifier"""

		input_volume = Input(shape=(12, 12, 12, 1), name='input_volume')
		# Did not normalize during preprocessing to save memory (floats take more space than uint8 datatype)
		#state_input_normalized = Lambda(lambda x: x / 255.0)(input_volume)
		X = Conv3D(4, kernel_size=(3, 3, 3), activation='relu', name='conv_layer1')(input_volume)
		X = BatchNormalization(name='BatchNorm_layer_1')(X)

		X = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', name='conv_layer2')(X)
		X = BatchNormalization(name='BatchNorm_layer_2')(X)

		X = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', name='conv_layer3')(X)
		X = BatchNormalization(name='BatchNorm_layer_3')(X)

		X = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', name='conv_layer4')(X)
		X = BatchNormalization(name='BatchNorm_layer_4')(X)

		# Flatten the convolutional layer to connect to the subsequent fully connected layer
		X = Flatten(name='flatten_layer')(X)

		image_features = Dense(7, activation='relu', name='image_features')(X)

		# Add nucleus fetures
		nucleus_input = Input(shape=(self.nucleus_features,), name="nucleus_input")
		multimodal_input = concatenate([image_features, nucleus_input])

		# Create a MLP for classification
		X = Dense(64, activation='relu', name='mlp_layer_1')(multimodal_input)
		X = Dropout(0.3, name="dropout_layer_2")(X)
		X = Dense(64, activation='relu', name='mlp_layer_2')(X)
		X = Dropout(0.2, name="dropout_layer_3")(X)

		y_hat = Dense(self.n_classes, activation='softmax', name='output_layer')(X)

		self.model = Model([input_volume,nucleus_input], y_hat)

		self.model.compile(optimizer='adam', loss=self.weighted_crossentropy, metrics=['accuracy'])


	def predict(self, x_image, x_nucleus, batch_size):
		return self.model.predict([x_image, x_nucleus], batch_size)


	def load_checkpoint(self, checkpoint_path):
		try:
			self.model.load_weights(checkpoint_path)
		except Exception as error:
			print("File could not be found")
			print(error)


	def set_model_checkpoint(self, cp_path):
		"""Create Callback functions"""
		# During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.
		# This is the callback for writing checkpoints during training
		callback_checkpoint = ModelCheckpoint(filepath=cp_path, verbose=1, save_weights_only=True)
		self.callbacks = [callback_checkpoint]


	def save_model(self, filepath):
		model_json = self.model.to_json()
		with open(filepath, "w+") as json_file:
		    json_file.write(model_json)


	def plot_model(self, filepath):
		plot_model(self.model, to_file=filepath)


	def get_w_array(self):
		w_array = [[1,1,1,1,1.1,1,1.1,1,1,1],
				[1,1,1,1,1,1.1,1.1,1,1,1],
				[1,1,1,1.1,1,1,1.1,1,1,1],
				[1,1,1.2,1,1.1,1,1.2,1,1,1],
				[1,1,1,1,1,1.1,1.2,1,1,1],
				[1,1,1,1,1.2,1,1.2,1,1,1],
				[1,1,1,1,1.1,1.2,1.2,1,1,1],
				[1,1,1,1,1,1,1.2,1,1,1],
				[1,1,1,1,1,1,1.2,1,1,1],
				[1,1,1,1,1,1,1,1,1,1]]
		return np.array(w_array)

	def weighted_crossentropy(self, y_true, y_pred):
		nb_cl = self.n_classes
		final_mask = K.zeros_like(y_pred[:, 0])
		y_pred_max = K.max(y_pred, axis=1)
		y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
		y_pred_max_mat = K.equal(y_pred, y_pred_max)
		for c_p, c_t in product(range(nb_cl), range(nb_cl)):
			final_mask += (self.w_array[c_t, c_p] * K.cast(y_pred_max_mat[:, c_p] ,tf.float32) * K.cast(y_true[:, c_t],tf.float32))
		return K.categorical_crossentropy(y_pred, y_true) * final_mask



	def plot_training_graphs(self,history):
		# Plots accuracy and loss against epochs

		# Accuracy
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epochs')
		plt.legend(['Train', 'Dev'], loc='upper right')
		#plt.show()
		plt.savefig('model3_acc_20ep_'+str(self.lr)+'.png')

		plt.clf()
		plt.cla()
		# Loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epochs')
		plt.legend(['Train','Dev'], loc='upper right')
		#plt.show()
		plt.savefig('model3_loss_20ep_'+str(self.lr)+'.png')