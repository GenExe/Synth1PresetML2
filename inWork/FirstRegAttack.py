# Import Comet and create experiment
from comet_ml import Experiment

# Comet variables
API_KEY = "gKo0lOrCw6burIxXrjggq8OMY"
PROJECT = "MultiOutputNormed"
WORKSPACE = "PresetGen"
VSTPARAMETER = 25

#experiment = Experiment(api_key=API_KEY, project_name=PROJECT, workspace=WORKSPACE, log_code=True)
#experiment.set_code()
# Dependencies
import xml.etree.ElementTree as ET
import re
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
import tensorflow as tf
import keras as keras
import keras.layers as layers
import kapre as kapre
import joblib as joblib

# matplotlib.use('agg')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # no graka...
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

ACTUALEXNAME = "mfccs5Second_Env10_MultipleBatchMinMax" + datetime.now().strftime("%m%d%Y%H%M%S")
print(ACTUALEXNAME)
# Dependencies
SR = 22050


def getOneLinerMfcc(mfcc):
    new = np.mean(mfcc, axis=0)
    return new


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        SR = sample_rate


    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled


def getAudio(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        SR = sample_rate
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return audio


def Conv1D(SR=SR, DT=1.0):
    i = layers.Input(shape=(1, int(SR * DT)), name='input')
    x = kapre.Melspectrogram(n_dft=512, n_hop=160,
                             padding='same', sr=SR, n_mels=128,
                             fmin=0.0, fmax=SR / 2, power_melgram=2.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='melbands')(i)
    x = kapre.Normalization2D(str_axis='batch', name='batch_norm')(x)
    x = layers.Permute((2, 1, 3), name='permute')(x)
    x = kapre.TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='tanh'), name='td_conv_1d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_2d_1')(x)
    x = kapre.TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_2d_2')(x)
    x = kapre.TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_2d_3')(x)
    x = kapre.TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_2d_4')(x)
    x = kapre.TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_4')(x)
    x = layers.GlobalMaxPooling2D(name='global_max_pooling_2d')(x)
    x = layers.Dropout(rate=0.1, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=keras.regularizers.l2(0.001), name='dense')(x)
    o = layers.Dense(1, name='dense_output')(x)

    model1 = kapre.Model(inputs=i, outputs=o, name='1d_convolution')

    return model1


# Set the path to the full UrbanSound dataset
fulldatasetpath = "I://Synth1PresetTestFiles"

features = []

testserializer = []

dirFiles = os.listdir("I://Synth1PresetTestFiles")
dirFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
# mfccss = []
# mfccss = np.load("I://zeroCrossFromPreset_frame64hop16.dat", allow_pickle=True)
mfccss1 = joblib.load("I://mfccFromPreset_128fft256hop64_1.dat")
mfccss2 = joblib.load("I://mfccFromPreset_128fft256hop64_2.dat")
mfccss = np.append(mfccss1, mfccss2)[1::2]
del mfccss1, mfccss2

value = 0
for file in dirFiles:
    if "xml" in file:
        sampleNumber = int(file.split(".")[0][4:])
        tempName = file.split(".")[0] + ".wav"
        fileName = os.path.join(os.path.abspath(fulldatasetpath), tempName)
        tree = ET.parse(os.path.join(os.path.abspath(fulldatasetpath), file))
        root = tree.getroot()
        classlabels = []
        for x in range(99):
            classlabels.append(int(root[x + 2].attrib["presetValue"]))
        #        classLabel = int(root[value + 2].attrib["presetValue"])
        data = mfccss[sampleNumber]
        features.append([data, classlabels])
        value = value + 1
        # print("Appended " + fileName + " Class = " + str(classlabels) + " SampleNumber = " + str(sampleNumber))

del dirFiles
print("All Data Appended")
# np.array(mfccss).dump("mfccsFromPreset.dat")

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
del features
# Convert features and corresponding classification labels into numpy arrays
featureArray = np.array(featuresdf.feature.tolist())
valueArray = np.array(featuresdf.class_label.tolist())
# valueArray = tf.keras.utils.normalize(valueArray, axis=0, order=2)
# min_d = np.min(valueArray)
# max_d = np.max(valueArray)
# valueArray = (valueArray - min_d) / (max_d - min_d)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(valueArray)
valueArray = scaler.transform(valueArray)
del featuresdf
# Encode the classification labels
le = LabelEncoder()
# yy = to_categorical(le.fit_transform(valueArray))

# split the dataset
from sklearn.model_selection import train_test_split

del mfccss, tree, root

x_train, x_test, y_train, y_test = train_test_split(featureArray, valueArray, test_size=0.2, random_state=42)

num_labels = valueArray.shape
filter_size = 2

# Construct model
# model = Sequential([
#     Dense(16, activation=tf.nn.leaky_relu, input_shape=(40, 988,)),
#     Dense(32, activation=tf.nn.leaky_relu),
#     layers.GlobalMaxPool1D(),
#     Dense(64, activation=tf.nn.leaky_relu),
#     Dense(1)
# ])
del featureArray, valueArray
valueArray = []
mfccss = []
# model = Sequential([
#     Dense(16, activation=tf.nn.leaky_relu, input_shape=(128, 1723,)),
#     Dense(32, activation=tf.nn.leaky_relu),
#     layers.GlobalMaxPool1D(),
#     Dense(64, activation=tf.nn.leaky_relu),
#     Dense(99)
# ])
model = Sequential()
model.add(Dense(16, input_shape=(128, 1723,), activation="relu"))
# model.add(keras.layers.BatchNormalization())
# model.add(layers.Activation("relu"))
model.add(Dense(32, activation="relu"))
model.add(layers.GlobalMaxPool1D())
model.add(Dense(64, activation=tf.nn.leaky_relu))
model.add(Dense(99))

# model = Sequential([
#     Dense(16, activation="relu", input_shape=(128, 1723,)),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.4),
#     Dense(32, activation="relu"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.4),
#     layers.GlobalMaxPool1D(),
#     Dense(64, activation=tf.nn.leaky_relu),
#     Dense(99)
# ])

# model = Conv1D()


# model = Sequential([
#     Conv2D(64, (3, 3), activation=tf.nn.leaky_relu, input_shape=(40, 988,)),
#     MaxPool2D(64, (2, 2)),
#     Conv2D(64,  (3, 3), activation=tf.nn.leaky_relu),
#     MaxPool2D(64,  (2, 2)),
#     Conv2D(64,  (3, 3), activation=tf.nn.leaky_relu),
#     layers.Flatten(),
#     Dense(64, activation=tf.nn.leaky_relu),
#     Dense(1)
# ])

# model = Sequential([Dense(256, activation="tanh", input_dim=40, kernel_initializer="uniform"),
#                      Dense(256, activation="tanh", input_dim=40, kernel_initializer="uniform"),
#                      Dense(1, activation="linear", kernel_initializer="uniform")])
print(model.summary())
optimizer = optimizers.RMSprop(0.001)

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy', 'mae'])
# model = Sequential()
#
# model.add(Dense(256, input_shape=(40,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(num_labels))
# model.add(Activation('softmax'))
#
# # Compile the model
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

num_epochs = 100
num_batch_size = 32

start = datetime.now()

# model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# reduce_lr_callback = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

# my_callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=2),
#     tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
#     tf.keras.callbacks.TensorBoard(log_dir='./logs'),
# ]
print("start Training....")
history = model.fit(
    x_train, y_train, batch_size=32,
    epochs=num_epochs, validation_split=0.2, verbose=0) # validation_data=(x_test, y_test)

duration = datetime.now() - start
model.save(ACTUALEXNAME)
print(model.summary())

print("Training completed in time: ", duration)

print(ACTUALEXNAME)

score = model.evaluate(x_train, y_train, verbose=0)
print('train accuracy: {}'.format(score))
# experiment.log_metric("train_acc", score)

score = model.evaluate(x_test, y_test, verbose=0)
print('test accuracy: {}'.format(score))
# experiment.log_metric("val_acc", score)

# print("First 10 Value Predictions")
# exampleMfccs = x_test[:10]
#
# exampleresult = model.predict(exampleMfccs)
#
# for val in exampleresult:
#     print(val)
