from comet_ml import Experiment
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import concatenate
from data import DataGrabber
import xml.etree.ElementTree as ET
import re
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
import keras as keras
import keras.layers as layers
import kapre as kapre
import os
from datetime import datetime
from keras.layers import Dense
from keras import optimizers



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable gpu for training

API_KEY = "gKo0lOrCw6burIxXrjggq8OMY"
PROJECT = "MultiOutputNormed"
WORKSPACE = "PresetGen"
VSTPARAMETER = 25

experiment = Experiment(api_key=API_KEY, project_name=PROJECT, workspace=WORKSPACE, log_code=True)
experiment.set_code()

ACTUALEXNAME = "mfccs5Second_Env10_MultipleBatchMinMax163264end2_" + datetime.now().strftime("%m%d%Y%H%M%S")
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

features_mfcc = []
features_env10 = []

testserializer = []

dirFiles = os.listdir("I://Synth1PresetTestFiles")
dirFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
amplitude10 = DataGrabber.getAmplitude10()

mfccss = DataGrabber.getMfccs5Seconds_128_256_64()

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
        data_mfcc = mfccss[sampleNumber]
        data_env10 = amplitude10[sampleNumber, 1]
        features_mfcc.append([data_mfcc, classlabels])
        features_env10.append([data_env10, classlabels])
        value = value + 1
        # print("Appended " + fileName + " Class = " + str(classlabels) + " SampleNumber = " + str(sampleNumber))

del dirFiles
print("All Data Appended")

# Convert into a Panda dataframe
features_mffc_new = pd.DataFrame(features_mfcc, columns=['feature', 'class_label'])
features_env10_new = pd.DataFrame(features_env10, columns=['feature', 'class_label'])


# Convert features and corresponding classification labels into numpy arrays
featureArray_mfcc = np.array(features_mffc_new.feature.tolist())
featureArray_env10 = np.array(features_env10_new.feature.tolist())
valueArray = np.array(features_mffc_new.class_label.tolist())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(valueArray)
valueArray = scaler.transform(valueArray)
# Encode the classification labels

# split the dataset
from sklearn.model_selection import train_test_split

del features_mfcc, features_env10
del mfccss, tree, root
x_train, x_test, y_train, y_test = train_test_split(featureArray_mfcc, valueArray, test_size=0.2, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(featureArray_env10, valueArray, test_size=0.2, random_state=42)

del featureArray_mfcc, featureArray_env10

features_mffc = 1

filter_size = 2


# define two sets of inputs
inputA = Input(shape=(128, 1723,))
inputB = Input(shape=(11025,))
# the first branch operates on the first input
x = Dense(16, activation="relu")(inputA)
x = Dense(32, activation="relu")(x)
x = layers.GlobalMaxPool1D()(x)
x = Dense(64, activation=tf.nn.leaky_relu)(x)
x = Model(inputs=inputA, outputs=x)

# the second branch opreates on the second input
y = Dense(16, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(64, activation=tf.nn.leaky_relu)(y)
y = Model(inputs=inputB, outputs=y)

# combine the output of the two branches
combined = concatenate([x.output, y.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(2, activation="relu")(combined)
z = Dense(99)(z)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)

print(model.summary())
optimizer = optimizers.RMSprop(0.001)

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy', 'mae'])

num_epochs = 100
num_batch_size = 32

start = datetime.now()

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ModelCheckpoint(filepath=ACTUALEXNAME + '.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.0001, verbose=1)
]
print("start Training....")
history = model.fit(
    [x_train, x_train2], y_train, batch_size=32,
    epochs=num_epochs, validation_data=([x_test, x_test2], y_test), verbose=0, callbacks=my_callbacks)  # validation_data=(x_test, y_test)

duration = datetime.now() - start
model.save(ACTUALEXNAME)
print(model.summary())

print("Training completed in time: ", duration)

print(ACTUALEXNAME)
