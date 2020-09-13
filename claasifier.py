import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler

#print(os.listdir("./training"))

"""fig, axs = plt.subplots(11, 5, figsize = (32,32))
count = 0
# for every class in the dataset
for i in os.listdir('./training'):
  # get the list of all images that belong to a particular class
  train_class = os.listdir(os.path.join('training', i))
  
  # plot 5 images per class
  for j in range(5):
    img = os.path.join('training', i, train_class[j])
    axs[count][j].title.set_text(i)
    axs[count][j].imshow(PIL.Image.open(img))  
  count += 1

print(fig.tight_layout())"""

#Image Augmentation
train_datagen=ImageDataGenerator(rescale=1./255,
                                zoom_range=0.2,
                                horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

#Data Generator for train,valid,test
train_generator=train_datagen.flow_from_directory(
        'training',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

validation_generator=test_datagen.flow_from_directory(
        'validation',
        target_size=(256,256),
        batch_size=32,
        class_mode='categorical')

test_generator=test_datagen.flow_from_directory(
        'evaluation',
        target_size=(256,256), 
        batch_size=32,
        class_mode='categorical')

#load the model
basemodel=InceptionResNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(256,256,3)))

#print(basemodel.summary())

#freeze the weights so that they remain same
#basemodel.trainable=False
basemodel.trainable=False

# Add classification head to the model
headmodel=basemodel.output
headmodel=GlobalAveragePooling2D(name='global_average_pool')(headmodel)
headmodel=Flatten(name='flatten')(headmodel)
headmodel=Dense(256,activation='relu',name='dense_1')(headmodel)
headmodel=Dropout(0.3)(headmodel)
headmodel=Dense(128,activation='relu',name='dense_2')(headmodel)
headmodel=Dropout(0.3)(headmodel)
headmodel=Dense(11,activation='softmax',name='dense_3')(headmodel)
model = Model(inputs = basemodel.input, outputs = headmodel)

# compile the model
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01,momentum=0.9),metrics=['accuracy'])

# using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="model_tuned2.h5", verbose=1, save_best_only=True)

#history = model.fit(train_generator,
#                    steps_per_epoch= train_generator.n // 32, 
#                   epochs = 5,
#                    validation_data= validation_generator,
#                    validation_steps= validation_generator.n // 32,
#                    callbacks=[checkpointer, earlystopping])

#select the mode of optimization
mode = "Speed" 
if mode == 'Storage':
        optimization = tf.lite.Optimize.OPTIMIZE_FOR_SIZE
elif mode == 'Speed':
        optimization = tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
else:
        optimization = tf.lite.Optimize.DEFAULT

model = tf.keras.models.load_model('model_tuned2.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()


tflite_model_file = 'converted_model.tflite'
with open(tflite_model_file, "wb") as f:
        f.write(tflite_model)


