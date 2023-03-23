import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Precision, Recall, CategoricalAccuracy
import pandas as pd
from keras.callbacks import CSVLogger, ModelCheckpoint
from dataset import val, train
from keras.applications.inception_resnet_v2 import InceptionResNetV2

# Define the number of classes in your dataset
num_classes = 2

# Load the InceptionResNetV2 model and remove the last layer
base_model = InceptionResNetV2(weights='imagenet', include_top=False)

x = base_model.output

# Add a global average pooling layer and a dense output layer
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create a new model with the Inception v2 base and the new output layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the weights of the base model so that they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with an appropriate loss function, optimizer, and evaluation metric

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[CategoricalAccuracy(), Precision(), Recall(), ])
model.summary()

#define callbacks
logdir = 'logs'
csv_logger = CSVLogger('training_history.csv')
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

# #Train the Model
history = model.fit(train, epochs=20, validation_data=val, callbacks=[csv_logger, checkpoint])

#save the history in a datframe
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history_df.csv', index=False)