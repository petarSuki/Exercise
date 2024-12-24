#Importovanje standardnih biblioteka
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

import torch

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

x = torch.randn(10, 10).to(device)


#Importovanje tensorflow biblioteka - FUNCTIONAL API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Ово значи да се могу подесити параметри да се постепено повећава коришћење GPU меморије током извршавања програма, како би се избегла грешка услед недостатка меморије.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

#os.makedirs(POS_PATH)
#os.makedirs(NEG_PATH)
#os.makedirs(ANC_PATH)

#Uncompress TAR GZ Labelled facces in the wild dataset
#!tar -xf /data/negative/lfw.tgz

"""
#Move Labelled faces in the wild images to the following repository data/negative
for directory in os.listdir('lfw'):
  for file in os.listdir(os.path.join('lfw', directory)):
    EX_PATH = os.path.join('lfw', directory, file)
    NEW_PATH = os.path.join(NEG_PATH, file)
    os.replace(EX_PATH, NEW_PATH)
"""
#Import uuid library to generate unique image names
import uuid

os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))

def data_aug(img):
  data = []
  for i in range(9):
    img = tf.image.stateless_random_brightness(img, max_delta = 0.02, seed=(1,2))
    img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,2))
    img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
    img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
    img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))

    data.append(img)
  return data

for file_name in os.listdir(os.path.join(ANC_PATH)):
  img_path = os.path.join(ANC_PATH, file_name)
  img = cv2.imread(img_path)
  augmented_images = data_aug(img)

  for image in augmented_images:
    cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

for file_name in os.listdir(os.path.join(POS_PATH)):
  img_path = os.path.join(POS_PATH, file_name)
  img = cv2.imread(img_path)
  augmented_images = data_aug(img)

  for image in augmented_images:
    cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())
    


anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(3000)
'''
dir_test = anchor.as_numpy_iterator()

print(dir_test.next())
'''
def preprocess(file_path):

  #Read in image from file path
  byte_img = tf.io.read_file(file_path)
  #Load in the image
  img = tf.io.decode_jpeg(byte_img)

  #Preprocessing steps - resizing the image to be 100x100x3
  img =  tf.image.resize(img, (100,100))
  #Scale image to be between 0 and 1
  img = img / 255.0

  #Return image
  return img

#(anchor, positive) -> 1,1,1,1,1
#(anchor, negative) -> 0,0,0,0,0

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)
'''
samples = data.as_numpy_iterator()

example = samples.next()

example
'''
def preprocess_twin(input_img, validation_img, label):
  return(preprocess(input_img), preprocess(validation_img), label)
'''
res = preprocess_twin(*example)

plt.imshow(res[1])

res[2]
'''
#Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

#Training partition (taking 70% of all data for training)
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

#Testing partition (skipping the 70% we took for training and then testing on the other 30% of data)
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

def make_embedding():
  inp = Input(shape=(100,100,3), name = 'input_image')

  #First block
  c1 = Conv2D(64, (10,10), activation = 'relu')(inp)
  m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

  #Second block
  c2 = Conv2D(128,(7,7), activation = 'relu')(m1)
  m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

  #Third block
  c3 = Conv2D(128, (4,4), activation = 'relu')(m2)
  m3 = MaxPooling2D(64, (2,2), padding = 'same')(c3)

  #Final embedding block
  c4 = Conv2D(256, (4,4), activation = 'relu')(m3)
  f1 = Flatten()(c4)
  d1 = Dense(4096, activation = 'sigmoid')(f1)

  return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding = make_embedding()

embedding.summary()

class L1Dist(Layer):

  def __init__(self, **kwargs):
    super().__init__()

  def call(self, input_embedding, validation_embedding):
    # Debugging: Print types
    print(f"Input embedding type: {type(input_embedding)}")
    print(f"Validation embedding type: {type(validation_embedding)}")

    if isinstance(input_embedding, list):
        input_embedding = input_embedding[0]
    if isinstance(validation_embedding, list):
        validation_embedding = validation_embedding[0]

    input_embedding = tf.convert_to_tensor(input_embedding)
    validation_embedding = tf.convert_to_tensor(validation_embedding)

    return tf.math.abs(input_embedding - validation_embedding)

l1 = L1Dist()

input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))

inp_embedding = embedding(input_image)  # Tensor of shape (batch_size, 4096)
val_embedding = embedding(validation_image)    # Tensor of shape (batch_size, 4096)

siamese_layer = L1Dist()

distances = siamese_layer(inp_embedding, val_embedding)

classifier = Dense(1, activation = 'sigmoid')(distances)

classifier

siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNewtork')

siamese_network.summary()

#Making the Siamese Model
def make_siamese_model():

  #Anchor iamge input in the network
  input_image = Input(name='input_image', shape=(100,100,3))

  #Validation image in the network
  validation_image = Input(name='validation_img', shape=(100,100,3))

  #Combine siamese distance components
  siamese_layer = L1Dist()
  siamese_layer._name = 'distance'
  distances = siamese_layer(embedding(input_image), embedding(validation_image))

  #Clasification layer
  classifier = Dense(1, activation='sigmoid')(distances)

  return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()

siamese_model.summary()

binary_cross_loss = tf.losses.BinaryCrossentropy()

opt = tf.keras.optimizers.Adam(0.0001)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model = siamese_model)

test_batch = train_data.as_numpy_iterator()

batch_1 = test_batch.next()

batch_1[2]

X = batch_1[:2]

y = batch_1[2]

y

@tf.function
def train_step(batch):

  #Record all of operations
  with tf.GradientTape() as tape:
    #Get anchor and positive/negative image
    X = batch[:2]
    #Get label
    y = batch[2]

    #Forward pass
    yhat = siamese_model(X, training=True)
    #Calculate loss
    loss = binary_cross_loss(y, yhat)
  print(loss)

  #Calculate gradients
  grad = tape.gradient(loss, siamese_model.trainable_variables)

  #Calculate updated weights and apply to siamese model
  opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

  #Return loss
  return loss

#Import metric calculations
from tensorflow.keras.metrics import Precision, Recall

def train(data, EPOCHS):
  #Loop through epochs
  for epoch in range(1, EPOCHS+1):
    print('\n Epoch {}/{}'.format(epoch, EPOCHS))
    progbar = tf.keras.utils.Progbar(len(data))

    #Creating a metric object
    r = Recall()
    p = Precision()

    #Loop through each batch
    for idx, batch in enumerate(data):
      #Run train step here
      loss = train_step(batch)
      yhat = siamese_model.predict(batch[:2])
      r.update_state(batch[2], yhat)
      p.update_state(batch[2], yhat)
      progbar.update(idx+1)
    print(loss.numpy(), r.result().numpy(), p.result().numpy())

    #Save checkpoints
    if epoch % 10 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50

train(train_data, EPOCHS)

# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall

# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input, test_val])

# Post processing the results
[1 if prediction > 0.5 else 0 for prediction in y_hat ]

y_true

# Creating a metric object
m = Recall()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()

# Creating a metric object
m = Precision()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
  yhat = siamese_model.predict([test_input, test_val])
  r.update_state(y_true, yhat)
  p.update_state(y_true,yhat)

print(r.result().numpy(), p.result().numpy())

# Set plot size
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[0])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

# Renders cleanly
plt.show()

# Save weights
siamese_model.save('siamesemodel_final.h5')


