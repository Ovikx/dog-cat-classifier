import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model
import numpy as np
from image_preprocessor import ImagePreprocessor
from early_stopping import EarlyStopping

np.set_printoptions(suppress=True)

pet_types = {
    0 : 'cat',
    1 : 'dog',
}
pix = 224

es = EarlyStopping(depth=5, ignore=30, method='consistency')

ip = ImagePreprocessor(normalization=255, training_threshold=0.7, color_mode='RGB')
package = ip.preprocess_dirs(
    ['images/cat', 'images/dog'],
    [0, 1],
    True
)

train_features = package['TRAIN_IMAGES']
train_labels = package['TRAIN_LABELS']
test_features = package['TEST_IMAGES']
test_labels = package['TEST_LABELS']

train_ds = tf.data.Dataset.from_tensors((train_features, train_labels)).shuffle(10000)
test_ds = tf.data.Dataset.from_tensors((test_features, test_labels)).shuffle(10000)

class PetPredictor(Model):
    def __init__(self):
        super(PetPredictor, self).__init__()
        self.conv1_1 = Conv2D(64, (3,3), activation='relu', input_shape=(pix,pix,3))
        self.mp1 = MaxPool2D()
        self.conv2_1 = Conv2D(128, (3,3), activation='relu')
        self.mp2 = MaxPool2D()
        self.conv3_1 = Conv2D(256, (3,3), activation='relu')
        self.mp3 = MaxPool2D()
        self.conv4_1 = Conv2D(512, (3,3), activation='relu')
        self.mp4 = MaxPool2D()
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(1024, activation='relu')
        self.d3 = Dense(1, activation='sigmoid')
    
    def call(self, x):
        x = self.conv1_1(x)
        x = self.mp1(x)
        x = self.conv2_1(x)
        x = self.mp2(x)
        x = self.conv3_1(x)
        x = self.mp3(x)
        x = self.conv4_1(x)
        x = self.mp4(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

model = PetPredictor()

loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=0.001)

train_loss = keras.metrics.Mean()
train_accuracy = keras.metrics.BinaryAccuracy()

test_loss = keras.metrics.Mean()
test_accuracy = keras.metrics.BinaryAccuracy()

@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(features, labels):
    predictions = model(features, training=False)
    t_loss = loss_function(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

for epoch in range(200):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for features, label in train_ds:
        train_step(features, label)

    for features, label in test_ds:
        test_step(features, label)
    
    if es.check(test_loss.result()):
        print('BREAKING LOOP')
        break
    
    print(
        f'Epoch {epoch+1} || '
        f'Training Loss: {train_loss.result()}, '
        f'Training Accuracy: {train_accuracy.result()}, '
        f'Testing Loss: {test_loss.result()}, '
        f'Testing Accuracy: {test_accuracy.result()}'
    )

print('---------------EXTERNAL TESTING PREDICTIONS---------------\n0 is cat, 1 is dog')
print(f'Dog 1 : {model.predict(np.array(ip.file_to_array("images/external testing/dog1.jpg")))}')
print(f'Dog 2 : {model.predict(np.array(ip.file_to_array("images/external testing/dog2.jpg")))}')
print(f'Walter : {model.predict(np.array(ip.file_to_array("images/external testing/walter.jpg")))}')
print(f'Cat 1 : {model.predict(np.array(ip.file_to_array("images/external testing/cat1.jpg")))}')
print(f'Cat 2 : {model.predict(np.array(ip.file_to_array("images/external testing/cat2.jpg")))}')