import tensorflow as tf
from tensorflow.keras import layers, models

class AlexNet(tf.keras.Model):
    def __init__(self, num_classes = 50):
        super(AlexNet, self).__init__()

        self.conv1 = layers.Conv2D(96, kernel_size = 11, strides = 4, padding = "same", activation = "relu")
        self.pool1 = layers.MaxPooling2D(pool_size = 3, strides = 2)

        self.conv2 = layers.Conv2D(256, kernel_size = 5, strides = 1, padding = "same", acitvation = "relu")
        self.pool2 = layers.MaxPooling2D(pool_size = 3, strides = 2)

        self.conv3 = layers.Conv2D(384, kernel_size = 3, strides = 1, padding = "same", activation = "relu")
        self.conv4 = layers.Conv2D(384, kernel_size = 3, strides = 1, padding = "same", activation = "relu")
        self.conv5 = layers.Conv2D(256, kernel_size = 3, strides = 1, padding = "same", activation = "relu")
        self.pool5 = layers.MaxPooling2D(pool_size = 3, stride = 2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation = "relu")
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(4096, activation = "relu")
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(num_classes, activation = "softmax")
    
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
