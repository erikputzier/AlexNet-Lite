import tensorflow as tf
from tensorflow.keras import layers

class AlexNet(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.conv1 = layers.Conv2D(4, kernel_size=3, strides=1, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.dropout_conv1 = layers.Dropout(0.2)
        self.pool1 = layers.MaxPooling2D(pool_size=2, strides=2)

        self.conv2 = layers.Conv2D(8, kernel_size=3, strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.dropout_conv2 = layers.Dropout(0.2)
        self.pool2 = layers.MaxPooling2D(pool_size=2, strides=2)

        self.conv3 = layers.Conv2D(16, kernel_size=3, strides=1, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.dropout_conv3 = layers.Dropout(0.3)

        self.conv4 = layers.Conv2D(16, kernel_size = 3, strides = 1, padding = "same")
        self.bn4 = layers.BatchNormalization()
        self.dropout_conv4 = layers.Dropout(0.3)

        self.conv5 = layers.Conv2D(8, kernel_size=3, strides=1, padding="same")
        self.bn5 = layers.BatchNormalization()
        self.dropout_conv5 = layers.Dropout(0.3)
        self.pool5 = layers.MaxPooling2D(pool_size=2, strides=2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(64, activation="relu")
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(32, activation="relu")
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(num_classes, activation="softmax")
    
    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.dropout_conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.dropout_conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.dropout_conv3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        x = self.dropout_conv4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = tf.nn.relu(x)
        x = self.dropout_conv5(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x