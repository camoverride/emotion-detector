import tensorflow as tf


pretrained_model = tf.keras.applications.MobileNet()
tf.saved_model.save(pretrained_model, "./mobilenet/1/")
