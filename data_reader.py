import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

# Toy data
train_imgs = tf.constant(['train/img1.png', 'train/img2.png',
                          'train/img3.png', 'train/img4.png',
                          'train/img5.png', 'train/img6.png'])
train_labels = tf.constant([0, 0, 0, 1, 1, 1])

val_imgs = tf.constant(['val/img1.png', 'val/img2.png',
                        'val/img3.png', 'val/img4.png'])
val_labels = tf.constant([0, 0, 1, 1])

# create TensorFlow Dataset objects
tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))
val_data = Dataset.from_tensor_slices((val_imgs, val_labels))

# create TensorFlow Iterator object
iterator = Iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)
next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

with tf.Session() as sess:

    # initialize the iterator on the training data
    sess.run(training_init_op)

    # get each element of the training dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

    # initialize the iterator on the validation data
    sess.run(validation_init_op)

    # get each element of the validation dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break
