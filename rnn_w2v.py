
import tensorflow as tf
import numpy as np
import gensim
from nltk import sent_tokenize, word_tokenize
import re
import json

w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
w2v_len = 300

def strip_junk(t):
    return re.sub(r'[^a-zA-Z\-\s]', u'', t, flags=re.UNICODE)

def featurize(text):
    '''
     featurize the text to train and target dataset
    '''

    words = text.lower().split()
    words = [strip_junk(w) for w in words]

    input_words = []
    output_word = []

    for i in range(0, len(words) - max_len):
        input_words.append([])
        for w in words[i:i+max_len]:
            if w in w2v_model:
                input_words[-1].append(w2v_model[w])
                print(w2v_model[w])
            else:
                input_words[-1].append(np.random.uniform(0, 1, w2v_len).asarray())
        if words[i+max_len] in w2v_model:
            output_word.append(w2v_model[words[i+max_len]])
        else:
            output_word[-1].append(np.random.uniform(0, 1, w2v_len).asarray())

    return input_words, output_word

class DataReader:
    def __init__(self, filename):
        self.file = open(filename, 'r')

    def get_next(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            line = self.file.readline()
            if line is not None:
                batch.append(json.loads(line)['text'])
            else:
                self.file.seek(0)

        return [featurize(t) for t in batch]

#set hyperparameters
max_len = 40
num_units = 128
learning_rate = 0.001
batch_size = 20
epoch = 60
temperature = 0.5

def run(train_data, target_data, unique_chars, len_unique_chars):
    '''
     main run function
    '''
    x = tf.placeholder("float", [None, max_len, len_unique_chars])
    y = tf.placeholder("float", [None, len_unique_chars])
    weight = tf.Variable(tf.random_normal([num_units, len_unique_chars]))
    bias = tf.Variable(tf.random_normal([len_unique_chars]))

    prediction = rnn(x, weight, bias, len_unique_chars)
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    num_batches = int(len(train_data)/batch_size)

    for i in range(epoch):
        print("----------- Epoch {0}/{1} -----------".format(i+1, epoch))
        count = 0
        for _ in range(num_batches):
            train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
            count += batch_size
            sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})

        #get on of training set as seed
        seed = train_batch[:1:]

        #to print the seed 40 characters
        seed_chars = ''
        for each in seed[0]:
                seed_chars += unique_chars[np.where(each == max(each))[0][0]]
        print("Seed:", seed_chars)

        #predict next 1000 characters
        for i in range(1000):
            if i > 0:
                remove_fist_char = seed[:,1:,:]
                seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
            predicted = sess.run([prediction], feed_dict = {x:seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            probabilities = sample(predicted)
            predicted_chars = unique_chars[np.argmax(probabilities)]
            seed_chars += predicted_chars
        print('Result:', seed_chars)
    sess.close()


def train_w2v(text):
    sentences = [word_tokenize(strip_junk(t)) for t in sent_tokenize(text)]
    my_w2v = gensim.models.Word2Vec(sentences, min_count=1)
    my_w2v.train(sentences, total_examples=len(sentences), epochs=10)
    my_w2v.save("my_w2v.bin")
    print(my_w2v.wv.most_similar(positive="brutal"))

if __name__ == "__main__":
    #get data from https://s3.amazonaws.com/text-datasets/nietzsche.txt
    reader = DataReader('dataset/review.json')
    for i in range(4):
        print(reader.get_next(3))
    #train_w2v(text)
    #run(train_data, target_data, unique_chars, len_unique_chars)
