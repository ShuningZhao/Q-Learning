import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import string
import os
import tarfile


batch_size = 50
num_steps = 4
learning_rate = 0.15
lstm_size = 50
#fc_units = 15

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    # extract
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
        with tarfile.open('reviews.tar.gz', "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data2/'))
    # get all files
    print("READING DATA")
    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,
                                        'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,
                                        'data2/neg/*')))
    print("Parsing %s files" % len(file_list))

    # read data from files
    for f in file_list:
        with open(f, "r", encoding = "utf-8") as openf:
            s = openf.read()

            # trans the input string into list of words
            no_punct = ''.join(c for c in s if c not in string.punctuation)
            tmp = no_punct.lower().split()
            
            # keep the first 40 only
            if len(tmp)>40:
                tmp = tmp[:40]

            if len(tmp)<40:
                for _ in range(40-len(tmp)):
                    tmp.append("UNK")

            # use the provided dict to trans the words into int
            for i in range(40):
                if tmp[i] in glove_dict:
                    tmp[i] = np.float32(glove_dict[tmp[i]])
                else:
                    tmp[i] = np.float32(glove_dict["UNK"])
            # append tmp list into the data list
            tmp = np.array(tmp,dtype=np.float32)
            data.append(tmp)
            
    data = np.array(data,dtype=np.float32)
    print("Data Load Finished!")
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and danamed "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119}
    """
    print("embeddings Load start!")
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    
    embeddings = []
    word_index_dict = {}

    vectors = ["0"]*50
    embeddings.append(vectors)
    word_index_dict["UNK"] = 0

    index = 1
    for line in data:
        vectors = line.split()
        word = vectors.pop(0)
        vectors = np.array([np.float32(i) for i in vectors],dtype=np.float32)
        
        embeddings.append(vectors)
        word_index_dict[word] = index        
        index += 1

    # print(len(embeddings))
    embeddings = np.array(embeddings,dtype=np.float32)
    print("embeddings Load Finished!")
    
    return embeddings, word_index_dict

def get_a_cell(dropout_keep_prob):
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_size,
                                            forget_bias=0.0,
                                            state_is_tuple=True)  ##, forget_bias=0.0
    return tf.contrib.rnn.DropoutWrapper(cell=lstmCell,
                                         output_keep_prob= dropout_keep_prob)

def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""
    # dropout_keep_prob
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    # input placeholder
    input_data = tf.placeholder(dtype=tf.int32, shape=[batch_size, 40],
                                name="input_data")
    print("input done!")

    # labels placeholder
    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2],
                                name="labels")
    print("labels done!")

    # embedding
##    init = tf.constant_initializer(glove_embeddings_arr)
##    ##  embedding = tf.Variable(glove_embeddings_arr)
##    embedding = tf.get_variable("embedding",
##                                shape = [400001, 50],
##                                initializer = init)
    inputs = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)
    print("embedding done!")

    # lstm cell
    cell = tf.contrib.rnn.MultiRNNCell([get_a_cell(dropout_keep_prob) for _ in range(num_steps)], state_is_tuple=True)
    print("cell done!")

    # initialize the state
    state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell,
                                             inputs,
                                             dtype = tf.float32,
                                             initial_state = state)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.gather(outputs, int(outputs.get_shape()[0])-1) 
    # outputs = tf.reshape(outputs,[-1,lstm_size])

##    softmax_w = tf.get_variable("softmax_w", [size, lstm_size], dtype=tf.float32)
##    softmax_b = tf.get_variable("softmax_b", [lstm_size], dtype=tf.float32)
##    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

    # weights = tf.truncated_normal_initializer(stddev=0.1)
    weights = tf.Variable(tf.truncated_normal([lstm_size, 2], stddev=0.1), dtype=tf.float32)
    # weights = tf.get_variable('w',[lstm_size, 40])
    # biases = tf.zeros_initializer()
    biases = tf.Variable(tf.constant(0.1, shape=[2]),dtype=tf.float32)
    # biases = tf.get_variable('b',[40],initializer=tf.constant_initializer(0.0))
    
    print("rnn done!")

    logits = tf.matmul(outputs,weights) + biases
    print("prediction done!")

    # estimate = tf.reshape(tf.argmax(labels, 1),[-1,1])
    # loss = tf.reduce_mean(tf.losses.mean_squared_error(logits,labels),name="loss")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels),name="loss")
    # loss = -tf.reduce_mean(labels*tf.log(logits))
    print("loss done!")

    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    print("optimizer done!")

    ## correct_pred = tf.equal(tf.cast(tf.round(prediction),tf.int64),labels)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="accuracy")
    print("accuracy done!")
    
##    # get lstm cell output
##    outputs, states = tensorflow.contrib.static_rnn(cell, input_data, dtype = tf.float32)
##
##    # look up embeddings for inputs
##    valid_embeddings = tf.nn.embedding_lookup(embeddings, input_data)
##
##    logits = tf.matmul(outputs[-1], w)
##
##    # prediction
##    prediction = tf.nn.softmax(logits)
##
##    # loss tensor
##    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labels))
##    
##    #construct the sgd optimizer
##    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
##
##    # accuracy tensor
##    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
##    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
