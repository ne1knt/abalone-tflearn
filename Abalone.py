import tflearn
import numpy as np
import re
import csv
#input_file = 'abalone.csv'
r = csv.reader(open('abalone_not_processed.csv')) 
lines = [l for l in r]
#print("Data len:", lines)
#print("First records:", r[1][8])
# Preprocessing the csv file
def preprocess_csv(data):
    for i in range(1, len(data)):
        #print("currnet field:", (data[i][8]))
        #onvert age
        if float(data[i][8]) <= 8:
            data[i][8] = 0
        elif 9 <= float(data[i][8]) <= 10:
            data[i][8] = 1
        else:
            data[i][8] = 2
    return data

#preprocess data
data = preprocess_csv(lines)

writer = csv.writer(open('abalone.csv', 'w'))
writer.writerows(lines)



#Load CSV
from tflearn.data_utils import load_csv
data, labels = load_csv('abalone.csv', has_header=True, categorical_labels=True, n_classes=3)

# Preprocessing
def preprocess(data):
    for i in range(len(data)):
        #onvert 'sex' field
        data[i][0] = 1. if data[i][0] == 'F' else 0.
    return np.array(data, dtype=np.float32)

#preprocess data
data = preprocess(data)

#Neural network
net = tflearn.input_data(shape=[None, 8])
net = tflearn.fully_connected(net, 128, name='dense1', regularizer='L2', weight_decay=0.001 )
net = tflearn.fully_connected(net, 256)
sgd = tflearn.SGD(learning_rate=1.0, lr_decay=0.96, decay_step=500)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net, optimizer=sgd,
                     loss='categorical_crossentropy', name='target')


#training
#define model
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(data, labels, n_epoch=100, batch_size=16, show_metric=True)
# Manually save model
model.save("abalone.tfl")

#test results
first = [0, 0.385, 0.255, 0.1, 0.3175, 0.137, 0.068, 0.092]
second = [0, 0.39, 0.31, 0.085, 0.344, 0.181, 0.0695, 0.079]
third = [0, 0.71, 0.555, 0.195, 1.9485, 0.9455, 0.3765, 0.495]
#first = preprocess(first)
pred = model.predict([first])
print("Predicted age first:", pred[0][0])
print("Real age: 0")

pred = model.predict([second])
print("Predicted age:", pred[0][1])
print("Real age: 0")

pred = model.predict([third])
print("Predicted age:", pred[0][2])
print("Real age: 2")

