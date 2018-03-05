#!/usr/bin/env python

import tensorflow as tf
import numpy 
import sys

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

###
### Parse options
###
from optparse import OptionParser
usage="%prog [options] <train-data-file> <train-label-file> <test-data-file> <test-label-file> <test-predict-label-file> <train-predict-label-file>"
parser = OptionParser(usage)

parser.add_option('--feat-dim', dest='featDim', 
                   help='Feature dimension [default: %default]', 
                   default=20, type='int');

parser.add_option('--num-class', dest='numClass', 
                   help='The number of classes [default: %default]', 
                   default=5, type='int');

(o,args) = parser.parse_args()
if len(args) != 6 : 
  parser.print_help()
  sys.exit(1)
  
#(trDataFile, trLabelFile, tsDataFile, tsLabelFile) = map(int,args);
(trDataFile, trLabelFile, tsDataFile, tsLabelFile, tsPredFile, trPredFile) = args

miniBatch = 100
nEpoch = 5000
lr = 0.0001
valRate = 10 # validation data rate (valRate %) 

hidNode_map = {
  'hid1':1000,
  'hid2':1000,
  'hid3':1000,
  'hid4':500,
  'hid5':500
}
lastLayer = 'hid3'
hidNode1 = hidNode_map['hid1']
hidNode2 = hidNode_map['hid2']
hidNode3 = hidNode_map['hid3']
hidNode4 = hidNode_map['hid4']
hidNode5 = hidNode_map['hid5']
lastNode = hidNode_map[lastLayer]

### End parse options 

print 'DNN Experiment...'
print 'numFeat : %d, numClass : %d' %(o.featDim,o.numClass)
print 'miniBatch : %d, nEpoch : %d, lr : %f' %(miniBatch,nEpoch,lr)
print 'hidden node - hid1 : %d, hid2 : %d, hid3 : %d, hid4 : %d, hid5 : %d' %(hidNode1,hidNode2,hidNode3,hidNode4,hidNode5)
print 'last layer : %s' %(lastLayer)


### Define function ###

def dense_to_one_hot(labels_dense, num_classes=5):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel() - 1] = 1
  return labels_one_hot

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def trainShuff(trainData,trainLabel):
    
    length=trainData.shape[0]
    rng=numpy.random.RandomState(0517)
    train_ind=range(0,length)
    rng.shuffle(train_ind)

    RanTrainData=trainData[train_ind,]
    RanTrainLabel=trainLabel[train_ind,]

    return RanTrainData,RanTrainLabel

def next_batch(pre_index, batch_size, data_size):
  """Return the next `batch_size` examples from this data set."""
  start = pre_index
  check_index = start + batch_size
  if  check_index > data_size:
    # Start next epoch
    start = 0

  end = start + batch_size
  return start, end
### End define function ###


### Read file of train/test data and label

d1 = open(trDataFile)
buf = d1.read()
oriTrData = numpy.fromstring(buf, dtype=numpy.float32, sep=' ')
oriTrData = oriTrData.reshape(len(oriTrData)/o.featDim,o.featDim)
d1.close()

l1 = open(trLabelFile)
buf = l1.read()
oriTrLabel = numpy.fromstring(buf, dtype=numpy.uint8, sep=' ')
oriTrLabel = dense_to_one_hot(oriTrLabel,o.numClass)
l1.close()

d2 = open(tsDataFile)
buf = d2.read()
tsData = numpy.fromstring(buf, dtype=numpy.float32, sep=' ')
tsData = tsData.reshape(len(tsData)/o.featDim,o.featDim)
d2.close()

l2 = open(tsLabelFile)
buf = l2.read()
tsLabel = numpy.fromstring(buf, dtype=numpy.uint8, sep=' ')
tsLabel = dense_to_one_hot(tsLabel,o.numClass)
l2.close()

# for check one-hot label
#trOnehotLabFile = "%s_oh" %(trLabelFile)
#wl1 = open(trOnehotLabFile,'w')

#for l in range(len(oriTrLabel)):
#  for m in range(o.numClass):
#    buf1 = "%f " %(oriTrLabel[l][m]) 
#    wl1.write(buf1)
#  wl1.write('\n')
#wl1.close()
# end of check


oriTrData, oriTrLabel=trainShuff(oriTrData, oriTrLabel) # shuffling

valInx = oriTrData.shape[0]/valRate
valData = oriTrData[0:valInx]
valLabel = oriTrLabel[0:valInx]

trData = oriTrData[valInx+1:oriTrData.shape[0]]
trLabel = oriTrLabel[valInx+1:oriTrLabel.shape[0]]

print 'validation rate : %d percent , index : 0~%d' %(valRate,valInx)

totalBatch = trData.shape[0]/miniBatch


### Main script ###

x = tf.placeholder("float", [None, o.featDim])
y_ = tf.placeholder("float", [None, o.numClass])
keepProb = tf.placeholder("float")

# tf.sigmoid / tf.nn.relu

W1 = weight_variable([o.featDim, hidNode1])
b1 = bias_variable([hidNode1])
#h1 = tf.nn.relu(tf.matmul(x,W1) + b1)
h1 = tf.sigmoid(tf.matmul(x,W1) + b1)

h1Drop = tf.nn.dropout(h1, keepProb)

W2 = weight_variable([hidNode1, hidNode2])
b2 = bias_variable([hidNode2])
#h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)
h2 = tf.sigmoid(tf.matmul(h1Drop,W2) + b2)

h2Drop = tf.nn.dropout(h2, keepProb)

W3 = weight_variable([hidNode2, hidNode3])
b3 = bias_variable([hidNode3])
#h3 = tf.nn.relu(tf.matmul(h2,W3) + b3)
h3 = tf.sigmoid(tf.matmul(h2Drop,W3) + b3)

h3Drop = tf.nn.dropout(h3, keepProb)

W4 = weight_variable([hidNode3, hidNode4])
b4 = bias_variable([hidNode4])
#h4 = tf.nn.relu(tf.matmul(h3,W4) + b4)
h4 = tf.sigmoid(tf.matmul(h3Drop,W4) + b4)

h4Drop = tf.nn.dropout(h4, keepProb)

W5 = weight_variable([hidNode4, hidNode5])
b5 = bias_variable([hidNode5])
#h5 = tf.nn.relu(tf.matmul(h4,W5) + b5)
h5 = tf.sigmoid(tf.matmul(h4Drop,W5) + b5)

h5Drop = tf.nn.dropout(h5, keepProb)

hidLayer_map = {
  'hid1':h1Drop,
  'hid2':h2Drop,
  'hid3':h3Drop,
  'hid4':h4Drop,
  'hid5':h5Drop
}

W_last = weight_variable([lastNode, o.numClass])
b_last = bias_variable([o.numClass])

y = tf.nn.softmax(tf.matmul(hidLayer_map[lastLayer],W_last) + b_last)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

epoch=0
testAcc=0
shuf=100
while(epoch<nEpoch):
  epoch=epoch+1
  if epoch%shuf==0:	
    trData, trLabel=trainShuff(trData, trLabel)
    #print 'shuffling...'

  for train_index in xrange(totalBatch):
    feed_dict={
      x: trData[train_index*miniBatch: (train_index+1)*miniBatch],
      y_: trLabel[train_index*miniBatch: (train_index+1)*miniBatch],
      keepProb: 0.4
      }
    sess.run(train_step,feed_dict)

  if epoch%100==0:
    pred_val = sess.run(y,feed_dict={x: valData, y_:valLabel, keepProb:1.0})
    val_acc = sess.run(accuracy, feed_dict={y: pred_val, y_: valLabel})
    print '%d epoch validation accuracy : %2.1f ' %(epoch,(val_acc*100))


#pre_index = 0
#for i in range(100000):
#  beg_index, end_index = next_batch(pre_index, miniBatch, len(trData))
#  pre_index = end_index
#  feed_dict = {x: trData[beg_index:end_index], y_: trLabel[beg_index:end_index]}
  #print(beg_index,end_index)
#  sess.run(train_step, feed_dict)


pred_train = sess.run(y,feed_dict={x: trData, y_:trLabel, keepProb:1.0})

wp2 = open(trPredFile,'w')

for l in range(len(pred_train)):
  for m in range(o.numClass):
    buf2 = "%f " %(pred_train[l][m]) 
    wp2.write(buf2)
  wp2.write('\n')
wp2.close()
train_acc = sess.run(accuracy, feed_dict={y: pred_train, y_: trLabel})
print 'train accuracy : %2.1f ' %(train_acc*100)

pred_test = sess.run(y,feed_dict={x: tsData, y_:tsLabel, keepProb:1.0})

wp = open(tsPredFile,'w')

for j in range(len(pred_test)):
  for k in range(o.numClass):
    buf = "%f " %(pred_test[j][k]) 
    wp.write(buf)
  wp.write('\n')
wp.close()

test_acc = sess.run(accuracy, feed_dict={y: pred_test, y_: tsLabel})
print 'test accuracy : %2.1f ' %(test_acc*100)



