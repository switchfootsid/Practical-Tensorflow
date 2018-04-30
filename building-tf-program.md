### 1. Summary Writer

- Writing log files of the program.
- Make sure to create a Summary Writer only after defining the
  entire graph
- Useful for Tensorboard

```writer = tf.summary.FileWriter([logdir], [graph])```

- ```[graph]``` is the graph object you are working on. 
- This can be either the default graph ```tf.get_default_graph()```
- Or through sess.graph, which returns the graph the current session is handling. 
  Note this can only be invoked within a session. 

#### Approach 1:
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

#### Approach 2:
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	output = sess.run([x], feed_dict = {input_x: batch})


### 2. Defining Variables
- Constants live in the graph and are replicated where the
  graph is loaded
- Variables are stored separately, may live on a parameter server.
- Variables are updated during training (weights, biases)
- Capital "V" cos this is a class with many Ops

```weights = tf.get_variable(name, shape=None, dtype=None, intializer, regularizer, trainable)```

```W = tf.get_variable("big_matrix", shape=(784,10), initializer=tf.zeros_initializer())```

#### 3. Initializing Variables
- Every variable must be initialized before using it
- Another way, is to load a variable's value externally
  from a file
```
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# sess.run(tf.global_variables_initializer([a,b]))
```
### 4. Importing Data

#### 4a. The old way: ```placeholders``` and ```feed_dict```
- Old way" placeholders and feed_dict
```
a = tf.placeholder(tf.float32, shape[3]) # vector of 3 elements
b = tf.constant([5,5,5], tf.float32)
c = a + b
```
To calculate C, you MUST supply the value of all placeholders that connect to the output while running the Op 

```
with tf.Session() as sess:
	for batch in list_of_batches:
		out = sess.run(c, feed_dict={batch: [1,2,3]})
		print (out)
```
#### 4b. The New Way: Dataset API
- Placeholders often end up processing data in a single thread
  and slows down the entire program
- Tensorflow also offers queues as an alternative. But messy!
- Enter tf.Data! Pass numpy arrays as tf.data object

```
tf.data.TextLineDataset(filenames): each file in those
files becomes one entry. Its good for datasets whose
entries are delimited by newlines such as NMT or CSV. 

tf.data.FixedLengthRecordDataset(filenames): each data point
int the dataset is of the same length. CIFAR/ImageNet

tf.data.TFRecordDataset(filenames): TFRecord format
```

#### Approach 1:
```
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```
#### Approach 2:
```
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
```
#### Approach 3:
```
dataset = tf.data.FixedLengthRecordDataset([file1, file2, file3, ...])
oneshot_iterator = dataset.make_one_shot_iterator()
mult_iterator = dataset.make_initializer_iterator()
X,Y = iterator.get_next() 
```
