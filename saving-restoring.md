# Saving, Restoring and Exporting:

**Persisting models on disk:**  
Any interaction with your filesystem to persist TF data needs the following two objects:
1. **Saver** - options to save the *full graph* by default or selected variables using the *var_list* argument
2. **Session object** - select the *graph* you want to load (default graph is current graph) and device placement using *config*

### Saving (and restoring) TF graphs involves saving the following:
  * Graph Ops and Variables 
  * Meta-data like weights 
  * Collections like hyper-parameters - learning rates, optimizer

Assume we're working with a hypothetical architecture called *SundayNet*. On a Sunday afternoon, you halted the training of *SundayNet* and decided to go out for a cold lassi (Indian sweetened milk). 

Monday morning you decide to get back to *SundayNet*. The following code block shows how to resume training from where you left off by loading the book-keeping file called *checkpoints* and access the graph:
```
# This function returns a Saver
# Load previously saved meta-graph in the default graph
saver = tf.train.import_meta_graph(‘results/model.ckpt-10.meta')

# Access the default graph
graph = tf.get_default_graph()
```
Now, you can split open the model and access TF ops, variables and tensors:
```
global_step_tensor = graph.get_tensor_by_name('loss/global_step:0')
train_op = graph.get_operation_by_name('loss/train_op')
hyperparameters = tf.get_collection('hyperparameters')
```

### Aside on Protobufs:
Protobufs or pb files facilitate storage, versioning and updates of models in production. To simplify things I like to think of protobuf files as Google's version of *"Using JSON while development but adding compression capabilities for leaner storage and efficiency for production"*

Snap back to reality, when *saving* TF data using the **Saver object** you mainly get the following 4 types of files:
 * The **checkpoint file** is just a bookkeeping file that you can use in combination of high-level helper for loading different time saved chkp files.
 * The **.meta file** holds the compressed Protobufs graph of your model and all the metadata associated (collections, learning rate, operations, etc.) *so you can retrain it*
 * The **.index file** holds an immutable key-value table linking a serialised tensor name and where to find its data in the chkp.data files
 * The **.data files** hold the data (weights) itself (this one is usually quite big in size). There can be many data files because they can be sharded and/or created on multiple timestep while training.
 
#### Tensorflow trick for PB:
All operations dealing with protobufs in tensorflow have this “_def” suffix that indicate “protocol buffer definition”.For example:   
1. **Load the protobufs of a saved graph**: ```tf.import_graph_def.``` 
2. **Get current graph as a protobuf:** ```Graph.as_graph_def()```
 
## Three ways to save and restoring models for inference:  

1. **Saver object** for saving checkpoints and restoring **within a session**. 
2. **Writing a protobuf file** to disk and loading pb file for inference using ```tf.GraphDef() and import_graph_def```
3. **Saver object** for saving checkpoints, **restoring** and **converting to pb** file for inference.
4. **Exporting using SavedModel**

### 1. Saving and Restoring:

### 2. Exporting PB File or Frozen Graphs:

### 3. Converting checkpoints to PB file: 

### 4. SavedModel: Using TF Serving:

Tensorflow's preferred approach for saving and restoring a model is to use the ```SavedModel, builder, and loader``` functionality. This actually wraps the ```Saver class``` in order to provide a higher-level serialization, which is more suitable for production purposes.
