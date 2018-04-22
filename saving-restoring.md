## Saving Models:

Persisting models on disk:
Any interaction with your filesystem to persist TF data needs the following two objects:
1. **Saver** - options to save the *full graph* by default or selected variables using the *var_list* argument
2. **Session object** - select the *graph* you want to load (default graph is current graph) and device placement using *config*

### Saving (and restoring) TF graphs involves saving the following:
  * Graph Ops and Variables 
  * Meta-data like weights 
  * Collections like hyper-parameters - learning rates, optimizer

Assume we're working with a hypothetical architecture called *SundayNet*. On a Sunday afternoon, you halted the training of *SundayNet* and decided to go out for a cold lassi (Indian sweetened milk). 

Monday morning you decide to get back to your *SundayNet*. The following code block shows how to resume training from where you left off by loading the book-keeping file called *checkpoints* and access the graph:
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

### Protofbus:
To simplify things I like to think of protobuf files as *"Using JSON while development but when moving to production, protobufs gives you compression capabilities for leaner storage and efficiency"*
