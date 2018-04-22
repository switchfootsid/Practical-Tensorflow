## Saving Models:

Persisting models on disk:
Any interaction with your filesystem to persist TF data needs the following two objects:
1. **Saver** - options to save the *full graph* by default or selected variables using the *var_list* argument
2. **Session object** - select the *graph* you want to load (default graph is current graph) and device placement using *config*

### Saving (and restoring) TF graphs involves saving the following:
  * Graph Ops and Variables 
  * Meta-data like weights 
  * Collections like hyper-parameters - learning rates, optimizer

The following code block shows how to import pre-existing checkpoints and access the graph
```
# This function returns a Saver
# Load previously saved meta-graph in the default graph
saver = tf.train.import_meta_graph(‘results/model.ckpt-10.meta')

# Access the default graph
graph = tf.get_default_graph()
```

### Protofbus:
To simplify things I like to think of protobuf files as *"Using JSON while development but when moving to production, protobufs gives you compression capabilities for leaner storage and efficiency"*
