### Saving Models:

Persisting models on disk:
Any interaction with your filesystem to save TF data needs:
1. Saver object - save full graph or var_list
2. Session object - graph you want to load and device config placement

Saving and restoring operations and meta-data:
* Graph Ops and Variables 
* Meta-data like weights 
* Collections like hyper-parameters - LR, optimizer

```
# This function returns a Saver
# Load previously saved meta-graph in the default graph
saver = tf.train.import_meta_graph(‘results/model.ckpt-10.meta')

# Access the default graph
graph = tf.get_default_graph()
```
