


# This function returns a Saver
# Load previously saved meta-graph in the default graph
```
saver = tf.train.import_meta_graph(‘results/model.ckpt-10.meta')
```
# Access the default graph
```
graph = tf.get_default_graph()
```
