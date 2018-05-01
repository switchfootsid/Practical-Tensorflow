# Distributed Training

## Data Parallelism:
- In *data parallelism*, the model is replicated on serveral workers (devices/machines). Each worker trains on a split of the mini-batch, computes the gradients and weight updates are synchronized between all works ensure a consitent graph is being trained. 
- This has the advantage of handling large I/O and making multiple passes of the data very quickly. 

## Parameter Server Architecture:
- All machines/nodes/devices are split between parameter servers and workers.
- Parameter server contains a replica of the entire graph (variables/nodes) that is broadcasted to each of the worker. 

In each iteration:  
1. **Each worker** reads its own split from the mini-batch and computes gradients . 
2. Worker sends the gradient updates to one (or more) parameter servers . 
3. The **parameter servers** aggregate all the graidents from the workers and *calculate a new model* . 
4. The *new model is now broadcast* to each of the workers . 

## Distributed Tensorflow for Noobs
Distributed TensorFlow applications consist of a cluster containing one or more parameter servers and workers. 
- Workers calculate gradients during training, they are typically placed on a GPU . 
- Parameter servers only need to aggregate gradients and broadcast updates, so they are typically placed on CPUs, not GPUs.
- GPU have a slow I/O (possibly due to DMA?), much faster for CPU . 

- A tensorflow cluser is a set of nodes that process the computation graph parallely.
- Each node runs a task
- Each task is defined by the network address

```
parameter_servers = ["localhost:2222"]
workers = ["localhost:2223", "localhost:2224", "localhost:2225"]
cluster = tf.train.ClusterSpec({"parameter_server": parameter_servers, "worker": workers})
```

- Each task must run a Tensorflow Server that allows it to do actual computations and communicate with other nodes in the cluster to facilitate parallelization. 
- **Workers** compute gradients 
- **Parameter servers** aggregates and keeps track of the current versions of the parameters.

### Between-graph Replication:
- Separate but an identical computation graph is built on each of the worker tasks.
- 
