## Distributed Tensorflow for Noobs

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
