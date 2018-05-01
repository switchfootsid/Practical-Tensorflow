# Distributed Training

## Data Parallelism (Between-Graph Replication)
- In *data parallelism*, the model is replicated on serveral workers (devices/machines). Each worker trains on a split of the mini-batch, computes the gradients and weight updates are synchronized between all works ensure a consitent graph is being trained. 
- This has the advantage of handling large I/O and making multiple passes of the data very quickly. 

## Parameter Server Architecture:
- All machines/nodes/devices are split between parameter servers and workers. One of the workers is the *master*.  
- The *master worker* coordinates model training, initializing, counts the number of training steps monitors the session, saves-restores model checkpoints to recover from failures.  
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

### Note on Practical Distributed TF:  
As someone who wants to focus on developing (learning) new deep architectures, I must warn you that managing parameters-workers in Tensorflow code is a little messy. It requires the developer to write a lot of control statements and tracking IP addresses.  

One disadvantage of Distributed TensorFlow, is that you have to **manage the starting and stopping of servers explicitly**. This means keeping track of the host endpoints (IP addresses and ports) of all your TensorFlow servers in your program and starting and stopping those servers manually.

```
parameter_servers = ["localhost:2222"]
workers = ["localhost:2223", "localhost:2224", "localhost:2225"]
cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})
if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
...
```
It is error-prone and impractical to create a ```ClusterSpec``` using host endpoints (IP address and port number). Use instead a cluster manager such as Kubernetes to reduce the complexity of configuring and launching TensorFlow applications. The main options are either a cloud managed solution. I *highly recommend* using **Google Cloud Machine Learning Engine.** By wrapping your code in high-level **TF Esimtators API**, distributed training is a breeeze on and requires **no code changes** on CMLE.
