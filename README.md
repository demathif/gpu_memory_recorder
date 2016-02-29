A simple tool to record the GPU memory usage by calling nvidia-smi in the background.

Tested with nvidia-smi version 352.55 on CentOS Linux release 7.1.1503.

I used this code to track how my theano code is using the GPU.

## How to use

```python
from utils.memory_recorder import gpu_memory_recorder
import numpy as np
import os


import theano
import theano.tensor as T

# the id of the process we're recording
current_process_id = os.getpid()

# the directory where the memory data will be saved
log_dir = 'logs'

if (not os.path.exists(log_dir)):
    os.makedirs(log_dir)

# the name of the file where the data will be saved, if None, a name will
# be generated automatically
mem_usage_filename = 'memory.txt'

# interval for probing the gpu (in seconds)
interval = 0.2

# gpu id
gpu_id = 1

# create the mem recorder object
mem_recorder = gpu_memory_recorder(gpu_id=gpu_id,
                                   process_id=current_process_id,
                                   log_dir=log_dir,
                                   log_filename=mem_usage_filename,
                                   recording_interval=interval)

# start recording
mem_recorder.start_recording()


# write some theano code
x = T.ftensor4()
y = T.ftensor4()

z = 2 * x + y

f = theano.function([x, y], z.mean())

# do some computation
for i in xrange(100):
    a = np.random.sample((i, 100, 100, 100)).astype('float32')
    b = np.random.sample((i, 100, 100, 100)).astype('float32')

    c = f(a, b)

    # we can generate the chart at any point after we started recording.
    # the 50 means it will use only the last 50 data points when
    # generating the chart. if no argument is given, it generates the 
    # chart for the whole recording period so far
    if (i % 25 == 0):
        mem_recorder.generate_chart(50)

# stop recording
mem_recorder.stop_recording()

# save chart in log folder
mem_recorder.generate_chart()

```

## Example output

![Chart](https://github.com/ppalasek/gpu_memory_recorder/blob/master/logs/memory.png?raw=true)
