﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿Mxnet-version batch hard triplet loss
---
Based on <In defense of triplet loss> (https://arxiv.org/abs/1703.07737) 
Based on some tricks from omoindrot's repository. (https://github.com/omoindrot/tensorflow-triplet-loss)

## Introduction
-  In this paper,authors propose a new format of triplet loss called batch hard. For more details of batch hard triplet loss, you can read <In defense of triplet loss> (https://arxiv.org/abs/1703.07737).
-  More efficient than triplet loss which proposed by facenet. More details in (https://arxiv.org/abs/1503.03832) 
-  Inplement the hard mining method and soft-margin
-  Can be used in many tasks. Firstly I code this to do some research on re-id and image retrieval tasks.
-  In the future maybe add batch all triplet loss. Compared to batch hard, sometimes it can make the experiment more efficient.

## Architecture
1. Using resnetV2 to get 128-dimension embeddings
1. Using triplet loss to train embeddings
1. the network is defined in resnet.py

## Requirements
The code has been tested with CUDA 8.0 and ubuntu 16.04.\
python3\
mxnet-cu80==1.3

how to train:\
See parsers in train.py. Then Set your dataset path and some params of based resnet network.
The network has been defined in resnet.py.Batch_hard.py now has been deprecated. 
































