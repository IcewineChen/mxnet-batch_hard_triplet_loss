#!/usr/bin/env python
# -*- coding = utf-8 -*-
import mxnet as mx
import batch_hard
import numpy as np
import cv2
import os
import argparse
import multiprocessing
import random
import logging
import resnet
import dataset

log_path = './logs'
logfile = './logs/train.log'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_stream = logging.StreamHandler()
file_stream = logging.FileHandler(logfile,'w')
logger.addHandler(console_stream)
logger.addHandler(file_stream)

# define metric
class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)


class DataBase(object):
    def __init__(self,data,label):
        self.data = data
        self.label = label

    def getdata(self):
        return self.data

    def getlabel(self):
        return self.label

class DataIter(mx.io.DataIter):
    def __init__(self, images, labels, batch_size, height, width, process_num):
        assert process_num <= 40
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.count = len(images)
        self.height = height
        self.width = width
        self.images = images  
        self.labels = labels   
        self.cursor = -self.batch_size

        self.queue = multiprocessing.Queue(maxsize=4)
        self.started = True
        self.processes = [multiprocessing.Process(target=self.write) for i in range(process_num)]
        for process in self.processes:
            process.daemon = True
            process.start()

    @staticmethod
    def get_pos_mask(labels):
        indices_equal = np.eye(labels.shape[0])
        indices_not_equal = mx.nd.array(np.logical_not(indices_equal).astype(float))
        labels_equal = mx.nd.equal(mx.nd.expand_dims(labels,0),mx.nd.expand_dims(labels,1))
        pos_mask = mx.nd.equal(indices_not_equal,labels_equal)
        return pos_mask

    @staticmethod
    def get_neg_mask(labels):
        labels_equal = mx.nd.equal(mx.nd.expand_dims(labels,0),mx.nd.expand_dims(labels,1)).asnumpy()
        neg_mask = mx.nd.array(np.logical_not(labels_equal).astype(float))
        return neg_mask

    def get_closet_neg(self):
        pass

    def get_hardest_pos(self):
        pass

    def generate_batch(self):
        ret = []
        labels = []
        while len(ret) < self.batch_size:
            idx = random.sample(range(self.count), 1)
            mat = cv2.imread(self.images[idx])
            mat = cv2.resize(mat, (self.height, self.width))
            label = self.labels[idx]
            threshold = 250
            if np.mean(mat) > threshold:
                continue
            ret.append(mat)
            labels.append(label)
        return ret, labels

    def write(self):
        while True:
            if not self.started:
                break
            batch_data,batch_labels = self.generate_batch()
            batch = [x.transpose(2,0,1) for x in batch_data]
            data = mx.nd.array(batch)
            labels = mx.nd.array(batch_labels)
            pos_mask = self.get_pos_mask(batch_labels)
            neg_mask = self.get_neg_mask(batch_labels)

            data_plus_mask = [data,pos_mask,neg_mask]
            data_batch = DataBase(data_plus_mask,labels)
            self.queue.put(data_batch)

    def __del__(self):
        self.started = False
        for process in self.processes:
            process.join()
            while not self.queue.empty():
                self.queue.get(block=False)

    def next(self):
        if self.queue.empty():
            logging.debug("next_batch")
        if self.iter_next():
            return self.queue.get(block=True)
        else:
            raise StopIteration

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.count

    def reset(self):
        self.cursor = -self.batch_size

def network(units, embedding_dims, num_stage, filter_list, num_class, bottle_neck, bn_mom, workspace, margin, type='euclidean'):
    data = mx.sym.Variable('data')
    pos_mask = mx.sym.Variable('pos_mask')
    neg_mask = mx.sym.Variable('neg_mask')

    embeddings = resnet.resnet(
        data=data,
        units=units,
        embedding_dims=embedding_dims,
        num_stage=num_stage,
        filter_list=filter_list,
        num_class=num_class,
        data_type="imagenet",
        bottle_neck=bottle_neck,
        bn_mom=bn_mom,
        workspace=workspace)
    return batch_hard.batch_hard_triplet_loss(embeddings=embeddings,
                                                   pos_mask=pos_mask,
                                                   neg_mask=neg_mask,
                                                   margin=margin,
                                                   type=type)

def multi_factor_scheduler(begin_epoch,epoch_size,step,factor=0.1):
    step_ = [epoch_size * (x - begin_epoch) for x in step if x -begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_,factor=factor) if len(step_) else None

def main():
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    if not os.path.exists(args.trainlist):
        raise FileNotFoundError('Can\'t find trainlist file')

    images, labels = dataset.load_dataset(args.trainlist,
                                               args.dataset_path)

    kv = mx.kvstore.create(args.kv_store)
    devs = [mx.gpu(int(i)) for i in args.gpu_list]

    if args.depth == 18:
        units = [2, 2, 2, 2]
    elif args.depth == 34:
        units = [3, 4, 6, 3]
    elif args.depth == 50:
        units = [3, 4, 6, 3]
    elif args.depth == 101:
        units = [3, 4, 23, 3]
    elif args.depth == 152:
        units = [3, 8, 36, 3]
    elif args.depth == 200:
        units = [3, 24, 36, 3]
    elif args.depth == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))

    if isinstance(args.num_class,int):
        num_classes = args.num_classes
    else:
        raise TypeError("This dataset has not been supported")

    symbol = network(units=units,embedding_dims=128,num_stage=4,filter_list=[64, 256, 512, 1024, 2048],num_class=num_classes,
                     bottle_neck=True,bn_mom=0.9,workspace=512,margin=0.3)

    if args.pretrain_used == False:
        begin_epoch = 0
    else:
        begin_epoch = args.begin_epochs

    epoch_size = max(int(len(images) / args.batch_size / kv.num_workers), 1)
    train_data = DataIter(images=images,labels=labels,batch_size=args.batch_size,height=args.height,
                         width=args.width,process_num=args.process_num)
    optimizer = mx.optimizer.SGD(momentum=0.99)
    model = mx.model.FeedForward(
        allow_extra_params=True,
        ctx=devs,
        symbol=symbol,
        num_epoch=epoch_size,
        begin_epoch=begin_epoch,
        learning_rate=0.01,
        wd=0.001,
        momentum=0.9,
        initializer=mx.initializer.Load(args.model_path,default_init=mx.initializer.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)),
        optimizer=optimizer,
        lr_scheduler=multi_factor_scheduler(begin_epoch, args.epoch_size, step=[30, 60, 90], factor=0.1)
    )
    model.fit(
        X=train_data,
        eval_metric=mx.metric.CrossEntropy(),
        #eval_metric=Auc(),
        kvstore=kv,
        batch_end_callback=mx.callback.Speedometer(args.batch_size,20),
        epoch_end_callback=mx.callback.do_checkpoint(args.model_prefix,1)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('batch_hard_reid')
    parser.add_argument('--depth', required=True)
    parser.add_argument('--dataset_path')
    parser.add_argument('--log_path')
    parser.add_argument('--trainlist', required=True)
    parser.add_argument('--batch_size', default=72)
    parser.add_argument('--process_num',default=4)
    parser.add_argument('num_classes')
    parser.add_argument('--height', default=256,help='height')
    parser.add_argument('--width', default=128,help='width')
    parser.add_argument('--crop_height', default=288)
    parser.add_argument('--crop_width', default=144)
    parser.add_argument('--pretrain_used')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--absolute_path', required=True)
    parser.add_argument('--gpu_list',required=True,default=[0])
    parser.add_argument('--dataset_type',required=True)
    parser.add_argument('--model_prefix')
    parser.add_argument('--kv_store',type=int)
    args = parser.parse_args()
    logging.info(args)
    main()
    print('Done.')