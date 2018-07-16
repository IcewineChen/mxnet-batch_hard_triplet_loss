#!/usr/bin/env python
# -*- coding = utf-8 -*-
import mxnet as mx
import numpy as np
import cv2
import os
import argparse
import multiprocessing
import random
import logging
import resnet
import dataset

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
<<<<<<< HEAD
        self.label_width = 1
        self.provide_data = [("samples", (self.batch_size, 3, height, width))]
=======

        self.provide_data = [( "samples", (self.batch_size, 3, height, width))]
>>>>>>> 70284ad85604842bc7f1d833509f6046f39b4396
        #self.provide_data = [("samples", (self.batch_size, 3, height, width)),
        #                     ("pos_mask", (self.batch_size, self.batch_size)),
        #                     ("neg_mask", (self.batch_size, self.batch_size))]
        self.provide_label = [("labels", (self.batch_size,))]

        self.queue = multiprocessing.Queue(maxsize=4)
        self.started = True
        self.processes = [multiprocessing.Process(target=self.write) for i in range(process_num)]

        for process in self.processes:
            process.daemon = True
            process.start()
    """
    @staticmethod
    def get_pos_mask(labels):
        # labels = mx.nd.array(labels)
        indices_equal = np.eye(labels.shape[0])
        indices_not_equal = mx.nd.array(np.logical_not(indices_equal).astype(float))
        labels_equal = mx.nd.equal(mx.nd.expand_dims(labels,0),mx.nd.expand_dims(labels,1))
        pos_mask = mx.nd.equal(indices_not_equal,labels_equal)
        return pos_mask
    
    @staticmethod
    def get_neg_mask(labels):
        # labels = mx.nd.array(labels)
        labels_equal = mx.nd.equal(mx.nd.expand_dims(labels,0),mx.nd.expand_dims(labels,1)).asnumpy()
        neg_mask = mx.nd.array(np.logical_not(labels_equal).astype(float))
        return neg_mask
    """
    def generate_batch(self):
        ret = []
        labels = []
        while len(ret) < self.batch_size:
            idx = random.sample(range(self.count), 1)
            mat = cv2.imread(self.images[idx[0]])
            mat = cv2.resize(mat, (self.height, self.width))
            label = self.labels[idx[0]]
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
            batch_data, batch_labels = self.generate_batch()
            batch = [x.transpose(2,0,1) for x in batch_data]
            data = [mx.nd.array(batch)]
            labels = [mx.nd.array(batch_labels)]
            # labels = mx.nd.expand_dims(mx.nd.array(batch_labels),axis=0)
            # pos_mask = self.get_pos_mask(labels)
            # neg_mask = self.get_neg_mask(labels)

            #data_plus_mask = [data,pos_mask,neg_mask]
            #data_batch = DataBase(data_plus_mask,labels)
            data_batch = DataBase(data,labels)
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

def network(units, embedding_dims, num_stage, filter_list, num_class, bottle_neck, bn_mom, workspace, margin, batch_size, type='euclidean'):
    samples = mx.sym.Variable("samples")
    labels = mx.sym.Variable("labels")
    labels = mx.sym.reshape(labels,(1,72))
    # concat = mx.sym.Concat(*data,dim=0,name="concat")
    # pos_mask = mx.sym.Variable('pos_mask')
    # neg_mask = mx.sym.Variable('neg_mask')
<<<<<<< HEAD
    embeddings = resnet.resnet(
=======
    # labels = mx.sym.Variable('pids')
    # data = mx.sym.expand_dims(data,axis=0)
    tripletloss = resnet.resnet(
>>>>>>> 70284ad85604842bc7f1d833509f6046f39b4396
        data=samples,
        labels=labels,
        batch_size=batch_size,
        margin=0.3,
        units=units,
        embedding_dims=embedding_dims,
        num_stage=num_stage,
        filter_list=filter_list,
        num_class=num_class,
        data_type="imagenet",
        bottle_neck=bottle_neck,
        bn_mom=bn_mom,
        workspace=workspace,
        type='euclidean'
    )
    # test
    return tripletloss
    # embeddings = mx.sym.expand_dims(embeddings,axis=0)
<<<<<<< HEAD

    return batch_hard.batch_hard_triplet_loss(embeddings=embeddings,
                                                   #pos_mask=pos_mask,
                                                   #neg_mask=neg_mask,
                                                   labels=labels,
                                                   shape=batch_size,
                                                   margin=margin,
                                                   type=type)
=======
    # return batch_hard.batch_hard_triplet_loss(embeddings=embeddings,
    #                                               #pos_mask=pos_mask,
    #                                               #neg_mask=neg_mask,
    #                                               shape=batch_size,
    #                                               labels=labels,
    #                                               margin=margin,
    #                                               type=type)
>>>>>>> 70284ad85604842bc7f1d833509f6046f39b4396

def multi_factor_scheduler(begin_epoch,epoch_size,step,factor=0.1):
    step_ = [epoch_size * (x - begin_epoch) for x in step if x -begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_,factor=factor) if len(step_) else None

def main():

    log_path = './logs'
    logfile = './logs/train.log'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_stream = logging.StreamHandler()

    if not os.path.exists(log_path):
        os.mkdir(log_path)
        file_stream = logging.FileHandler(logfile, 'w')
        logger.addHandler(console_stream)
        logger.addHandler(file_stream)

    if not os.path.exists(args.trainlist):
        raise FileNotFoundError('Can\'t find trainlist file')

    labels, images = dataset.load_dataset(args.trainlist,args.dataset_path)

    kv = mx.kvstore.create(args.kv_store)
    # devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    devs = [mx.gpu(0)]
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

    if isinstance(args.num_classes,int):
        num_classes = args.num_classes
    else:
        raise TypeError("This dataset has not been supported")

    symbol = network(units=units,embedding_dims=128,num_stage=4,filter_list=[64, 256, 512, 1024, 2048],num_class=num_classes,
                     bottle_neck=True,bn_mom=0.9,workspace=512,margin=0.3,batch_size=args.batch_size,type='euclidean')

    if args.pretrain_used == False:
        begin_epoch = 0
    else:
        begin_epoch = args.begin_epochs

    epoch_size = max(int(len(images) / args.batch_size / kv.num_workers), 1)
    train_data = DataIter(images=images,labels=labels,batch_size=args.batch_size,height=args.height,
                         width=args.width,process_num=args.process_num)

    optimizer = mx.optimizer.SGD({'learning_rate': 0.01, 'momentum': 0.99})
    
    # model = mx.mod.Module(symbol,data_names=["samples"],label_names=["labels"],context=devs)
    # model.bind(data_shapes=train_data.provide_data,
    #            label_shapes=train_data.provide_label)
    # model.init_params(initializer=mx.initializer.Xavier(magnitude=2.))

    model = mx.mod.Module(symbol=symbol,data_names=('samples',),
                          label_names=('labels',),context=devs)
    model.bind(data_shapes=train_data.provide_data,
               label_shapes=train_data.provide_label,inputs_need_grad=True)
    model.init_params(initializer=mx.initializer.Xavier(magnitude=2.))
    model.init_optimizer(optimizer='sgd',
                         optimizer_params=(('learning_rate',0.1),
                                            ('momentum',0.99)
                        ))
    metric = mx.metric.create('acc')
    
    for epoch in epoch_size:
        train_data.reset()
        metric.reset()
        for batch in train_data:
            model.forward(batch,is_train=True)
            model.update_metric(metric,batch.label)
            model.backward()
            model.update()
        print("epoch %d, Training %s").format(epoch,metric.get())
            

    # model = mx.model.FeedForward(
    #    allow_extra_params=True,
    #    ctx=devs,
    #    symbol=symbol,
    #    num_epoch=epoch_size,
    #    begin_epoch=begin_epoch,
    #    learning_rate=0.01,
    #    wd=0.001,
    #    momentum=0.9,
    #    initializer=mx.initializer.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2),
    #    optimizer=optimizer,
    #    lr_scheduler = multi_factor_scheduler(0, 200, step=[30, 60, 90], factor=0.1)
    #)

    # model.fit(
    #    train_data,
    #    num_epoch=0,
    #    eval_metric='acc'
    #    eval_metric=Auc(),
    #    kvstore=kv,
    #    batch_end_callback=mx.callback.Speedometer(args.batch_size,20),
    #    epoch_end_callback=mx.callback.do_checkpoint(args.model_prefix)
    #)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('batch_hard_reid')
    parser.add_argument('--depth', required=True, type=int)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--log_path')
    parser.add_argument('--trainlist', required=True)
    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--process_num', default=4)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--height', default=256, help='height')
    parser.add_argument('--width', default=128, help='width')
    parser.add_argument('--crop_height', default=288)
    parser.add_argument('--crop_width', default=144)
    parser.add_argument('--pretrain_used' ,type=bool, default=False)
    parser.add_argument('--model_path')
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model_prefix')
    parser.add_argument('--kv_store', type=str, default='device')
    args = parser.parse_args()
    logging.info(args)
    main()
    print('Done.')