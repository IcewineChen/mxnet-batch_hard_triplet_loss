#!/usr/bin/env python
# -*- coding = utf-8 -*-
import mxnet as mx

def residual_unit(data,num_filter,stride,dim_match,name,bottle_neck=True,bn_mom=0.9,workspace=512,memonger=False):
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(data, labels, batch_size, margin, units, embedding_dims, num_stage, filter_list, num_class, data_type, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False, type='euclidean'):
    num_unit = len(units)
    assert(num_unit == num_stage)
    # data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(int(units[i])-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    # get embeddings
    emb = mx.sym.FullyConnected(data=flat, num_hidden=embedding_dims,name='embeddings')

    # print(emb.infer_shape_partial(samples=(72, 3, 256, 144)))

    dot_product = mx.sym.dot(emb,mx.sym.transpose(emb),name='dot')
    square_L2_emb = mx.sym.square(mx.sym.norm(dot_product,ord=2,axis=0),name='squ_l2')
    distances = mx.sym.broadcast_sub(
        mx.sym.broadcast_add(mx.sym.expand_dims(square_L2_emb,0),mx.sym.expand_dims(square_L2_emb,axis=1)),
        mx.sym.broadcast_add(square_L2_emb,square_L2_emb),
        name='distance_1'
    )
    distances = mx.sym.maximum(distances,0.0)
    if type=='euclidean':
        distances = mx.sym.sqrt(distances,name='eu_distance')
    elif type=='sqeuclidean':
        distances = distances
    else:
        raise NotImplementedError('Haven\'t implement this type of distance')

    pos_indices_equal=mx.sym.eye(batch_size)
    pos_indices_not_equal = mx.sym.logical_not(pos_indices_equal)
    pos_labels_equal = mx.sym.broadcast_equal(mx.sym.expand_dims(labels,0), mx.sym.expand_dims(labels,1))

    pos_mask = mx.sym.broadcast_logical_and(pos_indices_not_equal,pos_labels_equal,name='pos_mask')

    neg_labels_equal = mx.sym.broadcast_equal(mx.sym.expand_dims(labels,axis=0),mx.sym.expand_dims(labels,axis=1))
    neg_mask = mx.sym.logical_not(neg_labels_equal,name='neg_mask')

    # batch_hard
    pos_anchor_dist = mx.sym.broadcast_mul(pos_mask,distances,name='pos_dist')
    hardest_pos_dist = mx.sym.max(pos_anchor_dist,axis=1,keepdims=True,name='pos_hardest')

    max_neg_anchor_dist = mx.sym.max(distances,axis=1,keepdims=True)

    neg_anchor_dist = mx.sym.broadcast_add(distances,
                                           mx.sym.broadcast_mul(max_neg_anchor_dist,mx.sym.broadcast_sub(
                                               mx.sym.full(shape=(batch_size,batch_size),val=1),
                                               neg_mask)),name='neg_dist'
                                           )

    hardest_neg_dist = mx.sym.min(neg_anchor_dist,name='neg_hardest')

    loss_shape = [batch_size,1]
    margin = mx.sym.full(shape=loss_shape,val=margin)
    cpm_zero = mx.sym.full(shape=loss_shape,val=0.0)
    triplet_loss = mx.sym.maximum(
        mx.sym.broadcast_add(
            mx.sym.broadcast_sub(hardest_pos_dist,hardest_neg_dist),margin
        ),
        cpm_zero
    )
    # triplet_loss = mx.sym.maximum(hardest_pos_dist - hardest_neg_dist + margin, mx.sym.full(shape=loss_shape,val=0.0))
    # arg, out1, aux = hardest_pos_dist.infer_shape_partial(samples=(72, 3, 256, 144), labels=(72,))
    # m1, out2, m3 = triplet_loss.infer_shape_partial(samples=(72, 3, 256, 128), labels=(72,))
    triplet_loss = mx.sym.mean(triplet_loss,name='tri_loss')
    return mx.sym.MakeLoss(triplet_loss)

