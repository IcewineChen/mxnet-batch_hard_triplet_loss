#!/usr/bin/env python
# -*- coding = utf-8 -*-
import mxnet as mx
import numpy as np

#import mxboard

def get_pos_mask(labels, shape):
    indices_equal = mx.sym.eye(shape)
    indices_not_equal = mx.sym.logical_not(indices_equal)
    labels_equal = mx.sym.broadcast_equal(mx.sym.expand_dims(labels,0), mx.sym.expand_dims(labels,1))
    pos_mask = mx.sym.broadcast_equal(indices_not_equal,labels_equal)
    return pos_mask

def get_neg_mask(labels):
    labels_equal = mx.sym.broadcast_equal(mx.sym.expand_dims(labels,0),mx.sym.expand_dims(labels,1))
    neg_mask = mx.sym.logical_not(labels_equal)
    return neg_mask

def _pairwise_distances(embeddings, type='euclidean'):
    dot_product = mx.sym.dot(embeddings,mx.sym.transpose(embeddings))
    dot_product = mx.sym.square(mx.sym.norm(dot_product,ord=2,axis=0))
    distances = mx.sym.expand_dims(dot_product, 0) - mx.sym.broadcast_sub(dot_product, dot_product) + mx.sym.expand_dims(dot_product,1)
    # distances = mx.sym.expand_dims(dot_product, 0) - 2.0 * dot_product + mx.sym.expand_dims(dot_product, 1)
    distances = mx.sym.maximum(distances, 0.0)
    eps = 1e-16

    # mask = mx.sym.broadcast_equal(distances, 0.0)
    # distances = distances + mask * eps
    
    if type == 'euclidean':
        distances = mx.sym.sqrt(distances)
    elif type == 'sqeuclidean':
        distances = distances
    else:
        raise NotImplementedError('Haven\'t implement this type of distance')

    # distances = distances * (1.0 - mask)
    return distances

def batch_hard_triplet_loss(embeddings, labels, shape, margin, type='euclidean'):
    pairwise_dist = _pairwise_distances(embeddings)
    pos_mask = get_pos_mask(labels, shape)
    neg_mask = get_neg_mask(labels)
    anchor_positive_dist = mx.sym.dot(pos_mask, pairwise_dist)
    #print(anchor_positive_dist.infer_shape_partial(samples=(72,3,256,144)))
    hardest_positive_dist = mx.sym.max(anchor_positive_dist, axis=1, keepdims=True)
    #mxboard.summary.scalar_summary("hardest_positive_dist",mx.sym.mean(hardest_positive_dist))

    max_anchor_negative_dist = mx.sym.max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + mx.sym.broadcast_mul(max_anchor_negative_dist, mx.sym.broadcast_sub(mx.sym.ones((shape,shape)),neg_mask))

    # anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - neg_mask)

    hardest_negative_dist = mx.sym.min(anchor_negative_dist, axis=1, keepdims=True)
    #mxboard.summary.scalar_summary("hardest_negative_dist", mx.sym.mean(hardest_negative_dist))
    margin = mx.sym.full(shape=(shape,shape),val=margin)
    hard_mining = mx.sym.broadcast_add(mx.sym.broadcast_sub(hardest_positive_dist,hardest_negative_dist),margin)
    triplet_loss = mx.sym.maximum(hard_mining, 0.0)
    #triplet_loss = mx.sym.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    if margin == 'soft':
        triplet_loss = mx.sym.log10(
            mx.sym.broadcast_add(
                mx.sym.broadcast_sub(mx.sym.exp(hardest_positive_dist),hardest_positive_dist),mx.sym.ones((shape,shape)))
        )
    triplet_loss = mx.sym.mean(triplet_loss)

    return mx.sym.MakeLoss(triplet_loss)

