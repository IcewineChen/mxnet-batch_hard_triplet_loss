#!/usr/bin/env python
# -*- coding = utf-8 -*-
import mxnet as mx
import numpy as np

import mxboard

def _pairwise_distances(embeddings, type='eulcidean'):
    dot_product = mx.sym.dot(embeddings,mx.sym.transpose(embeddings))
    dot_product = mx.sym.square(mx.sym.norm(dot_product,ord=2,axis=0))
    distances = mx.sym.expand_dims(dot_product, 0) - 2.0 * dot_product + mx.sym.expand_dims(dot_product, 1)
    distances = mx.sym.maximum(distances, 0.0)
    eps = 1e-16

    mask = mx.sym.broadcast_equal(distances, 0.0)
    distances = distances + mask * eps
    
    if type == 'euclidean':
        distances = mx.sym.sqrt(distances)
    if type == 'sqeuclidean':
        distances = distances
    else:
        raise NotImplementedError('Haven\'t implement this type of distance')

    distances = distances * (1.0 - mask)
    return distances



def batch_hard_triplet_loss(embeddings, pos_mask, neg_mask, margin, type='euclidean'):
    pairwise_dist = _pairwise_distances(embeddings)

    anchor_positive_dist = mx.sym.dot(pos_mask, pairwise_dist)

    hardest_positive_dist = mx.sym.max(anchor_positive_dist, axis=1, keepdims=True)
    mxboard.summary.scalar_summary("hardest_positive_dist",mx.sym.mean(hardest_positive_dist))

    max_anchor_negative_dist = mx.sym.max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - neg_mask)

    hardest_negative_dist = mx.sym.min(anchor_negative_dist, axis=1, keepdims=True)
    mxboard.summary.scalar_summary("hardest_negative_dist", mx.sym.mean(hardest_negative_dist))

    triplet_loss = mx.sym.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    if margin == 'soft':
        triplet_loss = mx.sym.log10(mx.sym.exp(hardest_positive_dist-hardest_positive_dist)+1)

    triplet_loss = mx.sym.mean(triplet_loss)

    return mx.sym.MakeLoss(triplet_loss)

