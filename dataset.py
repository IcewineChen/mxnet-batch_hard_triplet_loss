#!/usr/bin/env python
# -*- coding = utf-8 -*-
import numpy as np
import os

def load_dataset(csv_file, image_root, fail_on_missing=True):
    dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')
    pids, fids = dataset.T

    if image_root is not None:
        missing = np.full(len(fids), False, dtype=bool)
        for i, fid in enumerate(fids):
            missing[i] = not os.path.isfile(os.path.join(image_root, fid))

        missing_count = np.sum(missing)
        if missing_count > 0:
            if fail_on_missing:
                raise IOError('Using the `{}` file and `{}` as an image root {}/'
                            '{} images are missing'.format(
                                csv_file, image_root, missing_count, len(fids)))
            else:
                print('[Warning] removing {} missing file(s) from the'
                    ' dataset.'.format(missing_count))
                fids = fids[np.logical_not(missing)]
                pids = pids[np.logical_not(missing)]

    return pids, fids