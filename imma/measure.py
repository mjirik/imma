#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)

# import copy
import numpy as np
import scipy.ndimage
from scipy.sparse import csc_matrix
from . import image_manipulation as ima


class CooccurrenceMatrix(object):
    def __init__(self, data, return_counts=True, dtype=int):
        self.update_cooccurrence_matrix(data, return_counts=return_counts)
        self.dtype = dtype

    def update_cooccurrence_matrix(self, data, return_counts=True):
        self.cooccurrence_matrix = cooccurrence_matrix(data, return_counts=return_counts)

    def get(self, intensity0, intensity1):
        return self.cooccurrence_matrix[intensity0][intensity1]

    def to_ndarray(self):
        keys = self.keys()
        inv_keys = self.inv_keys()
        sz = len(self.cooccurrence_matrix)
        ndnghb = np.zeros([sz, sz], dtype=self.dtype)
        for keyx in self.cooccurrence_matrix:
            nghbx = self.cooccurrence_matrix[keyx]
            for keyy in nghbx:
                value = nghbx[keyy]
                ndnghb[inv_keys[keyx], inv_keys[keyy]] = value
        return ndnghb

    def keys(self):
        return sorted(list(self.cooccurrence_matrix.keys()))

    def inv_keys(self):
        keys = list(self.keys())
        ii = list(range(len(keys)))
        return dict(zip(keys, ii))


def cooccurrence_matrix(data, return_counts=True):
    # csc_matrix((3, 4), dtype=np.int8).toarray()

    i = 0
    nbm = {}
    it = np.nditer(data, flags=['multi_index'])
    while not it.finished:
        print("iter ", i)
        i += 1
        mindex0 = it.multi_index
        # print("%d <%s>" % (it[0], mindex0), end=' ')
        data_value0 = data[mindex0]
        for axn in range(len(mindex0)):
            mindex1 = list(mindex0)
            mindex1[axn] = mindex1[axn] + 1
            mindex1 = tuple(mindex1)
            if np.all(np.asarray(mindex1) < np.asarray(data.shape)):
                data_value1 = data[mindex1]
                # budeme vyplnovat jen spodni
                # if data_value0 < data_value1:
                #     data_value0s = data_value0
                #     data_value1s = data_value1
                # else:
                #     data_value0s = data_value1
                #     data_value1s = data_value0
                data_value0s = data_value0
                data_value1s = data_value1

                if data_value0s not in nbm.keys():
                    nbm[data_value0s] = {}
                if data_value1s not in nbm[data_value0s].keys():
                    nbm[data_value0s][data_value1s] = 0
                elif not return_counts:
                    # make it faster
                    continue

                nbm[data_value0s][data_value1s] += 1

                if data_value1s not in nbm.keys():
                    nbm[data_value1s] = {}
                if data_value0s not in nbm[data_value1s].keys():
                    nbm[data_value1s][data_value0s] = 0

                # if data_value0s != data_value1s:
                #     # we dont want to put the numer into diagonal for twice
                # on diagonal there will be doubled values
                nbm[data_value1s][data_value0s] += 1

        it.iternext()
    return nbm


def objects_neighbors(labeled_ndarray, labels=None, exclude=None):
    """
    Neighbors for one or more object. Objects with label 0 are ignored.

    :param labeled_ndarray: 3D ndarray
    :param labels: Integer label or list of ints. If is set to None, all labels are processed.
    :param exclude: List of labels to exclude.
    :return:
    """

    if np.min(labeled_ndarray) < 0:
        ValueError("Input image cannot contain negative labels.")

    if exclude is None:
        exclude = []
    bboxes = scipy.ndimage.find_objects(labeled_ndarray)
    bbox_margin = 1

    output = [None] * len(bboxes)
    if labels is None:
        labels = range(1, len(bboxes) + 1)
    else:
        if type(labels) is not list:
            labels = [labels]

    for i in labels:
        bbox = bboxes[i - 1]
        ilabel = i
        if bbox is not None:
            exbbox = ima.extend_crinfo(bbox, labeled_ndarray.shape, bbox_margin)
            cropped_ndarray = labeled_ndarray[exbbox]
            object = (cropped_ndarray == ilabel)
            dilat_element = scipy.ndimage.morphology.binary_dilation(
                object,
                structure=np.ones([3, 3, 3])
            )

            neighborhood = cropped_ndarray[dilat_element]

            neighbors = np.unique(neighborhood)
            neighbors = neighbors[neighbors != ilabel]
            # neighbors = neighbors[neighbors != 0]
            for exlabel in exclude:
                neighbors = neighbors[neighbors != exlabel]
            output[i - 1] = list(neighbors)

    return output
