#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)
# import funkcí z jiného adresáře
import os.path

import unittest

import numpy as np
import os

# import io3d
import io3d.datasets
import imma.labeled as ima
import imma.image as imim


class LabeledTest(unittest.TestCase):


    def test_crop_from_specific_data(self):
        datap = io3d.datasets.generate_abdominal()
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        crinfo_auto1 = ima.crinfo_from_specific_data(segmentation, [5])
        crinfo_auto2 = ima.crinfo_from_specific_data(segmentation, 5)
        crinfo_auto3 = ima.crinfo_from_specific_data(segmentation, [5, 5, 5])

        crinfo_expected = [[0, 99], [20, 99], [45, 99]]

        self.assertEquals(crinfo_auto1, crinfo_expected)
        self.assertEquals(crinfo_auto1, crinfo_auto2)
        self.assertEquals(crinfo_auto1, crinfo_auto3)

    def test_crop_from_specific_data_with_slices(self):
        datap = io3d.datasets.generate_abdominal()
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        crinfo_auto1 = ima.crinfo_from_specific_data(segmentation, [5], with_slices=True)
        self.assertEqual(type(data3d[crinfo_auto1]), np.ndarray, "We are able to use slices in data.")

        crinfo_auto1 = imim.fix_crinfo(crinfo_auto1, with_slices=False)
        crinfo_expected = [[0, 99], [20, 99], [45, 99]]

        self.assertEquals(crinfo_auto1, crinfo_expected)

    def test_select_objects_by_seeds(self):
        shape = [12, 15, 12]
        data = np.zeros(shape)
        value1 = 1
        value2 = 1
        data[:5, :7, :6] = value1
        data[-5:, :7, :6] = value2

        seeds = np.zeros(shape)
        seeds[9, 3:6, 3] = 1

        selected = ima.select_objects_by_seeds(data, seeds)
        # import sed3
        # ed =sed3.sed3(selected, contour=data, seeds=seeds)
        # ed.show()
        unique = np.unique(selected)
        #
        self.assertEqual(selected.shape[0], shape[0])
        self.assertEqual(selected.shape[1], shape[1])
        self.assertEqual(selected.shape[2], shape[2])
        self.assertEqual(selected[1, 1, 1], 0)
        self.assertEqual(selected[-2, 1, 1], 1)
        self.assertEqual(len(unique), 2)
        self.assertGreater(np.count_nonzero(data), np.count_nonzero(selected))

if __name__ == "__main__":
    unittest.main()
