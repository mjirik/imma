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

    def test_squeeze_labels(self):
        seeds = np.zeros([50, 60, 70])
        seeds[20:30, 20:30, 20:30] = 30
        seeds[20:30, 50:70, 20:30] = 60
        seeds[20:30, 10:200, 40:60] = 61
        squeezed_seeds = ima.squeeze_labels(seeds)
        self.assertEqual(np.max(squeezed_seeds), 3)

    def test_squeeze_labels_with_negative_and_collision(self):
        seeds = np.zeros([50, 60, 70])
        seeds[20:30, 20:30, 20:30] = -10
        seeds[20:30, 50:70, 20:30] = 1
        seeds[20:30, 10:200, 40:60] = 61
        squeezed_seeds = ima.squeeze_labels(seeds)
        self.assertEqual(np.max(squeezed_seeds), 3)

    def test_dist_segmentation(self):
        seeds = np.zeros([8, 10])
        seeds[1:7, 8] = 1
        seeds[6, 5] = 2
        seeds[2, 0] = 3

        import matplotlib.pyplot as plt
        # plt.imshow(seeds, interpolation="nearest")
        # plt.show()

        segm = ima.distance_segmentation(seeds)

        # plt.imshow(segm, interpolation="nearest")
        # plt.show()

        self.assertEqual(segm[2, 2], 3)
        self.assertEqual(segm[5, 5], 2)

        # dist, inds = scipy.in

    def test_select_labels(self):
        datap = io3d.datasets.generate_abdominal()
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        selection = ima.select_labels(segmentation, 1)
        self.assertGreater(np.sum(selection), 50, "select at least few pixels")
        # crinfo1 = ima.crinfo_from_specific_data(segmentation, [5])
        # crinfo2 = ima.extend_crinfo(crinfo1, data3d.shape, 3)
        # self.assertEqual(type(data3d[crinfo2]), np.ndarray, "We are able to use slices in data with extended crinfo.")

    def test_select_labels_with_slab(self):
        datap = io3d.datasets.generate_abdominal()
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        selection = ima.select_labels(segmentation, "liver", slab=datap["slab"])
        self.assertGreater(np.sum(selection), 50, "select at least few pixels")


    def test_biggest_object(self):
        datap = io3d.datasets.generate_abdominal()
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        seg_biggest = ima.get_one_biggest_object(segmentation)
        # newlab = ima.get_nlabels(datap["slab"], "new", return_mode="str")
        self.assertEqual(type(seg_biggest), np.ndarray)
        self.assertTrue(np.array_equal(seg_biggest.shape, segmentation.shape))
        self.assertTrue(np.array_equal(seg_biggest.shape, segmentation.shape))

    def test_biggest_object_label(self):
        datap = io3d.datasets.generate_abdominal()
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        slab = datap["slab"]
        # import io3d.datasets
        # import imma.image_manipulation as ima
        datap = io3d.datasets.generate_abdominal()
        bigges_area_label = ima.max_area_index(segmentation)
        self.assertEqual(bigges_area_label, slab["liver"])

        # test also the old version
        bigges_area_label2 = ima.max_area_index2(segmentation, 30)
        self.assertEqual(bigges_area_label, bigges_area_label2)

    @unittest.skip("This test just checks the time requirements")
    def test_biggest_object_label_timeit(self):
        import timeit
        t0 = timeit.timeit(
            setup="""
import io3d.datasets
import imma.image_manipulation as ima
datap = io3d.datasets.generate_abdominal()
            """,
            stmt="ima.max_area_index(datap['segmentation'], 100)", number=3)
        t1 = timeit.timeit(
            setup="""
import io3d.datasets
import imma.image_manipulation as ima
datap = io3d.datasets.generate_abdominal()
            """,
            stmt="ima.max_area_index2(datap['segmentation'], 100)", number=3)
        # seg_biggest_i = ima.max_area_index(segmentation, 100)
        # seg_biggest_i = ima.max_area_index(segmentation, 100)
        # %timeit
        # newlab = ima.get_nlabels(datap["slab"], "new", return_mode="str")
        # self.assertEqual(type(seg_biggest), np.ndarray)
        # self.assertEqual(np.array_equal(seg_biggest.shape), segmentation.shape)
        # self.assertEqual(np.array_equal(seg_biggest.shape), segmentation.shape)
        self.assertGreater(t1, t0)
        # print(t0, t1)


if __name__ == "__main__":
    unittest.main()
