#! /usr/bin/env python
# -*- coding: utf-8 -*-

import loguru
import unittest
import pytest
from pathlib import Path
import pandas as pd
import imma.via
import numpy as np


def test_via():
    pth_csv = Path(__file__).parent / "slice_raster_cr.csv"
    df = pd.read_csv(pth_csv)
    mask = imma.via.mask_from_via_annotation(df, shape=[1323, 300], filename_via="slice_raster_cr.png")
    un = np.unique(mask)
    assert 0 in un
    assert 1 in un
    assert 2 in un






