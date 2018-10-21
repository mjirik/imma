#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import logging

logger = logging.getLogger(__name__)
import numpy as np

import unittest
from collections import OrderedDict
from imma import obj


class ObjTestCase(unittest.TestCase):
    def generate_dict_data(self):
        data = {
            'a': 1,
            'b': 2,
            'c': {
                'aa': 11,
                'bb': 22,
                'cc': {
                    'aaa': 111
                }
            }
        }
        return data

    def test_get_standard_arguments(self):
        from collections import OrderedDict
        args = obj.get_default_args(Foo)
        self.assertEqual(type(args), OrderedDict)
        self.assertIn("first", args)
        self.assertEqual(args["first"], None)
        self.assertEqual(args["second"], 5)
        self.assertEqual(args["third"], [])

    @unittest.skip("Waiting for implementation")
    def test_get_standard_arguments_with_position_arg(self):
        from collections import OrderedDict
        args = obj.get_default_args(Bar)
        self.assertEqual(type(args), OrderedDict)
        self.assertIn("first", args)
        self.assertEqual(args["first"], None)
        self.assertEqual(args["second"], 5)
        self.assertEqual(args["third"], [])


class Foo:
    def __init__(self, first=None, second=5, third=[]):
        pass


class Bar:
    def __init__(self, zero, first=None, second=5, third=[]):
        pass


def main():
    unittest.main()
