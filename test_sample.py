# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:40:26 2022

@author: earyo
"""
#https://realpython.com/pytest-python-testing/
import os
import pytest

# content of test_sample.py
def func(x):
    return x + 1


def test_answer():
    assert func(3) == 4
    
