#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:58:13 2019

@author: murali
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn import  datasets
from sklearn import decomposition
from sklearn.cluster import DBSCAN 
from sklearn.cluster import AgglomerativeClustering 

from sklearn import metrics
from scipy.optimize import linear_sum_assignment

import math as m
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
import collections
from matplotlib.tri import Triangulation
