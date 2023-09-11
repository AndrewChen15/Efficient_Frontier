#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:07:53 2023

@author: andrewchen
"""

import numpy as np
import pandas as pd

def bond_price(times, cashflows, rate):
    
    d = 1 + rate
    answer = [1 / (d ** times[i]) * (cashflows[i]) for i in range(len(times))]
    
    return sum(answer)

def bootstrap(cashflows, prices):
    
    cf = np.matrix(cashflows)
    b = np.matrix(prices)
    
    answer =  cf.I @ b.T
    
    return answer

def bond_duration(times, cashflows, rate):
    
    t = np.array(times)
    c = np.array(cashflows)
    g = 1 + rate
    df = [(1 / (g ** times[i])) for i in range(len(times))]
    d = np.array(df)
    b = bond_price(times, cashflows, rate)
    p = (1/b) * t 
    i = p * c
    j = i * d.T
    answer = sum(j)
    
    return answer