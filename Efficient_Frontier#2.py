#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:08:19 2023

@author: andrewchen
"""

import numpy as np
import math 
import pandas as pd

def calc_portfolio_return(e, w):
    
    answer = e @ w.T
    
    return answer[0,0]

def calc_portfolio_stdev(v, w): 
    
    answer = w * v * w.T 
    
    
    return math.sqrt(float(answer))

def calc_global_min_variance_portfolio(v):
   
    k = np.ones((len(v),len(v[0])))
    c = k.T * v.I * k 
    M = k/c
    w = M @ k.T * v.I
    
    return w[0]

def calc_min_variance_portfolio(e, v, r) :
    
    k = np.ones((len(v),len(v[0])))
    
    a = float(k.T * v.I * e.T)
    
    b = float(e * v.I * e.T)
    
    c = float(k.T * v.I * k)
    
    A = np.array([[b,a],[a,c]])
    
    d = np.linalg.det(A)
    
    g = (1/d) * (b * k.T - a * e) * v.I
    
    h = (1/d) * (c * e - a * k.T) * v.I
    
    w = g + h * r
    
    return w
    
    
    
def calc_efficient_portfolios_stdev(e, v, rs):
    
    sig = []
    
    for i in range(len(rs)):
    
        print('r = ', rs[i], ', sigma = ', calc_portfolio_stdev(v, calc_min_variance_portfolio(e, v, rs[i]) ), '  w = ', calc_global_min_variance_portfolio(v))
        sig += [calc_portfolio_stdev(v, calc_min_variance_portfolio(e, v, rs[i]))]
    return np.array(sig)


def get_stock_prices_from_csv_files(symbols):
    
    df = pd.DataFrame(columns=['Date'])
    
    d = './%s.csv' % symbols[0]
    
    g = pd.read_csv(d)
    
    df = g[['Date']]
    
    
    for symbol in symbols:

        fn = './%s.csv' % symbol
        
        a = pd.read_csv(fn)
        
        prices = a[['Date','Adj Close']]
        
        prices = prices.rename(columns = {'Adj Close': symbol})
        
        df = df.join(prices.set_index('Date'), on = 'Date')
        
    df.index = df['Date']
    
    df.index = pd.to_datetime(df.index)
    
    del df['Date']
        
    return df
        
def get_stock_returns_from_csv_files(symbols): 
        
    return get_stock_prices_from_csv_files(symbols).pct_change()
        
def get_covariance_matrix(returns):

    return returns.cov()