#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:09:21 2023

@author: andrewchen
"""

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from a10task2 import * 

def plot_stock_prices(symbols):
    
    df = get_stock_prices_from_csv_files(symbols)
    for i in range(len(symbols)):
    
        plt.plot(df[symbols[i]], label = symbols[i])
        plt.legend()
        
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Prices')
    
def plot_stock_cumulative_change(symbols):

    df = get_stock_prices_from_csv_files(symbols)
    for i in range(len(symbols)):
    
    
        plt.plot(df[symbols[i]] / df[symbols[i]].values[0], label = symbols[i])
        plt.legend()
        
    plt.xlabel('Date')
    plt.ylabel('Relative Price')
    plt.title('Cumulative Change in Stock Prices')
        
    
def plot_efficient_frontier(symbols):
    
    df = get_stock_returns_from_csv_files(symbols)
    v = get_covariance_matrix(df)
    v = np.matrix(v)
    w = calc_global_min_variance_portfolio(v)
    e = []
    for c in df.columns:
    
        e += [np.mean(df[c])]
    e = np.matrix(e)
    re = calc_portfolio_return(e, w)
    st = calc_portfolio_stdev(v, w)
    rs = np.linspace(re-st, re+st)
    ef = calc_efficient_portfolios_stdev(e, v, rs)
    plt.plot(ef, rs, '-')
    plt.xlabel('Portfolio Standard Deviation')
    plt.ylabel('Portfolio Expected Return')
    plt.title('Efficient Frontier')