import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.metrics import brier_score_loss, log_loss
import sympy as sym

pred_on_admission = [0,0,0.5,0.5,0,0]

def pred_at_t_(mrn, df):
    days_since_admit = df[df['mrn']==mrn]['days_since_admit']

    # now shifting the pred made on admission to reflect days already spent on the unit
    pred = np.array(pred_on_admission)[int(days_since_admit):]
    # adding zeros to give a list of the correct length
    pred = np.append(pred,[0.]*int(days_since_admit))

    # handling the case where the patient has stayed longer that 4 days, which on admission we predict as 0 probability
    if np.array(pred).sum()==0:
        pred = np.array([1., 0., 0., 0., 0., 0.])
    # normalise to 1
    else:
        pred = pred*(1/pred.sum())

    return pred

def one_hot(days):
    buckets =[]

    for i in range(5):
        if days == i: buckets.append(1)
        else: buckets.append(0)

    if days>=5:buckets.append(1)
    else: buckets.append(0)

    return buckets

s = sym.Symbol('s')
r = sym.Symbol('r')
syms = sym.symbols('r0:8')
core_expression = (1-r)+r*s

def ex(ri): return core_expression.subs({r:ri})

def build_expression():
    expression = 1
    for i in range(8):
        expression = expression*ex(syms[i])
    return expression
