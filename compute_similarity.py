import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TOP_N_GENES = 10

def load_data():
    # for loading the data from csv into a df
    raise NotImplementedError

def summarize_cohorts(X, cohorts, order):
    # basic counts (mutation burden)
    raise NotImplementedError

def top_genes(X, cohorts, order):
    # top however many genes in each cohort
    raise NotImplementedError

def jaccard_matrix(X):
    # jaccard similarity matrix for every pair of tumors
    raise NotImplementedError

def similarity_by_cohort(sim, labels, order):
    # mean similarity for every pair of cohorts
    raise NotImplementedError

def within_between_values(sim, labels, c):
    # all within type pair similarities for cohort c
    # and all its pairs with other cohorts
    raise NotImplementedError

def within_vs_between(sim, labels, order):
    # mean within vs between type similarities for each cohort and overall
    raise NotImplementedError

def plot_burden(burden, cohorts, order):
    raise NotImplementedError

def plot_heatmap(table):
    raise NotImplementedError

def plot_within_between(sim, labels, order):
    raise NotImplementedError