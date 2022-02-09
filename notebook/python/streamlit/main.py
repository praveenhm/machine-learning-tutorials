import numpy as np
import pandas as pd
import streamlit as st
from covariance import create_data, altair_chart, matplotlib_chart, render_correlation
from regularization import render_sample_size_regularization

# Starting with the standard imports
import numpy as np
import pandas as pd
import pandas_profiling

# Preprocessing data
from sklearn.model_selection import train_test_split  # data-splitter
from sklearn.preprocessing import StandardScaler  # data-normalization
from sklearn.preprocessing import PolynomialFeatures  # for polynomials
from sklearn.pipeline import make_pipeline  # for pipelines

np.random.seed(42)  # for reproducible results

#
# Modeling and Metrics
#
# --For Regressor
from sklearn.linear_model import LinearRegression  # linear regression
from sklearn.metrics import mean_squared_error, r2_score  # model-metrics

# Now the Graphical libraries imports and settings
# %matplotlib inline
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # nicer looking plots
import altair as alt  # for interactive plots
from matplotlib import colors  # for web-color specs

pd.set_option('plotting.backend', 'matplotlib')  # pandas_bokeh, plotly, etc
plt.rcParams['figure.figsize'] = '20,10'  # landscape format figures
plt.rcParams['legend.fontsize'] = 12  # legend font size
plt.rcParams['axes.labelsize'] = 12  # axis label font size
plt.rcParams['figure.dpi'] = 144  # high-dpi monitors support
plt.style.use('ggplot')  # emulate ggplot style

# For latex-quality, i.e., publication quality legends and labels on graphs.
# Warning: you must have installed LaTeX on your system.
from matplotlib import rc

rc('font', family='Lora,serif', weight='bold')
rc('text', usetex=True)  # Enable it selectively
rc('font', size=14)

import warnings

warnings.filterwarnings('ignore')  # suppress warning
# ----------------------------- Main program -----------------------------------


subject = st.sidebar.radio("Select:", ["Correlation", "Sample Size Regularization"])


if subject == 'Correlation':
    render_correlation()
else:
    render_sample_size_regularization()




