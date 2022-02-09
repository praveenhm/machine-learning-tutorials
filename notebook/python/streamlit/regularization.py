import numpy as np
import pandas as pd
import streamlit as st
from covariance import create_data, altair_chart, matplotlib_chart

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


@st.cache
def generate_data(sample_size: int = 10) -> pd.DataFrame:
    """
    Generates a sine-wave dataset, with some gaussian noise added.
    """
    x = np.linspace(0, 2 * np.pi, sample_size)
    y =  np.sin(x) + 0.25*np.random.normal(0, 1, sample_size)
    data = pd.DataFrame(data={'x': x, 'y': y})
    return data


def fit_and_plot(sample: pd.DataFrame, degree: int = 1):
    """
    Fit a polynomial regression model to the data and plot the predictions over data.
    """
    X, y = sample[['x']], sample['y']
    sample_size = sample.shape[0]
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X, y)
    XX = pd.DataFrame(data={'x': np.linspace(sample.x.min(), sample.x.max(), 2000)})
    yhat = model.predict(XX)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel(r'$x\longrightarrow$')
    ax.set_ylabel(r'$y\longrightarrow$')
    ax.set_title(rf'Polynomial regression of degree {degree}, sample size of {sample_size}')
    ax.scatter(sample['x'], sample['y'], alpha=0.5, s=150, color='salmon')
    ax.scatter(XX.x, yhat, color='maroon', s=10, label="Model Predictions")
    ax.legend(loc='best')
    plt.tight_layout()
    st.pyplot(fig)


def render_sample_size_regularization():
    st.title("Regularization and Sample Size")
    lpara = '''
    The above graph illustrates how:
    ### degree
     the degree of a polynomial regression affects the prediction model
    ### sample size
     the amount of data affects the shape of the polynomial regression line
    '''
    rpara = '''
    ### Observation
    Observe how the linear plot (polynomial degree =1) stays essentially
     unchanged as we make the sample size very large. 
    Also, observe how we have overfitting of the plot as we take polynomials of greater degree, 
    when the dataset is still sparse. As the dataset size is increased, regularization takes place for these high 
    degree polynomials.

    Finally, notice the presence of the Runge's phenomenon at the periphery of the data.'''


    left, right = st.beta_columns(2)
    n_str = r'''\text{Select the sample size of the dataset } \mathscr{D}'''
    left.latex(n_str)
    sample_size = left.slider("Sample Size", max_value=200, min_value=5, value=5, step=1)
    deg_str = r'''\text{Train a model of polynomial degree } d'''
    right.latex(deg_str)
    degree = right.slider("Degree", max_value=20, min_value=1, value=1, step=1)

    sample = generate_data(sample_size)
    fit_and_plot(sample, degree)
    left, right = st.beta_columns(2)
    left.markdown(lpara)
    right.markdown(rpara)