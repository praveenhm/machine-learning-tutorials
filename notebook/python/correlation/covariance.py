#  Copyright (c) 2020.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#
#  Participants of the SupportVectors workshops are free to use it as part of their learning, and
#  allowed to use it for their personal projects.
#
#  For all other purposes, it may not be reused or shared without the explicit,
#  written permission of SupportVectors.
#
#  Author: Asif Qamar
#


import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt


def create_data() -> pd.DataFrame:
    """
    Creates linearly related data in the shape of the specific correlation
    """
    rho = st.slider('', min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    # Create the zero-centered, normalized covariance matrix.
    mu = np.array([0.0, 0.0])
    sigma = np.array(
        [[1, rho], [rho, 1]]
    )
    # Generate the data.
    data = np.random.multivariate_normal(mu, sigma, size=1000)

    df = pd.DataFrame(data={
        'x': data.T[0],
        'y': data.T[1],
        'delta': abs(data.T[0] - data.T[1])
    })

    return df


def altair_chart(df: pd.DataFrame) -> None:
    """
    Draw the chart using Altair plotting library
    :return: None
    """

    c = alt.Chart(df) \
        .mark_circle(size=100, opacity=0.5) \
        .encode(x=alt.X('x', ),
                y='y',
                color=alt.Color('delta', scale=alt.Scale(scheme='viridis'))) \
        .properties(height=600)
    xline = alt.Chart(pd.DataFrame({'y': [0]})) \
        .mark_rule(size=2, opacity=0.5) \
        .encode(y='y')
    yline = alt.Chart(pd.DataFrame({'x': [0]})) \
        .mark_rule(size=2, opacity=0.5) \
        .encode(x='x')
    chart = c + xline + yline
    st.altair_chart(chart, use_container_width=True)


def matplotlib_chart(df: pd.DataFrame) -> None:
    """
        Draw the chart using default Matplotlib plotting library of python.
        :return: None
        """
    fig, ax = plt.subplots()
    ax.scatter(x=df.x, y=df.y, alpha=0.3)
    ax.spines[['left', 'bottom']].set_position('center')
    ax.spines[['right', 'top']].set_visible(False)
    st.pyplot(fig)




