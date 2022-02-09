import streamlit as st
from covariance import create_data, altair_chart, matplotlib_chart

# ----------------------------- Main program -----------------------------------
st.title("Correlation")


'''
Use the below interactive graph to get a sense of how for different correlation values look.  
Play around with the slider to see data distribution for different values of the correlation.
(This only uses a linearly related data to illustrate the meaning of correlation.)


'''

message = r'''\text{Select a value of the correlation } \mathbf{\rho}'''
st.latex(message)

# Generate data with specific correlation.
df = create_data()

# Now generate the visualization using altair
altair_chart(df)

# Alternatively, if you want to see the visualization using matplotlib,
# uncomment the below.
# matplotlib_chart(df)
