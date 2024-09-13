# Code developed by Jamie Yan, a Hult master's student. Supervised by Michael de la Maza.
# Also available at: https://github.com/jamie1016jamie1016/100m_vs_longjump/blob/main/app.py
# Online app: https://100mvslongjumpdashboard.streamlit.app/

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from io import StringIO

df = pd.read_csv('csv.csv')

# Data cleaning steps
df['Gender'] = df['Gender'].str.upper()
df['100M'] = pd.to_numeric(df['100M'], errors='coerce')
df['Jump'] = pd.to_numeric(df['Jump'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.dropna(subset=['100M', 'Jump', 'Age', 'Gender'], inplace=True)
df = df[df['100M'] < 50]

# Define regression function
def perform_regression(x, y, regression_type):
    # Remove NaN values
    df_reg = pd.DataFrame({'x': x, 'y': y}).dropna()
    
    if regression_type == 'Linear':
        X = sm.add_constant(df_reg['x'])  # Adds a constant term to the predictor
        model = sm.OLS(df_reg['y'], X)
        results = model.fit()
        # Get equation parameters
        a = results.params['const']
        b = results.params['x']
        # Equation string
        equation = f'y = {a:.3f} + {b:.3f} * x'
        # R-squared
        r_squared = results.rsquared
        # Predicted values
        y_pred = results.predict(X)
        
    elif regression_type == 'Exponential':
        # Filter out y <= 0
        df_reg = df_reg[df_reg['y'] > 0]
        if df_reg.empty:
            raise ValueError("No data available after filtering out non-positive y values.")
        x = df_reg['x']
        y = df_reg['y']
        y_ln = np.log(y)
        X = sm.add_constant(x)
        model = sm.OLS(y_ln, X)
        results = model.fit()
        a_ln = results.params['const']
        b = results.params['x']
        a = np.exp(a_ln)
        equation = f'y = {a:.3f} * exp({b:.3f} * x)'
        y_pred = a * np.exp(b * x)
        # Calculate R-squared manually
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        
    elif regression_type == 'Logarithmic':
        # Filter out x <= 0
        df_reg = df_reg[df_reg['x'] > 0]
        if df_reg.empty:
            raise ValueError("No data available after filtering out non-positive x values.")
        x = df_reg['x']
        y = df_reg['y']
        x_ln = np.log(x)
        X = sm.add_constant(x_ln)
        model = sm.OLS(y, X)
        results = model.fit()
        a = results.params['const']
        b = results.params['x']
        equation = f'y = {a:.3f} + {b:.3f} * ln(x)'
        y_pred = results.predict(X)
        r_squared = results.rsquared
        
    else:
        raise ValueError("Invalid regression type")
    
    return equation, r_squared, x, y, y_pred


# Streamlit app code
st.title('Interactive Regression Dashboard')

# Sidebar for user inputs
st.sidebar.header('Filter Options')

gender_options = ['M', 'W', 'Both']
gender = st.sidebar.selectbox('Gender:', gender_options, index=2)

min_age = int(df['Age'].min())
max_age = int(df['Age'].max())
age_range = st.sidebar.slider('Age Range:',
                                min_value=min_age,
                                max_value=max_age,
                                value=(min_age, max_age),
                                step=5)

regression_type = st.sidebar.selectbox('Regression Type:', ['Linear', 'Exponential', 'Logarithmic'])

# Filter data based on selections
filtered_df = df.copy()
if gender != 'Both':
    filtered_df = filtered_df[filtered_df['Gender'] == gender]
filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    x = filtered_df['100M']
    y = filtered_df['Jump']

    # Perform regression
    try:
        equation, r_squared, x_vals, y_vals, y_pred = perform_regression(x, y, regression_type)
    except Exception as e:
        st.error(f"Error in regression: {e}")
        st.stop()

    # Sort x values for plotting the regression line
    sorted_indices = np.argsort(x_vals)
    x_sorted = x_vals.iloc[sorted_indices]
    y_pred_sorted = y_pred.iloc[sorted_indices]

    # Create the plot
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers',
        name='Data',
        marker=dict(color='blue')
    ))

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_sorted, y=y_pred_sorted,
        mode='lines',
        name=f'{regression_type} Regression',
        line=dict(color='red')
    ))

    # Update layout
    gender_label = 'Both Genders' if gender == 'Both' else ('Male' if gender == 'M' else 'Female')
    fig.update_layout(
        title=f'100m Time vs. Long Jump Distance ({gender_label}, Ages {age_range[0]}-{age_range[1]})',
        xaxis_title='100m Time (s)',
        yaxis_title='Long Jump Distance (m)',
        height=500,
        width=700
    )

    # Add equation and R-squared to the plot
    fig.add_annotation(
        x=0.5, y=1.1, xref='paper', yref='paper',
        text=f'{equation}<br>R² = {r_squared:.4f}',
        showarrow=False,
        font=dict(size=12),
        align='left',
        bordercolor='black',
        borderwidth=1
    )

    # Display the plot
    st.plotly_chart(fig)
