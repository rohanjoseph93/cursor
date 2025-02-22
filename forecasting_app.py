import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from datetime import datetime
import io
from itertools import product

def load_data(uploaded_file, file_type):
    """Load data from uploaded file based on file type"""
    if file_type == 'csv':
        # Add CSV delimiter options
        delimiter = st.selectbox('Select CSV delimiter:', [',', ';', '|', '\t'])
        return pd.read_csv(uploaded_file, delimiter=delimiter)
    else:  # Excel
        return pd.read_excel(uploaded_file)

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_forecast_plot(df, forecast, metric_name):
    """Create plotly visualization of forecast"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        name='Historical',
        mode='markers+lines'
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        mode='lines'
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title=f'Forecast for {metric_name}',
        xaxis_title='Date',
        yaxis_title=metric_name,
        hovermode='x unified'
    )
    
    return fig

def calculate_cv_parameters(df):
    """Calculate appropriate cross-validation parameters based on data size"""
    data_length = (df['ds'].max() - df['ds'].min()).days
    
    # If less than 1 year of data
    if data_length < 365:
        initial = f'{int(data_length * 0.5)} days'
        period = f'{int(data_length * 0.2)} days'
        horizon = f'{int(data_length * 0.1)} days'
    # If less than 2 years of data
    elif data_length < 730:
        initial = f'{int(data_length * 0.6)} days'
        period = f'{int(data_length * 0.2)} days'
        horizon = f'{int(data_length * 0.1)} days'
    # If more than 2 years of data
    else:
        initial = '730 days'
        period = '180 days'
        horizon = '365 days'
    
    return initial, period, horizon

def find_best_parameters(prophet_df, initial, period, horizon, forecast_periods):
    """Find best parameters using grid search"""
    # Parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    best_mape = float('inf')
    best_params = None
    best_forecast = None
    best_model = None
    
    # Use streamlit progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, params in enumerate(all_params):
        status_text.text(f'Testing parameter set {i+1}/{len(all_params)}...')
        progress_bar.progress((i + 1) / len(all_params))
        
        try:
            # Create and fit model with current parameters
            model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                seasonality_mode=params['seasonality_mode']
            )
            
            model.fit(prophet_df)
            
            # Create future dates for this parameter set
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = model.predict(future)
            
            # Perform cross validation
            cv_results = cross_validation(
                model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="threads"
            )
            
            # Calculate MAPE
            mape = calculate_mape(cv_results['y'], cv_results['yhat'])
            
            # Update best parameters if MAPE is lower
            if mape < best_mape:
                best_mape = mape
                best_params = params
                best_forecast = forecast
                best_model = model
        
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params, best_mape, best_forecast, best_model

def main():
    st.title('Time Series Forecasting App')
    st.write('Upload your Excel or CSV file with a datetime column and a metric to forecast')
    
    # File type selection
    file_type = st.radio('Select file type:', ['excel', 'csv'])
    
    # File uploader based on type
    if file_type == 'excel':
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
            df = load_data(uploaded_file, file_type)
            
            # Display first few rows of the data
            st.write("Preview of uploaded data:")
            st.write(df.head())
            
            # Select columns
            st.write("Select your columns:")
            date_col = st.selectbox('Select date column:', df.columns)
            metric_col = st.selectbox('Select metric to forecast:', df.columns)
            
            # Data preprocessing options
            st.sidebar.header('Data Preprocessing')
            handle_missing = st.sidebar.selectbox(
                'Handle missing values:',
                ['Drop', 'Forward Fill', 'Backward Fill', 'Linear Interpolation']
            )
            
            # Handle missing values
            if handle_missing == 'Drop':
                df = df.dropna()
            elif handle_missing == 'Forward Fill':
                df = df.fillna(method='ffill')
            elif handle_missing == 'Backward Fill':
                df = df.fillna(method='bfill')
            elif handle_missing == 'Linear Interpolation':
                df = df.interpolate(method='linear')
            
            # Prepare data for Prophet
            prophet_df = df[[date_col, metric_col]].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Ensure datetime format
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Calculate appropriate cross-validation parameters
            initial, period, horizon = calculate_cv_parameters(prophet_df)
            
            # Forecast Settings
            st.sidebar.header('Forecast Settings')
            forecast_periods = st.sidebar.slider('Forecast Periods (days)', 7, 365, 30)
            
            # Parameter Selection Method
            parameter_method = st.radio(
                'Choose Parameter Selection Method:',
                ['Auto-Optimize', 'Manual Parameters'],
                help='Auto-Optimize will find the best parameters based on MAPE'
            )
            
            # Initialize session state for best parameters if not exists
            if 'best_params' not in st.session_state:
                st.session_state.best_params = None
                st.session_state.best_mape = None
            
            # Manual Parameter Controls
            if parameter_method == 'Manual Parameters' or st.session_state.best_params is not None:
                st.sidebar.header('Model Parameters')
                
                # If we have optimized parameters, use them as defaults
                default_params = st.session_state.best_params if st.session_state.best_params else {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 1.0,
                    'holidays_prior_scale': 1.0,
                    'seasonality_mode': 'additive'
                }
                
                manual_params = {
                    'changepoint_prior_scale': st.sidebar.slider(
                        'Changepoint Prior Scale', 
                        0.001, 0.5, 
                        value=default_params.get('changepoint_prior_scale', 0.05)
                    ),
                    'seasonality_prior_scale': st.sidebar.slider(
                        'Seasonality Prior Scale', 
                        0.01, 10.0, 
                        value=default_params.get('seasonality_prior_scale', 1.0)
                    ),
                    'holidays_prior_scale': st.sidebar.slider(
                        'Holidays Prior Scale', 
                        0.01, 10.0, 
                        value=default_params.get('holidays_prior_scale', 1.0)
                    ),
                    'seasonality_mode': st.sidebar.selectbox(
                        'Seasonality Mode',
                        ['additive', 'multiplicative'],
                        index=0 if default_params.get('seasonality_mode', 'additive') == 'additive' else 1
                    )
                }
            
            # Generate Forecast Button
            if parameter_method == 'Auto-Optimize':
                if st.button('Generate Optimized Forecast'):
                    with st.spinner('Finding optimal parameters and generating forecast...'):
                        st.info('This may take a few minutes while we test different parameter combinations...')
                        
                        best_params, best_mape, forecast, model = find_best_parameters(
                            prophet_df, initial, period, horizon, forecast_periods
                        )
                        
                        # Store best parameters in session state
                        st.session_state.best_params = best_params
                        st.session_state.best_mape = best_mape
                        
                        # Display results
                        st.success(f'Best parameters found (MAPE: {best_mape:.2f}%):')
                        st.json(best_params)
                        
                        # Add option to use these parameters
                        st.info('You can now switch to "Manual Parameters" to fine-tune these results')
                        
                        # Display plots and downloads
                        display_results(prophet_df, forecast, model, metric_col)
            
            else:  # Manual Parameters
                if st.button('Generate Forecast with Current Parameters'):
                    with st.spinner('Generating forecast...'):
                        # Create and fit model with manual parameters
                        model = Prophet(
                            changepoint_prior_scale=manual_params['changepoint_prior_scale'],
                            seasonality_prior_scale=manual_params['seasonality_prior_scale'],
                            holidays_prior_scale=manual_params['holidays_prior_scale'],
                            seasonality_mode=manual_params['seasonality_mode']
                        )
                        
                        model.fit(prophet_df)
                        
                        # Generate forecast
                        future = model.make_future_dataframe(periods=forecast_periods)
                        forecast = model.predict(future)
                        
                        # Calculate MAPE
                        try:
                            cv_results = cross_validation(
                                model, initial=initial, period=period, horizon=horizon
                            )
                            mape = calculate_mape(cv_results['y'], cv_results['yhat'])
                            st.write(f'Model MAPE: {mape:.2f}%')
                        except Exception as cv_error:
                            st.warning(f'Could not calculate MAPE: {str(cv_error)}')
                        
                        # Display plots and downloads
                        display_results(prophet_df, forecast, model, metric_col)
            
        except Exception as e:
            st.error(f'Error: {str(e)}')

def display_results(prophet_df, forecast, model, metric_col):
    """Helper function to display forecast results"""
    # Plot forecast
    fig = create_forecast_plot(prophet_df, forecast, metric_col)
    st.plotly_chart(fig)
    
    # Component plots
    st.write("Trend Components")
    fig_comp = model.plot_components(forecast)
    st.pyplot(fig_comp)
    
    # Download options
    st.write("Download Forecast Results:")
    forecast_csv = forecast.to_csv(index=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as CSV",
            data=forecast_csv,
            file_name="forecast.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            forecast.to_excel(writer, index=False)
        
        st.download_button(
            label="Download as Excel",
            data=buffer.getvalue(),
            file_name="forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main() 
