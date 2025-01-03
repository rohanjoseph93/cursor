import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

def fetch_btc_data():
    # Fetch BTC data from its inception
    df = yf.download('BTC-USD', start='2010-07-17')  # First recorded BTC price
    df['Days'] = (df.index - df.index[0]).days
    df['Log_Price'] = np.log(df['Adj Close'])
    return df

def calculate_log_bands(df):
    # Calculate logarithmic regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['Days'], df['Log_Price'])
    
    # Calculate regression line
    log_reg = slope * df['Days'] + intercept
    
    # Calculate standard deviation of residuals
    residuals = df['Log_Price'] - log_reg
    std_dev = np.std(residuals)
    
    # Calculate bands (±2 standard deviations)
    upper_band = log_reg + 2 * std_dev
    lower_band = log_reg - 2 * std_dev
    
    return slope, intercept, std_dev, log_reg, upper_band, lower_band

def forecast_to_2030(df, slope, intercept, std_dev):
    # Create future dates until 2030
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, end='2030-12-31', freq='D')
    future_days = (future_dates - df.index[0]).days
    
    # Calculate forecasted values
    future_log_reg = slope * future_days + intercept
    future_upper = future_log_reg + 2 * std_dev
    future_lower = future_log_reg - 2 * std_dev
    
    # Convert logarithmic values to prices
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Regression': np.exp(future_log_reg),
        'Upper_Band': np.exp(future_upper),
        'Lower_Band': np.exp(future_lower)
    }).set_index('Date')
    
    return forecast_df

def plot_results(df, forecast_df):
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.semilogy(df.index, df['Adj Close'], label='Historical Price', color='blue')
    
    # Plot regression and bands for historical data
    plt.semilogy(df.index, np.exp(df['Log_Regression']), 'r--', label='Log Regression')
    plt.semilogy(df.index, np.exp(df['Upper_Band']), 'g--', label='Upper Band (+2σ)')
    plt.semilogy(df.index, np.exp(df['Lower_Band']), 'g--', label='Lower Band (-2σ)')
    
    # Plot forecasted values
    plt.semilogy(forecast_df.index, forecast_df['Regression'], 'r:', label='Forecast')
    plt.semilogy(forecast_df.index, forecast_df['Upper_Band'], 'g:', label='Forecast Bands')
    plt.semilogy(forecast_df.index, forecast_df['Lower_Band'], 'g:')
    
    plt.title('Bitcoin Price with Logarithmic Regression Bands (2010-2030)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD) - Log Scale')
    plt.grid(True)
    plt.legend()
    plt.savefig('bitcoin_log_forecast.png')
    plt.show()

if __name__ == "__main__":
    # Fetch and prepare data
    df = fetch_btc_data()
    
    # Calculate regression and bands
    slope, intercept, std_dev, log_reg, upper_band, lower_band = calculate_log_bands(df)
    
    # Add regression and bands to dataframe
    df['Log_Regression'] = log_reg
    df['Upper_Band'] = upper_band
    df['Lower_Band'] = lower_band
    
    # Generate forecast
    forecast_df = forecast_to_2030(df, slope, intercept, std_dev)
    
    # Plot results
    plot_results(df, forecast_df)
    
    # Print some statistics
    print("\nForecast Statistics for 2030:")
    last_forecast = forecast_df.iloc[-1]
    print(f"Regression Line Price: ${last_forecast['Regression']:,.2f}")
    print(f"Upper Band Price: ${last_forecast['Upper_Band']:,.2f}")
    print(f"Lower Band Price: ${last_forecast['Lower_Band']:,.2f}")