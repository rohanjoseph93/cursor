import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def perform_monte_carlo_simulation(prices_df, num_simulations=1000, days_to_forecast=2190):  # 2190 days = ~6 years (till 2030)
    # Calculate daily returns
    returns = np.log(1 + prices_df['Adj Close'].pct_change())
    
    # Calculate mean and standard deviation of daily returns
    mu = returns.mean()
    sigma = returns.std()
    
    # Get the last price as starting point
    last_price = prices_df['Adj Close'].iloc[-1]
    
    # Create price paths DataFrame
    simulation_df = pd.DataFrame()
    
    for i in range(num_simulations):
        # Generate random daily returns
        daily_returns = np.random.normal(mu, sigma, days_to_forecast)
        
        # Calculate price path
        price_path = [last_price]
        for r in daily_returns:
            price_path.append(price_path[-1] * np.exp(r))
        
        simulation_df[f'Simulation_{i}'] = price_path[1:]  # Exclude the initial price
    
    return simulation_df

def plot_simulations(historical_df, simulation_df):
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(historical_df.index, historical_df['Adj Close'], label='Historical', color='black')
    
    # Plot future simulations
    future_dates = pd.date_range(start=historical_df.index[-1], 
                                periods=len(simulation_df)+1)[1:]
    
    # Plot a sample of simulations (e.g., 100) to avoid overcrowding
    for i in range(0, min(100, len(simulation_df.columns))):
        plt.plot(future_dates, simulation_df[f'Simulation_{i}'], alpha=0.1, color='blue')
    
    plt.title('Dogecoin Price - Historical and Monte Carlo Simulations until 2030')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('dogecoin_forecast.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Fetch historical data directly from yfinance
    df = yf.download('DOGE-USD', start='2019-01-01')
    
    # Perform Monte Carlo simulation
    simulation_results = perform_monte_carlo_simulation(df)
    
    # Plot results
    # %%
    plot_simulations(df, simulation_results)
  
    # Calculate and print some statistics
    final_prices = simulation_results.iloc[-1]
    print("\nPrice Forecast Statistics for 2030:")
    print(f"Mean Price: ${final_prices.mean():.2f}")
    print(f"Median Price: ${final_prices.median():.2f}")
    print(f"95% Confidence Interval: ${np.percentile(final_prices, 2.5):.2f} - ${np.percentile(final_prices, 97.5):.2f}")
# %%
