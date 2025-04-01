import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Set page title and layout
st.set_page_config(page_title="S&P 500 Investment Simulator", layout="wide")
st.title("S&P 500 Investment Simulator")
st.write("Explore how missing the best or worst market days affects your investment returns")

# Initialize session state for caching data
if 'sp500_data' not in st.session_state:
    st.session_state.sp500_data = None

# Function to load S&P 500 data
def load_sp500_data(years):
    with st.spinner("Loading S&P 500 historical data..."):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        # Fetch S&P 500 data using yfinance
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)
        
        # Print column information for debugging
        st.write(f"Columns available: {sp500.columns.tolist()}")
        
        # Get the close price column
        if 'Adj Close' in sp500.columns:
            return sp500['Adj Close']
        else:
            return sp500['Close']

# Sidebar for parameters
st.sidebar.header("Investment Parameters")

# Parameter: Time period
years = st.sidebar.slider("Investment Period (Years)", min_value=5, max_value=30, value=20, step=1)

# Load data if needed or if years changed
if st.session_state.sp500_data is None or st.sidebar.button("Reload Data"):
    st.session_state.sp500_data = load_sp500_data(years)
    st.sidebar.success(f"Loaded {years} years of S&P 500 data")

# More parameters
investment_amount = st.sidebar.number_input("Initial Investment ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
monthly_contribution = st.sidebar.number_input("Monthly Contribution ($)", min_value=0, max_value=10000, value=0, step=100)
missed_best_days = st.sidebar.slider("Best Days Missed", min_value=0, max_value=30, value=10, step=1)
missed_worst_days = st.sidebar.slider("Worst Days Missed", min_value=0, max_value=30, value=0, step=1)

# Function to calculate returns with missed days
def calculate_returns(sp500_data, investment_amount, monthly_contribution, missed_best_days, missed_worst_days):
    # Calculate daily returns
    daily_returns = sp500_data.pct_change()
    daily_returns = daily_returns.fillna(0)
    
    # Debug info
    st.write(f"Type of daily_returns: {type(daily_returns)}")
    if isinstance(daily_returns, pd.DataFrame):
        st.write(f"Columns in daily_returns: {daily_returns.columns.tolist()}")
    
    # Sort returns for identifying best and worst days
    # Handle both Series and DataFrame
    if isinstance(daily_returns, pd.DataFrame):
        # If it's a DataFrame, we need to specify the column
        if len(daily_returns.columns) > 0:
            sorted_returns = daily_returns.sort_values(by=daily_returns.columns[0], ascending=False)
        else:
            st.error("DataFrame has no columns")
            return None
    else:
        # If it's a Series, we can sort directly
        sorted_returns = daily_returns.sort_values(ascending=False)
    
    # Create modified returns series excluding specified days
    modified_returns = daily_returns.copy()
    
    if missed_best_days > 0:
        best_days = sorted_returns.head(missed_best_days).index
        if isinstance(modified_returns, pd.DataFrame):
            for col in modified_returns.columns:
                modified_returns.loc[best_days, col] = 0
        else:
            modified_returns.loc[best_days] = 0
        
    if missed_worst_days > 0:
        worst_days = sorted_returns.tail(missed_worst_days).index
        if isinstance(modified_returns, pd.DataFrame):
            for col in modified_returns.columns:
                modified_returns.loc[worst_days, col] = 0
        else:
            modified_returns.loc[worst_days] = 0

    # Calculate growth with lump sum investment
    if monthly_contribution == 0:
        # Simple lump sum investment
        cumulative_returns = (1 + modified_returns).cumprod()
        
        # Handle both Series and DataFrame for the calculation
        if isinstance(cumulative_returns, pd.DataFrame):
            investment_growth = investment_amount * cumulative_returns.iloc[:, 0]
            final_value = investment_growth.iloc[-1]
        else:
            investment_growth = investment_amount * cumulative_returns
            final_value = investment_growth.iloc[-1]
            
        total_invested = investment_amount
    else:
        # Initialize array for investment growth with monthly contributions
        dates = modified_returns.index
        investment_growth = pd.Series(index=dates, dtype=float)
        investment_growth.iloc[0] = investment_amount
        
        # Get the actual returns series if it's a DataFrame
        if isinstance(modified_returns, pd.DataFrame):
            mod_returns_series = modified_returns.iloc[:, 0]
        else:
            mod_returns_series = modified_returns
        
        current_value = investment_amount
        total_invested = investment_amount
        
        for i, date in enumerate(dates):
            if i == 0:
                investment_growth.iloc[i] = current_value
                continue
                
            # Apply daily return
            if isinstance(modified_returns, pd.DataFrame):
                current_value *= (1 + modified_returns.iloc[i, 0])
            else:
                current_value *= (1 + modified_returns.iloc[i])
            
            # Check if this is a month-end date (approximate for contributions)
            if i > 0 and date.month != dates[i-1].month:
                current_value += monthly_contribution
                total_invested += monthly_contribution
                
            investment_growth.iloc[i] = current_value
            
        final_value = current_value

    return {
        'initial_investment': investment_amount,
        'total_invested': total_invested,
        'final_value': final_value,
        'total_return_pct': (final_value - total_invested) / total_invested * 100,
        'investment_period_years': len(daily_returns) / 252,  # Assuming 252 trading days per year
        'annualized_return': ((final_value / total_invested) ** (1 / (len(daily_returns) / 252))) - 1,
        'investment_growth': investment_growth,
        'best_days': sorted_returns.head(missed_best_days) if missed_best_days > 0 else None,
        'worst_days': sorted_returns.tail(missed_worst_days) if missed_worst_days > 0 else None
    }

# Run simulation if data is loaded
if st.session_state.sp500_data is not None:
    try:
        # Debug info about data type
        st.write(f"Type of loaded data: {type(st.session_state.sp500_data)}")
        if isinstance(st.session_state.sp500_data, pd.DataFrame):
            st.write(f"Columns in loaded data: {st.session_state.sp500_data.columns.tolist()}")
        
        # Run the calculation
        result = calculate_returns(
            st.session_state.sp500_data,
            investment_amount,
            monthly_contribution,
            missed_best_days,
            missed_worst_days
        )
        
        if result is None:
            st.error("Calculation failed. Please check the data.")
        else:
            # Create columns for the summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Invested", f"${result['total_invested']:,.2f}")
                
            with col2:
                st.metric("Final Value", f"${result['final_value']:,.2f}")
                
            with col3:
                st.metric("Total Return", f"{result['total_return_pct']:.2f}%")
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.metric("Investment Period", f"{result['investment_period_years']:.2f} years")
                
            with col5:
                st.metric("Annualized Return", f"{result['annualized_return']*100:.2f}%")
                
            with col6:
                st.metric("Monthly Contribution", f"${monthly_contribution:,.2f}")
            
            # Create a visualization of investment growth
            st.subheader("Investment Growth Over Time")
            
            # Plot the investment growth using Plotly
            fig = px.line(
                x=result['investment_growth'].index,
                y=result['investment_growth'].values,
                labels={"x": "Date", "y": "Value ($)"},
                title=f"Investment Growth (Missing {missed_best_days} best days and {missed_worst_days} worst days)"
            )
            
            # Add a horizontal line for the total invested amount
            fig.add_hline(
                y=result['total_invested'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Total Invested: ${result['total_invested']:,.0f}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show comparison with other scenarios
            st.subheader("Comparison with Other Scenarios")
            
            # Calculate additional scenarios for comparison
            scenarios = [
                {"name": "Regular Buy & Hold", "best": 0, "worst": 0},
                {"name": f"Missing {missed_best_days} Best Days", "best": missed_best_days, "worst": 0},
                {"name": f"Missing {missed_worst_days} Worst Days", "best": 0, "worst": missed_worst_days},
                {"name": f"Missing Both", "best": missed_best_days, "worst": missed_worst_days}
            ]
            
            comparison_data = []
            
            for scenario in scenarios:
                if scenario["best"] == missed_best_days and scenario["worst"] == missed_worst_days:
                    # This is the current scenario, use the result we already calculated
                    scenario_result = result
                else:
                    # Calculate this scenario
                    scenario_result = calculate_returns(
                        st.session_state.sp500_data,
                        investment_amount,
                        monthly_contribution,
                        scenario["best"],
                        scenario["worst"]
                    )
                
                if scenario_result is not None:
                    comparison_data.append({
                        "Scenario": scenario["name"],
                        "Final Value": scenario_result["final_value"],
                        "Total Return": scenario_result["total_return_pct"],
                        "Annualized Return": scenario_result["annualized_return"] * 100
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Plot comparison bar chart
                fig1 = px.bar(
                    comparison_df,
                    x="Scenario",
                    y="Final Value",
                    title="Final Investment Value by Scenario",
                    labels={"Final Value": "Final Value ($)"},
                    color="Scenario"
                )
                
                fig2 = px.bar(
                    comparison_df,
                    x="Scenario",
                    y="Annualized Return",
                    title="Annualized Return by Scenario",
                    labels={"Annualized Return": "Annualized Return (%)"},
                    color="Scenario"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Display best/worst days if missed
            if missed_best_days > 0 or missed_worst_days > 0:
                st.subheader("Days Excluded from Investment")
                
                tab1, tab2 = st.tabs(["Best Days Missed", "Worst Days Missed"])
                
                with tab1:
                    if missed_best_days > 0 and result['best_days'] is not None:
                        best_days_df = result['best_days'].reset_index()
                        if isinstance(result['best_days'], pd.DataFrame):
                            best_days_df.columns = ['Date', 'Return (%)']
                            best_days_df['Return (%)'] = best_days_df['Return (%)'] * 100
                        else:
                            best_days_df.columns = ['Date', 'Return (%)']
                            best_days_df['Return (%)'] = best_days_df['Return (%)'] * 100
                        
                        st.write(f"These are the {missed_best_days} best trading days you missed:")
                        st.dataframe(best_days_df, use_container_width=True)
                
                with tab2:
                    if missed_worst_days > 0 and result['worst_days'] is not None:
                        worst_days_df = result['worst_days'].reset_index()
                        if isinstance(result['worst_days'], pd.DataFrame):
                            worst_days_df.columns = ['Date', 'Return (%)']
                            worst_days_df['Return (%)'] = worst_days_df['Return (%)'] * 100
                        else:
                            worst_days_df.columns = ['Date', 'Return (%)']
                            worst_days_df['Return (%)'] = worst_days_df['Return (%)'] * 100
                        
                        st.write(f"These are the {missed_worst_days} worst trading days you missed:")
                        st.dataframe(worst_days_df, use_container_width=True)
            
            # Display raw data
            if st.checkbox("Show Raw Data"):
                st.subheader("S&P 500 Historical Data")
                st.dataframe(st.session_state.sp500_data, use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Try adjusting your parameters or reloading the data.")
else:
    st.warning("Please load the S&P 500 data first.") 