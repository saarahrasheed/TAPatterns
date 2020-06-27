# TAPatterns


## Technical Analysis Pattern Recognition


The trader_class conducts trades and assesses the performance of the trades.
The trading data is not included but template for the file is included under \data\DataFile.csv 


### Understanding the Trader Class 
The Trader class can be used as a template for testing trading strategies based on logic derived from price-action analysis and chart readings.
The trader class uses both long and short as well as long-only strategy to conduct trades based on patterns specified in the trade assess_condition() method. 


The assessment of trading strategies is done via:
1. Risk Reward Ratio
2. Drawdowns
3. Win-Loss Ratio
4. Aggregated Earnings