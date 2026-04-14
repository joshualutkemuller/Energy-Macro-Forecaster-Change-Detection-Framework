Summary:

This project uses a Vector Autoregression (VAR) model to analyze the relationship between oil prices (WTI), rig count, and industrial production (INDPRO). The goal is to understand how these variables interact over time and to generate short-term forecasts.



Approach:

• Loaded and cleaned monthly time series data

• Selected key variables: oil prices, rig count, industrial production

• Tested for stationarity using ADF tests

• Transformed data into monthly % changes to ensure stationarity

• Fitted a VAR model and selected lag using AIC (lag = 1)

• Analyzed relationships using model coefficients and impulse response functions

• Generated a 12-month forecast



Key Findings:

• Oil price changes have a positive impact on rig count

• Rig count shows strong persistence over time

• Industrial production responds to oil price changes but less to rig activity

• A shock to oil prices leads to a gradual increase in rig activity



Forecast Insight:

• Rig count growth is expected to remain slightly negative in the short term

• Trend stabilizes over time with smaller declines



Takeaways:

• Oil prices are a key driver of drilling activity

• Effects in the energy sector occur with lags, not instantly

• VAR models effectively capture interconnected economic dynamics

