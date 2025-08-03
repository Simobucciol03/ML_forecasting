import pandas as pd
from tti.indicators import * 
import matplotlib.pyplot as plt
import inspect
import tti.indicators as ti
from connection_database.login_mysql import DataProvider
from tti.utils.constants import TRADE_SIGNALS
from tti.indicators.properties import indicators_dic as id




def execute_simulation(indicator_object, close_values, output_file=None,
                       add_info=None, figures_output_path=None, **kwargs):
    """
    Executes trading simulation for all the indicators in the list for the
    input data.
    """

    indicator = indicator_object(**kwargs)

    print('\nSimulation for technical indicator:', type(indicator).__name__)

    # Get full indicator data
    data_indicator = indicator.getTiData()

    # Ensure indicator data is available
    if data_indicator is None or data_indicator.empty:
        print(f"Skipping {type(indicator).__name__} - No data available.")
        return None  # Skip this indicator

    box.append(data_indicator)  # Store indicator values

    # Compute trading signals for all time periods
    signals = []

    # Ensure we are using the correct indicator column
    indicator_col = data_indicator.columns[0]  # First column of indicator data

    # Avoid index out-of-bounds
    min_length = min(len(close_values), len(data_indicator))

    for i in range(1, min_length):  # Ensure loop does not exceed data limits
        price_slope = close_values['close'].iloc[i] - close_values['close'].iloc[i - 1]
        indicator_value = data_indicator[indicator_col].iloc[i]

        if (price_slope < 0 < indicator_value) or (price_slope > 0 and indicator_value > 0):
            signals.append(TRADE_SIGNALS['buy'][1])  # Store only -1, 0, or 1
        elif (price_slope > 0 > indicator_value) or (price_slope < 0 and indicator_value < 0):
            signals.append(TRADE_SIGNALS['sell'][1])
        else:
            signals.append(TRADE_SIGNALS['hold'][1])

    # Convert to DataFrame and align index
    signal_df = pd.DataFrame(signals, index=close_values.index[1:min_length], columns=[type(indicator).__name__])

    signal.append(signal_df)  # Store full signal series

    return data_indicator, 

# Create a list to store the values of the indicators
box = []
signal = []    # added

# In the main section, after getting the data from DataProvider
if __name__ == '__main__':
    dp = DataProvider() 
    da = dp.main()
    df = da[0]
    
    # Convert the index to DatetimeIndex if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        # Check if there's a date/datetime column
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            # Set the first date column found as index
            df = df.set_index(date_columns[0])
            # Make sure it's a DatetimeIndex
            df.index = pd.to_datetime(df.index)
        else:
            print("No date column found in the dataframe. Please ensure your data has a date column.")
            exit(1)

    # Run simulation for all the indicators implemented in the tti.indicators package
    for x in inspect.getmembers(ti):

        if inspect.isclass(x[1]):

            # Moving Average includes five indicators
            if x[1] == ti.MovingAverage:
                execute_simulation(indicator_object=x[1],
                    close_values=df[['close']],
                    #output_file=out_file, 
                    add_info='simple',
                    #figures_output_path='./figures/',
                    input_data=df,
                    ma_type='simple')

                execute_simulation(indicator_object=x[1],
                    close_values=df[['close']],
                    #output_file=out_file, 
                    add_info='exponential',
                    #figures_output_path='./figures/',
                    input_data=df,
                    ma_type='exponential')

                execute_simulation(indicator_object=x[1],
                    close_values=df[['close']],
                    #output_file=out_file, 
                    add_info='time_series',
                    #figures_output_path='./figures/',
                    input_data=df,
                    ma_type='time_series')

                execute_simulation(indicator_object=x[1],
                    close_values=df[['close']],
                    #output_file=out_file, 
                    add_info='triangular',
                    #figures_output_path='./figures/',
                    input_data=df,
                    ma_type='triangular')

                execute_simulation(indicator_object=x[1],
                    close_values=df[['close']],
                    #output_file=out_file, 
                    add_info='variable',
                    #figures_output_path='./figures/',
                    input_data=df,
                    ma_type='variable')

            # Stochastic Oscillator includes two indicators
            elif x[1] == ti.StochasticOscillator:
                execute_simulation(indicator_object=x[1],
                    close_values=df[['close']],
                    #output_file=out_file,
                    add_info='fast',
                    #figures_output_path='./figures/',
                    input_data=df,
                    k_slowing_periods=1)

                execute_simulation(indicator_object=x[1],
                    close_values=df[['close']],
                    #output_file=out_file, 
                    add_info='slow',
                    #figures_output_path='./figures/',
                    input_data=df,
                    k_slowing_periods=3)

            else:
                execute_simulation(indicator_object=x[1],
                    close_values=df[['close']],
                    #output_file=out_file, 
                    add_info=None,
                    #figures_output_path='./figures/',
                    input_data=df)

# Merge box and signal with df
if box and signal:
    # Concatenate both lists of indicators and signals
    box = pd.concat(box, axis=1)
    signal = pd.concat(signal, axis=1)
else:
    if not box:
        print("No indicators found.")
    if not signal:
        print("No signals found.")

# Merge all stored data into a single DataFrame
df_0 = pd.concat([df, box], axis=1)
df_1 = pd.concat([df, signal], axis=1)


######################################################################
########### This part will be removed in the final version ###########
######################################################################


tickers_input = str(da[1][0])



# # Salva il DataFrame finale con tutti gli indicatori
df_0.to_csv(f"/Users/simonebucciol/Desktop/project/csv_indicator/{tickers_input}_with_indicators.csv")
df_1.to_csv(f'/Users/simonebucciol/Desktop/project/csv_trading_signal/{tickers_input}_signal.csv')



######################################################################
########### This part will be removed in the final version ###########
######################################################################

# from tti.utils.plot import linesGraph

# # Definisci colori e trasparenze per i grafici
# lines_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
# alpha_values = [0.8, 0.6, 0.4, 0.7, 0.5, 0.3]

# # Grafico degli indicatori tecnici
# plt_indicators = linesGraph(
#     data=df_0,  # Usa il DataFrame con gli indicatori
#     y_label='Indicator Values',
#     title='Technical Indicators',
#     lines_color=lines_colors,
#     alpha_values=alpha_values,
#     areas=None
# )

# # Mostra il grafico
# plt_indicators.show()



# # Grafico dei segnali di trading
# plt_signals = linesGraph(
#     data=df_1,  # Usa il DataFrame con i segnali
#     y_label='Trade Signals (-1, 0, 1)',
#     title='Trading Signals',
#     lines_color=['black', 'green', 'red'],  # Nero per hold, verde per buy, rosso per sell
#     alpha_values=[1, 1, 1],  # Massima visibilità
#     areas=None
# )

# # Mostra il grafico
# plt_signals.show()