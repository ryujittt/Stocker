import sys
import ccxt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QTabWidget,QApplication, QMainWindow, QPushButton, QComboBox, QHBoxLayout, QVBoxLayout, QCompleter,QWidget, QLabel, QCheckBox, QLineEdit
import pandas as pd
import numpy as np
import time
from scipy.signal import correlate
from sklearn.metrics import r2_score
import os
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=120):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the ccxt exchange object
        self.exchange = ccxt.binance()

        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 1000, 600)
        self.setWindowTitle('Stocker')

        # Create ComboBoxes for symbol, timeframe, and limit
        self.symbol_combobox = QComboBox(self)
        self.symbol_combobox.addItems(['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'EUR/AUD', 'GBP/EUR', 'USD/SGD', 'USD/HKD'])
        self.symbol_combobox.setCurrentIndex(0)

        self.timeframe_combobox = QComboBox(self)
        self.timeframe_combobox.addItems(['1m', '5m', '15m', '1h', '4h','1d'])
        self.timeframe_combobox.setCurrentIndex(1)

        self.limit_combobox = QComboBox(self)
        self.limit_combobox.addItems(['5', '10', '15', '20', '50', '100', '200'])
        self.limit_combobox.setCurrentIndex(5)

        self.window_combobox = QComboBox(self)
        self.window_combobox.addItems(['1', '2', '4', '5', '8', '10', '12', '15', '20'])
        self.window_combobox.setCurrentIndex(0)

        self.extension_combobox = QComboBox(self)
        self.extension_combobox.addItems(['2', '3', '4', '5', '10', '15', '20', '25', '30', '35', '40', '45'])
        self.extension_combobox.setCurrentIndex(2)

        self.method_combobox = QComboBox(self)
        self.method_combobox.addItems(['Color', 'Close', 'Slope', 'Second_Slope', 'Volume', 'Adjusted_Volume', 'Average_Price'])
        self.method_combobox.setCurrentIndex(0)

        self.to_plot_combobox = QComboBox(self)
        self.to_plot_combobox.addItems(['Color', 'Close', 'Slope', 'Second_Slope', 'Volume', 'Average_Price'])
        self.to_plot_combobox.setCurrentIndex(1)

        # Create buttons for fetching and plotting data
        self.fetch_button = QPushButton('Forecast and Plot Data', self)
        self.fetch_button.clicked.connect(self.fetch_and_plot_data)

        self.historic_button = QPushButton('Historic Data', self)
        self.historic_button.clicked.connect(self.load_historic_ohlcv_data)

        self.test_button = QPushButton('Current Data', self)
        self.test_button.clicked.connect(self.load_test_ohlcv_data)

        # Create labels for the ComboBoxes and plot
        self.symbol_label = QLabel('Symbol:')
        self.timeframe_label = QLabel('Timeframe:')
        self.limit_label = QLabel('Limit:')
        self.window_label = QLabel('Smoothness:')
        self.extension_label = QLabel('Forecast:')
        self.method_label = QLabel('Method:')
        self.to_plot_label = QLabel('To Plot:')

        self.file_dir_label = QLabel('Dir Key:', self)
        self.file_dir_edit = QLineEdit(self)
        holder = ''
        suggestions = []
        
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

        # Set the full path for the file
        fdir = os.path.join(desktop_path, "ohlcv_data/")
        
        for filename in os.listdir(fdir):
            holder += f"{filename[:-4]},"
            suggestions.append(filename)
            
        self.file_dir_edit.setText('15')
        self.file_dir_edit.setPlaceholderText(holder)
        
        # Create a QCompleter and set it for the QLineEdit
        completer = QCompleter(suggestions, self)
        self.file_dir_edit.setCompleter(completer)
        

        # Create HBoxes for ComboBoxes, labels, and the button
        hbox_comboboxes = QHBoxLayout()
        hbox_comboboxes.addWidget(self.symbol_label)
        hbox_comboboxes.addWidget(self.symbol_combobox)
        hbox_comboboxes.addWidget(self.timeframe_label)
        hbox_comboboxes.addWidget(self.timeframe_combobox)
        hbox_comboboxes.addWidget(self.limit_label)
        hbox_comboboxes.addWidget(self.limit_combobox)
        hbox_comboboxes.addWidget(self.window_label)
        hbox_comboboxes.addWidget(self.window_combobox)
        hbox_comboboxes.addWidget(self.extension_label)
        hbox_comboboxes.addWidget(self.extension_combobox)
        hbox_comboboxes.addWidget(self.method_label)
        hbox_comboboxes.addWidget(self.method_combobox)
        hbox_comboboxes.addWidget(self.to_plot_label)
        hbox_comboboxes.addWidget(self.to_plot_combobox)
        self.log_checkbox = QCheckBox('Logarithmic Y-axis',self)
        self.log_compute = QCheckBox('Logarithmic Computaion',self)
        self.interp_data = QCheckBox('Interpolate',self)


        

        hbox_button = QHBoxLayout()
        hbox_button.addWidget(self.fetch_button)
        hbox_button.addWidget(self.test_button)
        hbox_button.addWidget(self.historic_button)

        self.fourier_series_radio_button = QCheckBox('Fourier Series',self)
        hbox_button.addWidget(self.fourier_series_radio_button)
        hbox_button.addWidget(self.log_checkbox)
        hbox_button.addWidget(self.log_compute)
        hbox_button.addWidget(self.interp_data)


        hbox_button.addWidget(self.file_dir_label)
        hbox_button.addWidget(self.file_dir_edit)
        # Add the checkbox to the layout
        


        # Create a central widget to hold layout
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.addLayout(hbox_comboboxes)
        central_layout.addLayout(hbox_button)

        self.max_price_label = QLabel(
            'Future Price Difference: \nFuture Price Target: \nMin Price Target: \nMax Price Target: \nR-squared: ', self)
        central_layout.addWidget(self.max_price_label)
        
        self.predicted_canvas = MplCanvas(self, width=12, height=8, dpi=100)  # Increase width to 12 for larger subplots
        self.original_canvas = MplCanvas(self, width=12, height=8, dpi=100)  # Increase width to 12 for larger subplots
        self.match_canvas = MplCanvas(self, width=12, height=8, dpi=100)  # Increase width to 12 for larger subplots
        
        # Create a tab widget
        self.tabs = QTabWidget(self)

        # Create three tab pages
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        # Add tabs to the tab widget
        self.tabs.addTab(self.tab1, "Predicted")
        self.tabs.addTab(self.tab2, "Original")
        self.tabs.addTab(self.tab3, "Match")

        # Set self.predicted_canvas as the central widget for tab1
        self.tab1_layout = QVBoxLayout()
        self.tab1_layout.addWidget(self.predicted_canvas)
        self.tab1.setLayout(self.tab1_layout)
        
        self.tab2_layout = QVBoxLayout()
        self.tab2_layout.addWidget(self.original_canvas)
        self.tab2.setLayout(self.tab2_layout)
        
        self.tab3_layout = QVBoxLayout()
        self.tab3_layout.addWidget(self.match_canvas)
        self.tab3.setLayout(self.tab3_layout)
        
        # Set the central widget to the tab widget
        central_layout.addWidget(self.tabs)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)
        
        
        

        # Initialize DataFrame and closing_prices_b
        self.df = None
        self.closing_prices_b = None



    def load_historic_ohlcv_data(self):

        substring_list_text = self.file_dir_edit.text()
        substring_list = [substring.strip() for substring in substring_list_text.split(',')]

        # Define the directory where your CSV files are located
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

        # Set the full path for the file
        fdir = os.path.join(desktop_path, "ohlcv_data/")
        

        # Create an empty DataFrame to store the combined data
        combined_df = pd.DataFrame(columns = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # Loop through the OHLCV CSV files in the directory
        for filename in os.listdir(fdir):
            if filename.endswith("csv") and any(substring in filename for substring in substring_list):                
                file_path = os.path.join(fdir, filename)
                df = pd.read_csv(file_path)
                df.columns = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume']

                # Scale each column in DataFrame B to the corresponding range of DataFrame A
                if self.interp_data.isChecked():
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        min_A = self.closing_prices_b[col].min()
                        max_A = self.closing_prices_b[col].max()

                        min_B = df[col].min()
                        max_B = df[col].max()

                        df[col] = (df[col] - min_B) / (max_B - min_B) * (max_A - min_A) + min_A

    
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                time.sleep(0.5)



        # Reset the index of the combined DataFrame
        combined_df.reset_index(drop=True, inplace=True)

        combined_df['TimeStamp'] = pd.to_datetime(combined_df['TimeStamp'])
        
        # df = pd.DataFrame(closing_prices, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        combined_df['Color'] = combined_df.apply(lambda row: 1 if row['Close'] > row['Open'] else -1, axis=1)


        # Compute the slope for the 'Close' column
        slope = np.gradient(combined_df['Close'], combined_df.index)

        # Add the slope as a new column in the DataFrame
        combined_df['Slope'] = slope

        # Compute the second slope (second derivative) for the 'Close' column
        second_slope = np.gradient(slope, combined_df.index)

        # Add the second slope as a new column in the DataFrame
        combined_df['Second_Slope'] = second_slope


        # Calculate the average of 'Close' and 'Price' columns and store it in a new column 'Average'
        combined_df['Average_Price'] = combined_df[['Close', 'Open']].mean(axis=1)

        combined_df['Adjusted_Volume'] = combined_df['Volume']*combined_df['Color']


        self.df = combined_df

    def load_test_ohlcv_data(self):
        symbol = self.symbol_combobox.currentText()
        timeframe = self.timeframe_combobox.currentText()
        limit = int(self.limit_combobox.currentText())

        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        self.closing_prices_b = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.closing_prices_b['Color'] = self.closing_prices_b.apply(lambda row: 1 if row['Close'] > row['Open'] else -1, axis=1)
        slope = np.gradient(self.closing_prices_b['Close'], self.closing_prices_b.index)
        self.closing_prices_b['Slope'] = slope
        second_slope = np.gradient(slope, self.closing_prices_b.index)
        self.closing_prices_b['Second_Slope'] = second_slope
        self.closing_prices_b['Average_Price'] = self.closing_prices_b[['Close', 'Open']].mean(axis=1)
        self.closing_prices_b['Adjusted_Volume'] = self.closing_prices_b['Volume'] * self.closing_prices_b['Color']

    def fetch_and_plot_data(self):
        if self.df is None:
            self.load_historic_ohlcv_data()
        limit = int(self.limit_combobox.currentText())
        selected_method = self.method_combobox.currentText()
        selected_plot_variable = self.to_plot_combobox.currentText()
        window_size = int(self.window_combobox.currentText())
        close_a = self.df[selected_method].iloc[:-limit].values
        close_b = self.closing_prices_b[selected_method].iloc[:].values
        close_a = np.convolve(np.array(close_a), np.ones(window_size) / window_size, mode='valid')
        close_b = np.convolve(np.array(close_b), np.ones(window_size) / window_size, mode='valid')

        
        if self.log_compute.isChecked():
            close_a= np.log(close_a)
            close_b = np.log(close_b)
        else:
            pass
            

        if self.fourier_series_radio_button.isChecked():
            fourier_coeffs_a = np.fft.fft(close_a)
            fourier_coeffs_b = np.fft.fft(close_b)
        
            correlation = correlate(fourier_coeffs_a, fourier_coeffs_b, mode='valid')
        else:
            correlation = correlate(close_a, close_b, mode='valid')
        correlation = correlate(close_a, close_b, mode='valid')
        max_correlation_index = np.argmax(correlation)
        start_index = max_correlation_index
        end_index = max_correlation_index + len(close_b)
        similiar_price_y = self.df[selected_plot_variable].iloc[start_index:end_index].values
        similiar_price_x = np.arange(len(similiar_price_y))
        extention = int(self.extension_combobox.currentText())
        predict_price_x = np.arange(len(similiar_price_y) + extention)
        predict_price_y = self.df[selected_plot_variable].iloc[start_index:end_index + extention].values
        original_price_y = self.closing_prices_b[selected_plot_variable].iloc[:].values
        original_price_x = np.arange(len(original_price_y))
        
        if not self.interp_data.isChecked():
            similiar_price_y = np.interp(similiar_price_y, (min(similiar_price_y), max(similiar_price_y)), [min(original_price_y), max(original_price_y)])
            predict_price_y = np.interp(predict_price_y, (min(predict_price_y), max(predict_price_y)), [min(original_price_y), max(original_price_y)])

        r_squared = r2_score(original_price_y[:len(similiar_price_y)], similiar_price_y)
        target = predict_price_y[-1] - similiar_price_y[-1]
        price_target = predict_price_y[-1]
        min_price = min(predict_price_y[extention:])
        max_price = max(predict_price_y[extention:])
        result = f'Future Price Difference: {target:.4f}' + f'\nFuture Price Target: {price_target:.4f}' + f'\nMin Price Target: {min_price:.4f}' + f'\nMax Price Target: {max_price:.4f}' + f"\nR-squared: {r_squared:.4f}"

        self.max_price_label.setText(result)

        for ax in [self.predicted_canvas,self.original_canvas,self.match_canvas]:
            ax.ax.clear()

        for ax in [self.predicted_canvas,self.original_canvas,self.match_canvas]:
            ax.ax.set_ylim(auto=True)
        vertical_line_x = similiar_price_x[-1]

        if self.log_checkbox.isChecked():
            self.predicted_canvas.ax.semilogy(predict_price_x, predict_price_y, label='Predicted', color='navy')
            self.predicted_canvas.ax.axvline(vertical_line_x, color='#e66407', linestyle='--')
            self.original_canvas.ax.semilogy(original_price_x, original_price_y, label='Original', color='#03a30b', linestyle='--')
            self.match_canvas.ax.semilogy(similiar_price_x, similiar_price_y, label='Match', color='#e60707', linestyle='--')
        else:
            self.predicted_canvas.ax.step(predict_price_x, predict_price_y, label='Predicted', color='navy')
            self.predicted_canvas.ax.axvline(vertical_line_x, color='#e66407', linestyle='--')
            self.original_canvas.ax.step(original_price_x, original_price_y, label='Original', color='#03a30b', linestyle='--')
            self.match_canvas.ax.step(similiar_price_x, similiar_price_y, label='Match', color='#e60707', linestyle='--')            



        plt.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
        for ax in [self.predicted_canvas,self.original_canvas,self.match_canvas]:
            ax.ax.set_xlabel('Steps')
            ax.ax.set_ylabel('Y')
            ax.ax.legend()
            ax.ax.grid(True)

        self.predicted_canvas.draw()
        self.original_canvas.draw()
        self.match_canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
