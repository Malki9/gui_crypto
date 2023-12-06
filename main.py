import tkinter as tk
from tkinter import messagebox, scrolledtext, Label, Frame
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # This line was missing
import pandas as pd
import os


import sys

from tensorflow.keras.models import load_model

import joblib
import torch


# Load the saved model and scaler
model = torch.load('H:\crypto GUI\crypto_prediction_model (1).h5')
scaler = joblib.load('H:\crypto GUI\scaler.save')


def predict_high_low(crypto_data):
    # Preprocess crypto_data as needed
    # ...

    # Make predictions using the loaded model
    scaled_data = scaler.transform(crypto_data)
    predicted = model.predict(scaled_data)
    predicted_prices = scaler.inverse_transform(predicted)

    # Assuming the predictions are for future dates
    # You need to correlate these predictions with dates
    predicted_high = np.max(predicted_prices)
    predicted_low = np.min(predicted_prices)

    return predicted_high, predicted_low


# The directory where LSTM_MODEL.py is located
model_dir = 'H:/crypto GUI'

# Check if the directory path is correct and exists
if not os.path.exists(model_dir):
    print(f"The directory {model_dir} does not exist.")
else:
    # Add the directory to sys.path
    sys.path.append(model_dir)

    # Now try importing your LSTM model
    try:
        from LSTM_MODEL import predict_high_low
    except ModuleNotFoundError:
        print("Failed to import LSTM_MODEL. Check if the file LSTM_MODEL.py exists in the directory.")
    except ImportError as e:
        print(f"An error occurred when trying to import from LSTM_MODEL: {e}")


def fetch_crypto_data(crypto_name, start_date, end_date, moving_avg_period=50):
    try:
        ticker = f"{crypto_name}-USD"
        data = yf.download(ticker, start=start_date, end=end_date)

        # Ensure that the date range is large enough for the moving average window
        if data.shape[0] < moving_avg_period:
            messagebox.showerror("Error", f"The date range is too short for a {moving_avg_period}-day moving average.")
            return

        # Calculate the moving average
        data['Moving Average'] = data['Close'].rolling(window=moving_avg_period).mean()

        # Clearing previous plot
        for widget in plot_frame.winfo_children():
            widget.destroy()

        # Plotting the data and moving average
        figure = plt.Figure(figsize=(6, 5), dpi=100)
        ax = figure.add_subplot(111)
        chart = FigureCanvasTkAgg(figure, plot_frame)
        chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        data['Close'].plot(kind='line', legend=True, ax=ax, color='r', marker='o', fontsize=10)
        data['Moving Average'].plot(kind='line', legend=True, ax=ax, color='b', linestyle='--', marker='x', fontsize=10)
        ax.set_title(f'{crypto_name} Price and Moving Average Over Time')

    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch data: {e}")


def fetch_and_calculate_correlation(selected_crypto, start_date, end_date):
    try:
        tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD',
                   'ADA-USD', 'DOT-USD', 'LINK-USD', 'BNB-USD', 'XLM-USD',
                   'DOGE-USD', 'USDT-USD', 'UNI-USD', 'SOL-USD', 'AAVE-USD']
        if selected_crypto not in tickers:
            tickers.append(selected_crypto)

        data = yf.download(tickers, start=start_date, end=end_date)['Close']

        correlation = data.corr()
        top_positive = correlation[selected_crypto].sort_values(ascending=False)[1:11]
        top_negative = correlation[selected_crypto].sort_values(ascending=True)[:10]

        return top_positive, top_negative

    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate correlations for {selected_crypto}: {e}")
        return None, None


def on_submit():
    crypto_name = crypto_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    # Fetch and plot data
    fetch_crypto_data(crypto_name, start_date, end_date)

    # Calculate and display correlations
    selected_crypto = f"{crypto_name}-USD"
    positive_corr, negative_corr = fetch_and_calculate_correlation(selected_crypto, start_date, end_date)
    display_correlations(positive_corr, negative_corr)


def display_correlations(positive_corr, negative_corr):
    # Clear previous data
    for widget in correlation_frame.winfo_children():
        widget.destroy()

    # Create a scrolled text widget for positive correlations
    positive_label = tk.Label(correlation_frame, text="Top 10 Positive Correlations:")
    positive_label.pack()
    positive_scrolled_text = scrolledtext.ScrolledText(correlation_frame, height=10, wrap=tk.WORD)
    positive_scrolled_text.insert(tk.INSERT, "\n".join(f"{coin}: {corr:.2f}" for coin, corr in positive_corr.items()))
    positive_scrolled_text.pack(fill=tk.BOTH, expand=True)
    positive_scrolled_text.configure(state='disabled')  # Disable editing of the text

    # Create a scrolled text widget for negative correlations
    negative_label = tk.Label(correlation_frame, text="Top 10 Negative Correlations:")
    negative_label.pack()
    negative_scrolled_text = scrolledtext.ScrolledText(correlation_frame, height=10, wrap=tk.WORD)
    negative_scrolled_text.insert(tk.INSERT, "\n".join(f"{coin}: {corr:.2f}" for coin, corr in negative_corr.items()))
    negative_scrolled_text.pack(fill=tk.BOTH, expand=True)
    negative_scrolled_text.configure(state='disabled')  # Disable editing of the text


def get_predictions(crypto_name):
    # Call the prediction function from your LSTM model
    predicted_high, predicted_low = predict_high_low(crypto_name)
    return predicted_high, predicted_low


def display_predictions():
    crypto_name = crypto_entry.get()
    predicted_high, predicted_low = get_predictions(crypto_name)

    # Display the predictions in the GUI
    prediction_label.config(text=f"Predicted High: {predicted_high}\nPredicted Low: {predicted_low}")


# Initialize Tkinter window
root = tk.Tk()
root.title("Cryptocurrency Analysis Tool")
root.geometry("1600x900")  # Adjust the size as needed

# Main horizontal frame that will contain all the components
main_frame = Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Left frame for input and predictions
left_frame = Frame(main_frame, borderwidth=2, relief="groove")
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Right frame for correlation data
right_frame = Frame(main_frame, borderwidth=2, relief="groove")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Input frame for user inputs
input_frame = Frame(left_frame)
input_frame.pack(padx=10, pady=10, fill=tk.X)

tk.Label(input_frame, text="Cryptocurrency (e.g., BTC):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
crypto_entry = tk.Entry(input_frame)
crypto_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
start_date_entry = tk.Entry(input_frame)
start_date_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(input_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky='w', padx=5, pady=5)
end_date_entry = tk.Entry(input_frame)
end_date_entry.grid(row=2, column=1, padx=5, pady=5)

submit_button = tk.Button(left_frame, text="Analyze", command=on_submit)
submit_button.pack(pady=10)

# Frame for the plot within the left frame
plot_frame = Frame(left_frame)
plot_frame.pack(fill=tk.BOTH, expand=True)

# Label to display predictions within the left frame
prediction_label = Label(left_frame, text="Predicted High: N/A\nPredicted Low: N/A")
prediction_label.pack()

# Frame for the correlations within the right frame
correlation_frame = Frame(right_frame)
correlation_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()
