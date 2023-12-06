import yfinance as yf


def fetch_crypto_data(crypto_name, start_date, end_date):
    ticker = f"{crypto_name}-USD"  # Assuming cryptocurrencies are represented like 'BTC-USD' in Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


import tkinter as tk


def on_submit():
    crypto_name = crypto_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    data = fetch_crypto_data(crypto_name, start_date, end_date)
    # Display data in the GUI or process as needed


root = tk.Tk()

crypto_entry = tk.Entry(root)
crypto_entry.pack()

start_date_entry = tk.Entry(root)
start_date_entry.pack()

end_date_entry = tk.Entry(root)
end_date_entry.pack()

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack()

root.mainloop()
