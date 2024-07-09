import streamlit as st
import yfinance as yf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("Stock Price Prediction")
st.write("LSTM Model")
st.write("TO DOWNLOAD INDIAN STOCKS WRITE .NS AFTER STOCK SYMBOL")

stock_symbol = st.text_input("Enter the stock symbol:")
n_days = st.number_input("Enter no. of days till which you want to predict:", min_value=1, value=1, step=1)

if st.button("Predict"):
    st.write("Downloading stock data...")
    stock_data = yf.download(stock_symbol, period="max")
    stock_data.reset_index(inplace=True)
    stock_data.to_csv(stock_symbol + ".csv")
    st.write("Data downloaded successfully!")

    st.write("SEQUENCE OF DATA:", len(stock_data))

    close_prices = stock_data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    close_prices_scaled = scaler.fit_transform(close_prices)

    sequence_length = 10
    X = []
    y = []
    for i in range(len(close_prices_scaled) - sequence_length):
        X.append(close_prices_scaled[i:i + sequence_length])
        y.append(close_prices_scaled[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = keras.Sequential([
        keras.layers.LSTM(50, activation="relu", input_shape=(sequence_length, 1)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=45, batch_size=30, verbose=1)

    # Evaluate model
    test_loss = model.evaluate(X_test, y_test)

    last_sequence = X_test[-1].reshape(1, sequence_length, 1)

    predictions = []
    for _ in range(n_days):
        next_day_prediction = model.predict(last_sequence)
        predictions.append(next_day_prediction)
        last_sequence = np.append(last_sequence[:, 1:, :], next_day_prediction.reshape(1, 1, 1), axis=1)

    predicted_prices = np.array(predictions).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    st.write("Predicted Prices for Next", n_days, "Days:")
    st.write(predicted_prices)
    real_close_prices = scaler.inverse_transform(y_test)
    plt.plot(range(len(real_close_prices), len(real_close_prices) + len(predicted_prices)), predicted_prices,color="red", label="Predicted price")
    plt.title("Stock price prediction for " + stock_symbol)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)
    # real_close_prices = scaler.inverse_transform(y_test)
    # plt.plot(real_close_prices, color="blue", label="Close price")
    # plt.plot(range(len(real_close_prices), len(real_close_prices) + len(predicted_prices)), predicted_prices, color="red", label="Predicted price")
    # plt.title("Stock price prediction for " + stock_symbol)
    # plt.xlabel("Days")
    # plt.ylabel("Price")
    # plt.legend()
    # st.pyplot(plt)

    # Calculate and display accuracy
    rmse = np.sqrt(test_loss)
    st.write(rmse)
    # accuracy = 100 - (rmse / np.mean(real_close_prices) * 100)
    # st.write("Accuracy:", round(accuracy, 2), "%")
