from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib import messages
from django.contrib.auth.models import User
import yfinance as yf
import json
from bs4 import BeautifulSoup
import pandas as pd
import requests
from .models import Wishlist  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout , Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from yahooquery import Ticker
from django.contrib.auth.decorators import login_required

def index(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    else:
        return render(request, "login.html")

def logout(request):
    auth_logout(request)
    return redirect('./') 

def login(request):
    try:
        if request.method == "POST":
            username=request.POST['username']
            password=request.POST['password']
            user=authenticate(username=username, password=password)
            if user is not None:
                auth_login(request,user)
                if user.last_name == 'user':
                    print('----------')
                    return redirect('UserDashboard')
                else:
                    return redirect('CenterDashboard')
            else:
                messages.info(request,"invalid login")
                return redirect('index')
    except Exception as ex:
        print(ex)
        return redirect('index')
    
def register(request):
    try:
        if request.method == 'POST':
            username = request.POST.get('username')
            email = request.POST.get('email')
            password1 = request.POST.get('password')
            password2 = request.POST.get('confirm_password')
            
            if  password1 != password2:
                return render(request, 'register.html', {'message': 'password and re entered password mismatched'})
            user = User.objects.create_user(username = username, email = email, password = password1)
            messages.info(request,"Registration Success") 
            return render(request, 'register.html', {'message' : 'Successfully registered'})
        else:
            return render(request, 'register.html')
    except Exception as ex: 
        
        print(ex)
        return render(request, 'register.html', {'message': ex})

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


def create_sequences(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i: (i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def dashboard(request):
    try:
        if not request.user.is_authenticated:
            return redirect("login")

        symbol = request.GET.get('symbol', 'AAPL').upper()
        stock = yf.Ticker(symbol)
        hist = stock.history(period="2y")  # Fetch 2 years of data

        if hist.empty or len(hist) < 60:
            return render(request, "index.html", {"error": "Not enough data for prediction"})

        df = hist[['Close']].copy()
        df_scaled = scaler.fit_transform(df)

        # Train-Test Split (80% Train, 20% Test)
        train_size = int(len(df_scaled) * 0.8)
        train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

        X_train, Y_train = create_sequences(train_data, 60)
        X_test, Y_test = create_sequences(test_data, 60)

        # Reshape inputs for CNN-LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Define the CNN-LSTM Hybrid Model
        model = Sequential([
            # CNN layers to extract features
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(60, 1), padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # LSTM layers for sequence modeling
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers for prediction
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")

        # Train the Model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, Y_train, epochs=2, batch_size=32, verbose=1, 
                 validation_data=(X_test, Y_test), callbacks=[early_stopping])

        # Predict Next Day's Price
        last_60_days = df_scaled[-60:]
        X_input = np.array([last_60_days]).reshape(1, 60, 1)
        predicted_price = model.predict(X_input)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        # Compute additional metrics
        predicted_high = predicted_price * 1.03  # 3% buffer
        predicted_low = predicted_price * 0.97
        moving_avg = df["Close"].rolling(window=50).mean().iloc[-1]

        # Convert float32 to Python float
        context = {
            "symbol": symbol,
            "stock_dates": df.index.strftime("%Y-%m-%d").tolist(),
            "stock_prices": df["Close"].tolist(),
            "current_price": float(df["Close"].iloc[-1]),
            "predicted_price": float(predicted_price),
            "predicted_high": float(predicted_high),
            "predicted_low": float(predicted_low),
            "moving_avg": float(moving_avg),
        }
        return render(request, "index.html", context)
    except Exception as ex:  
        print(ex)
        return render(request, 'login.html', {'message': ex})

def add_to_wishlist(request):
    if request.method == "POST":
        symbol = request.POST.get("symbol")

        if not symbol:
            messages.error(request, "Invalid stock symbol.")
            return redirect("dashboard")  

        # Check if already in wishlist
        if Wishlist.objects.filter(user=request.user, symbol=symbol).exists():
            messages.warning(request, "Stock is already in your wishlist.")
            return redirect("dashboard")

        # Save to database
        Wishlist.objects.create(user=request.user, symbol=symbol)
        messages.success(request, f"{symbol} added to your wishlist!")

    return redirect("dashboard")

def remove_from_wishlist(request, symbol):
    try:
        stock = Wishlist.objects.get(user=request.user, symbol=symbol)
        stock.delete()
        messages.success(request, f"{symbol} removed from your wishlist.")
    except Wishlist.DoesNotExist:
        messages.error(request, "Stock not found in your wishlist.")
    
    return redirect('portfolio')

def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        current_price = stock.history(period="1d")["Close"].iloc[-1]  # Latest closing price
        return round(current_price, 2)  # Round to 2 decimal places
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return "N/A"

def portfolio(request):
    wishlist_stocks = Wishlist.objects.filter(user=request.user)

    # Fetch the current price for each stock in the wishlist
    for stock in wishlist_stocks:
        stock.current_price = get_stock_price(stock.symbol)

    return render(request, 'portfolio.html', {'wishlist_stocks': wishlist_stocks})

def news(request):
    url = f" https://newsdata.io/api/1/news?apikey=pub_76336a70904d8b412ea588a6e9a56607494a2&q=Stock&country=in&language=en,ml&category=business "

    try:
        response = requests.get(url)
        news_data = response.json().get("results", [])
    except Exception as e:
        news_data = []
        print(f"Error fetching news: {e}")

    return render(request, "news.html", {"news_data": news_data})


def market(request):
    url = "https://finance.yahoo.com/markets/stocks/most-active/?start=0&count=100"

# Get the webpage content
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")

# Extract data into a list
    data = []
    for row in table.find_all("tr"):
        cols = [col.text.strip() for col in row.find_all("td")]
        if cols:
            data.append(cols)

# Convert to Pandas DataFrame
    df = pd.DataFrame(data)

    df.columns = ["Symbol",
    "Stock Name",
    "Price",
    "Change",
    "Change %",
    "Volume",
    "Avg Vol (3M)",
    "Market Cap",
    "P/E Ratio",
    "52 wk Change %",
    "a","b"
    ]

    df = df[['Symbol', 'Stock Name', 'Change','Avg Vol (3M)','Market Cap']]

    df.rename(columns={
        'Stock Name':'Name','Avg Vol (3M)': 'Avg_Vol', 'Market Cap': 'Market_Cap'
    }, inplace=True)

    # Convert DataFrame to a list of dictionaries
    stock_data = df.to_dict(orient='records')
    print(df)

    ind_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

    # Fetch stock metadata from yahooquery
    stocks = Ticker(ind_symbols)
    
    indian_stock_data = []
    for symbol in ind_symbols:
        info = stocks.summary_detail.get(symbol, {})
        price = info.get('regularMarketPrice', 'N/A')

        # If price is missing, fetch from yfinance
        if price == 'N/A':
            stock_yf = yf.Ticker(symbol)
            price = stock_yf.history(period="1d")['Close'].iloc[-1] if not stock_yf.history(period="1d").empty else 'N/A'
        
        indian_stock_data.append({
            'Symbol': symbol.replace(".NS", ""),  # Remove ".NS" for display
            'Name': stocks.quote_type.get(symbol, {}).get('shortName', 'N/A'),
            'Price': price,
            'Change': info.get('regularMarketChange', 'N/A'),
            'Change %': info.get('regularMarketChangePercent', 'N/A'),
            'Volume': info.get('regularMarketVolume', 'N/A'),
            'Avg_Vol': info.get('averageDailyVolume3Month', 'N/A'),
            'Market_Cap': info.get('marketCap', 'N/A'),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
        })

    # Convert to DataFrame (optional)
    df = pd.DataFrame(indian_stock_data)
    
    # Debugging: Print output to verify data
    print(df.to_dict(orient='records'))

    return render(request, 'market.html', {'stock_data': stock_data, 'indian_stocks': df.to_dict(orient='records')})


