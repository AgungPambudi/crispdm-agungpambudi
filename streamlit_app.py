#author          : Agung Pambudi
#email           : mail@agungpambudi.com
#linkedin        : https://linkedin.com/in/agungpambudi
#kaggle          : https://www.kaggle.com/agungpambudi
#version         : 0.5
#
#
#==============================================================================
#                                   _         _ _
# ___ ___ _ _ ___ ___ ___ ___ _____| |_ _ _ _| |_|  ___ ___ _____
#| .'| . | | |   | . | . | .'|     | . | | | . | |_|  _| . |     |
#|__,|_  |___|_|_|_  |  _|__,|_|_|_|___|___|___|_|_|___|___|_|_|_|
#    |___|       |___|_|

import streamlit as st
from datetime import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import math
sns.set_style("darkgrid")


def hitung_regression_roc_auc(y_true, y_pred, num_rounds = 10000):
    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_pairs = 0
    num_same_sign = 0

    for i, j in _yield_pairs(y_true, num_rounds):
        diff_true = y_true[i] - y_true[j]
        diff_score = y_pred[i] - y_pred[j]
        if diff_true * diff_score > 0:
            num_same_sign += 1
        elif diff_score == 0:
            num_same_sign += .5
        num_pairs += 1

    return num_same_sign / num_pairs


def _yield_pairs(y_true, num_rounds):
    
    import numpy as np

    if num_rounds == 'exact':
        for i in range(len(y_true)):
            for j in np.where((y_true != y_true[i]) & (np.arange(len(y_true)) > i))[0]:
                yield i, j     
    else:
        for r in range(num_rounds):
            i = np.random.choice(range(len(y_true)))
            j = np.random.choice(np.where(y_true != y_true[i])[0])
            yield i, j


def hitung_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def hitung_rmse(y_true, y_pred): 
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))


def hitung_r2score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)


def hitung_accuracyscore(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


def hitung_mae(y_true, y_pred): 
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)


def hitung_mse(y_true, y_pred): 
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)


def hitung_crossvalidation(model, predictor, target):
    from sklearn.model_selection import cross_val_score    
    scores = cross_val_score(model, predictor, target, cv = 10, scoring = 'accuracy') 
    return scores


st.title("PENERAPAN CRISP-DM PADA DATA SAHAM PT. TELKOM INDONESIA TBK (STUDI KASUS: BURSA EFEK INDONESIA TAHUN 2015-2022)")

st.markdown("""
	
	## Manfaat Penelitian
	Manfaat dari penelitian ini adalah :

	1.	Mengetahui hasil prediksi nilai saham perusahaan dalam periode 7 tahun dari 2015 hingga 2022 yang akan digunakan oleh investor untuk berinvestasi saham di IDX.
	2.	Bagi peneliti selanjutnya, penulis berharap dapat digunakan sebagai referensi untuk penelitian yang akan datang terkait dengan forecasting atau prediksi harga saham.
	
""")

st.sidebar.title("Operasi pada Dataset")
w1 = st.sidebar.checkbox("Tampilkan dataset", False)
w2 = st.sidebar.checkbox("Menampilkan setiap nama feature dan deskripsinya", False)
plot = st.sidebar.checkbox("Tampilkan plots", False)
plothist = st.sidebar.checkbox("Tampilkan hist plots", False)
linechart = st.sidebar.checkbox("Tampilkan diagram garis", False)
trainmodel = st.sidebar.checkbox("Melatih model", False)
dokfold = st.sidebar.checkbox("KFold", False)

def read_data():
    return pd.read_csv("./TLKM.csv")[["date", "previous", "open_price", "first_trade", "high", "low", "index_individual", "offer", "bid", "close"]]

df = read_data()

if w1:
	df["date"] = pd.to_datetime(df["date"])
	df["date"] = df["date"].dt.strftime("%Y-%m-%d")
	st.dataframe(df)

if w2:
	st.markdown(
        r"""
        ##### Nama Kolom	dan Keterangan
        ######  date: Tanggal jalannya perdagangan
        ######  previous: Harga penutupan hari bursa sebelumnya
        ######  open_price: Harga pembukaan pada hari tersebut
        ######  first_trade: -
        ######  high: Harga tertinggi pada hari tersebut
        ######  low: Harga terendah pada hari tersebut
        ######  close: Harga penutupan pada hari tersebut
        ######  index_individual: -
        ######  offer: Nilai penawaran harga jual pada hari tersebut
        ######  bid: Nilai penawaran harga beli pada hari tersebut
        ######  
        """
        )

if plothist:
    st.subheader("Distribusi setiap feature")
    options = ("previous", "open_price", "first_trade", "high", "low", "index_individual", "offer", "bid", "close")
    sel_cols = st.selectbox("Pilih feature", options, 1)
    st.write(sel_cols)
    fig = go.Histogram(x=df[sel_cols], nbinsx=50)
    st.plotly_chart([fig])
    
if plot:
    st.subheader("Korelasi antara close dan independent variabel")
    options = ("previous", "open_price", "first_trade", "high", "low", "index_individual", "offer", "bid", "close")
    w7 = st.selectbox("Pilih kolom", options, 1)
    st.write(w7)
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.scatter(df[w7], df["close"])
    plt.xlabel(w7)
    plt.ylabel("close")
    # plt.title(fig"{w7} vs Close")
    st.pyplot(fig)

if linechart:
	st.subheader("Diagram garis untuk semua feature")
	cols = ["previous", "open_price", "first_trade", "high", "low", "index_individual", "offer", "bid", "close"]
	df["date"] = pd.to_datetime(df["date"])
	df["date"] = df["date"].dt.strftime("%Y-%m-%d")
	df = df.set_index('date')
	df[cols]
	st.line_chart(df)

if trainmodel:
	st.header("Pemodelan")
	y = df["close"]
	X = df[["previous", "open_price", "first_trade", "high", "low", "index_individual", "offer", "bid"]].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	lrgr = LinearRegression()
	lrgr.fit(X_train, y_train)
	pred = lrgr.predict(X_test)

	rocauc = hitung_regression_roc_auc(y_test, pred)
	r2score = hitung_r2score(y_test, pred)
	mae = hitung_mae(y_test, pred)
	mse = hitung_mse(y_test, pred)
	mape = hitung_mape(y_test, pred)
	rmse = hitung_rmse(y_test, pred)

	rocauc_li = []
	r2score_li = []
	mae_li = []
	mse_li = []
	mape_li = []
	rmse_li = []
	
	rocauc_li.append(rocauc)
	r2score_li.append(r2score)
	mae_li.append(mae)
	mse_li.append(mse)
	mape_li.append(mape)
	rmse_li.append(rmse)
	
	summary = pd.DataFrame(columns=["R2", "ROC-AUC", "MAE", "MSE", "RMSE", "MAPE"])
	summary["R2"] = r2score_li
	summary["ROC-AUC"] = rocauc_li
	summary["MAE"] = mae_li
	summary["MSE"] = mse_li
	summary["RMSE"] = rmse_li
	summary["MAPE"] = mape_li
	
	st.write(summary)
	st.success('Model berhasil dilatih')


if dokfold:
	st.subheader("KFold 10")
	st.empty()
	my_bar = st.progress(0)

	from sklearn.model_selection import KFold

	X = df[["previous", "open_price", "first_trade", "high", "low", "index_individual", "offer", "bid"]].values
	y = df["close"]
		
	kf = KFold(n_splits=10)

	r2_list = []
	rocauc_list = []
	mae_list = []
	mse_list = []
	rmse_list = []
	mape_list = []
	
	idx = 1
	fig = plt.figure()
	i = 0
	for train_index, test_index in kf.split(X):
		my_bar.progress(idx * 10)
		
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		lrgr = LinearRegression()
		lrgr.fit(X_train, y_train)
		pred = lrgr.predict(X_test)
				
		rocauc_ = hitung_regression_roc_auc(y_test, pred)
		r2score_ = hitung_r2score(y_test, pred)
		mae_ = hitung_mae(y_test, pred)
		mse_ = hitung_mse(y_test, pred)
		mape_ = hitung_mape(y_test, pred)
		rmse_ = hitung_rmse(y_test, pred)	
		
		rocauc_list.append(rocauc_)
		r2_list.append(r2score_)
		mae_list.append(mae_)
		mse_list.append(mse_)
		rmse_list.append(rmse_)
		mape_list.append(mape_)

		plt.plot(pred, label = f"dataset-{idx}")
		idx += 1
	plt.legend()
	plt.xlabel("Titik data")
	plt.ylabel("Prediksi")
	plt.show()
	st.plotly_chart(fig)

	res = pd.DataFrame(columns=["R2", "ROC-AUC", "MAE", "MSE", "RMSE", "MAPE"])

	res["R2"] = r2_list
	res["ROC-AUC"] = rocauc_list
	res["MAE"] = mae_list
	res["MSE"] = mse_list
	res["RMSE"] = rmse_list
	res["MAPE"] = mape_list

	st.write(res)
	st.balloons()
