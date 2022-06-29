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


st.title("PENERAPAN CRISP-DM PADA DATA SAHAM PT. TELKOM INDONESIA TBK")

st.markdown("""
	The data set contains information about money spent on advertisement and their generated sales. Money
	was spent on TV, radio and newspaper ads.

	## Problem Statement
	Close (in thousands of units) for a particular product as a function of advertising budgets (in thousands of
	dollars) for TV, radio, and newspaper media. Suppose that in our role as statistical consultants we are
	asked to suggest.

	Here are a few important questions that you might seek to address:
	- Is there a relationship between advertising budget and sales?
	- How strong is the relationship between the advertising budget and sales?
	- Which media contribute to sales?
	- How accurately can we estimate the effect of each medium on sales?
	- How accurately can we predict future sales?
	- Is the relationship linear?

	We want to find a function that given input budgets for TV, radio and newspaper predicts the output sales
	and visualize the relationship between the features and the response using scatter plots.

	The objective is to use linear regression to understand how advertisement spending impacts sales.
	
""")

st.sidebar.title("Operasi pada Dataset")
w1 = st.sidebar.checkbox("Tampilkan dataset", False)
w2 = st.sidebar.checkbox("Menampilkan setiap nama feature dan deskripsinya", False)
plot = st.sidebar.checkbox("Tampilkan plots", False)
plothist = st.sidebar.checkbox("Tampilkan hist plots", False)
linechart = st.sidebar.checkbox("Tampilkan diagram garis", False)
# distView = st.sidebar.checkbox("Tampilan dist view", False)
# _3dplot = st.sidebar.checkbox("3D plots", False)
trainmodel = st.sidebar.checkbox("Melatih model", False)
dokfold = st.sidebar.checkbox("KFold", False)

# @st.cache
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
	st.subheader("Diagram garis")
	cols = ["previous", "open_price", "first_trade", "high", "low", "index_individual", "offer", "bid", "close"]
	df["date"] = pd.to_datetime(df["date"])
	df["date"] = df["date"].dt.strftime("%Y-%m-%d")
	df = df.set_index('date')
	df[cols]
	st.line_chart(df)

# if distView:
# 	st.subheader("Menampilkan distribusi gabungan")
# 	# Add histogram data

# 	# Group data together
# 	hist_data = [df["previous"].values, df["open_price"].values, df["first_trade"].values, df["high"].values, df["low"].values, df["index_individual"].values, df["offer"].values, df["bid"].values]

# 	group_labels = ["previous", "open_price", "first_trade", "high", "low", "index_individual", "offer", "bid"]

# 	# Create distplot with custom bin_size
# 	fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

# 	# Plot!
# 	st.plotly_chart(fig)

# if _3dplot:
# 	options = st.multiselect(
#      'Enter columns to plot',('previous', 'open_price', 'first_trade', 'high', 'low', 'index_individual', 'offer', 'bid'),('previous', 'open_price', 'first_trade', 'high', 'low', 'index_individual', 'offer', 'bid', 'close'))
# 	st.write('You selected:', options)
# 	st.subheader("previous & open_price vs Close")
# 	hist_data = [df["previous"].values, df["open_price"].values, df["first_trade"].values, df["high"].values, df["low"].values, df["index_individual"].values, df["offer"].values, df["bid"].values]

# 	#x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()
# 	trace1 = go.Scatter3d(
# 		x = hist_data[0],
# 		y = hist_data[1],
# 		z = df["close"].values,
# 		mode = "markers",
# 		marker = dict(
# 			size = 8,
# 			#color=df['sales'],  # set color to an array/list of desired values
# 			colorscale = "Viridis",  # choose a colorscale
# 	#        opacity=0.,
# 		),
# 	)

# 	data = [trace1]
# 	layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
# 	fig = go.Figure(data=data, layout=layout)
# 	st.write(fig)


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
