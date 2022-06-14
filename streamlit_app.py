import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set()


# Display a title
st.title('PENERAPAN CRISP-DM PADA DATA SAHAM PT. TELKOM INDONESIA TBK')

# Create the explanatory variables as DataFrame in pandas
df = pd.read_csv("./TLKM.csv")

# Display dataset when check box is ON
if st.checkbox('Lihat dataset dalam format data tabel'): 
  df["date"] = pd.to_datetime(df["date"])
  df["date"] = df["date"].dt.strftime("%Y-%m-%d")
  cols = ["previous","open_price","first_trade","high","low","close","change","volume","value","frequency","index_individual","offer","offer_volume","bid","bid_volume","listed_shares","tradeble_shares","weight_for_index","foreign_sell","foreign_buy","delisting_date","non_regular_volume","non_regular_value","non_regular_frequency"]
  df[cols] = df[cols].applymap('{:,.1f}'.format)
  st.dataframe(df)


# Show each column description when checkbox is ON.
if st.checkbox('Menampilkan setiap nama kolom dan deskripsinya'):
  st.markdown(
        r"""
        ##### Nama Kolom	dan Keterangan
        ######## date: Tanggal jalannya perdagangan
        ######## previous: Harga penutupan hari bursa sebelumnya
        ######## open_price: Harga pembukaan pada hari tersebut
        ######## first_trade: -
        ######## high: Harga tertinggi pada hari tersebut
        ######## low: Harga terendah pada hari tersebut
        ######## close: Harga penutupan pada hari tersebut
        ######## change: Perubahan harga pada hari tersebut
        ######## volume: Volume perdagangan (dalam satuan lembar)
        ######## value: Total nilai perdagangan pada hari tersebut
        ######## frequency: Frekuensi perdagangan pada hari tersebut
        ######## index_individual: -
        ######## offer: Nilai penawaran harga jual pada hari tersebut
        ######## offer_volume: Volume penawaran harga jual pada hari tersebut
        ######## bid: Nilai penawaran harga beli pada hari tersebut
        ######## bid_volume: Volume penawaran harga beli pada hari tersebut
        ######## listed_shares: Jumlah saham yang beredar di masyarakat
        ######## tradeble_shares: Jumlah saham yang dapat diperjualbelikan oleh masyarakat
        ######## weight_for_index: -
        ######## foreign_sell: Total penjualan oleh asing (dalam satuan lembar)
        ######## foreign_buy: Total pembelian oleh asing (dalam satuan lembar)
        ######## delisting_date: Tanggal penghapusan pencatatan saham di BEI
        ######## non_regular_volume: Volume pada pasar non-reguler
        ######## non_regular_value: Total nilai perdagangan pada pasar non-reguler
        ######## non_regular_frequency: Total frekuensi transaksi pada pasar non-reguler
        ######## 
        """
        )


# Plot the relation between target and explanatory variables
# when the checkbox is ON.
if st.checkbox('Plot hubungan antara variabel target dan variabel penjelas'):
  # Select one explanatory variable for ploting
  checked_variable = st.selectbox(
    'Pilih satu variabel penjelas:',
    df.drop(columns="close").columns
    )
  # Plot
  fig, ax = plt.subplots(figsize=(5, 3))
  ax.scatter(x=df[checked_variable], y=df["close"])
  plt.xlabel(checked_variable)
  plt.ylabel("close")
  st.pyplot(fig)


"""
## Preprocessing
"""

# Select the variables you will NOT use
Features_chosen = []
Features_NonUsed = st.multiselect(
  'Pilih variabel yang TIDAK akan anda gunakan:', 
  df.drop(columns="close").columns
  )

# # Drop the columns you selected
df = df.drop(columns=Features_NonUsed)


# # Choose whether you will perform logarithmic transformation
left_column, right_column = st.columns(2)
bool_log = left_column.radio(
      'Anda akan melakukan transformasi logaritmik?', 
      ('Tidak','Ya')
      )

df_log, Log_Features = df.copy(), []
if bool_log == 'Ya':
  Log_Features = right_column.multiselect(
          'Pilih variabel yang akan Anda lakukan transformasi logaritma',
          df.columns
          )
  # Perform the lagarithmic transformation
  df_log[Log_Features] = np.log(df_log[Log_Features])


# # Choose whether you will perform standardization
left_column, right_column = st.columns(2)
bool_std = left_column.radio(
      'Anda akan melakukan standardisasi?',
      ('Tidak','Ya')
      )

df_std = df_log.copy()
if bool_std == 'Ya':
  Std_Features_NotUsed = right_column.multiselect(
          'Pilih variabel yang TIDAK akan Anda lakukan standarisasi', 
          df_log.drop(columns=["date", "close"]).columns
          )
  # Assign the explanatory variables, 
  # excluded of ones in "Std_Features_NotUsed",
  # to "Std_Features_chosen"
  Std_Features_chosen = []
  for name in df_log.drop(columns=["date", "close"]).columns:
    if name in Std_Features_NotUsed:
      continue
    else:
      Std_Features_chosen.append(name)
  # Perform standardization
  sscaler = preprocessing.StandardScaler()
  sscaler.fit(df_std[Std_Features_chosen])
  df_std[Std_Features_chosen] = sscaler.transform(df_std[Std_Features_chosen])


# """
# ### Split the dataset into training and validation datasets
# """
# left_column, right_column = st.columns(2)
# test_size = left_column.number_input(
#         'Validation data size(rate: 0.0-1.0):',
#         min_value=0.0,
#         max_value=1.0,
#         value=0.2,
#         step=0.1,
#          )
# random_seed = right_column.number_input(
#               'Random seed(Nonnegative integer):',
#                 value=0, 
#                 step=1,
#                 min_value=0)


# # Split the dataset
# X_train, X_val, Y_train, Y_val = train_test_split(
#   df_std.drop(columns=["close"]), 
#   df_std['PRICES'], 
#   test_size=test_size, 
#   random_state=random_seed
#   )


# # Create an instance of liner regression
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)

# Y_pred_train = regressor.predict(X_train)
# Y_pred_val = regressor.predict(X_val)

# # Perform inverse conversion if the logarithmic transformation was performed.
# if "close" in Log_Features:
#   Y_pred_train, Y_pred_val = np.exp(Y_pred_train), np.exp(Y_pred_val)
#   Y_train, Y_val = np.exp(Y_train), np.exp(Y_val)



# """
# ## Show the results
# """

# """
# ### Accuracy of the model
# """
# R2 = r2_score(Y_val, Y_pred_val)
# st.write(f'R2 value: {R2:.2f}')


# """
# ### Plot the results
# """
# left_column, right_column = st.columns(2)
# show_train = left_column.radio(
#         'Plot the result of the training dataset:', 
#         ('Ya','Tidak')
#         )
# show_val = right_column.radio(
#         'Plot the result of the validation dataset:', 
#         ('Ya','Tidak')
#         )


# # Get the maximum value of all objective variable data,
# # including predicted values
# y_max_train = max([max(Y_train), max(Y_pred_train)])
# y_max_val = max([max(Y_val), max(Y_pred_val)])
# y_max = int(max([y_max_train, y_max_val])) 


# # Allows the axis range to be changed dynamically
# left_column, right_column = st.columns(2)
# x_min = left_column.number_input('x_min:',value=0,step=1)
# x_max = right_column.number_input('x_max:',value=y_max,step=1)
# left_column, right_column = st.columns(2)
# y_min = left_column.number_input('y_min:',value=0,step=1)
# y_max = right_column.number_input('y_max:',value=y_max,step=1)


# # Show the results
# fig = plt.figure(figsize=(3, 3))
# if show_train == 'Ya':
#   plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
# if show_val == 'Ya':
#   plt.scatter(Y_val, Y_pred_val,lw=0.1,color="b",label="validation data")
# plt.xlabel("close",fontsize=8)
# plt.ylabel("Prediction of PRICES",fontsize=8)
# plt.xlim(int(x_min), int(x_max)+5)
# plt.ylim(int(y_min), int(y_max)+5)
# plt.legend(fontsize=6)
# plt.tick_params(labelsize=6)

# # Display by Streamlit
# st.pyplot(fig)

