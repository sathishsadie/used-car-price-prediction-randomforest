import streamlit as st
import pickle
import pandas as pd


st.title('Used Car Price Prediction')



datas=pd.read_csv('ss.csv')

datas=datas.loc[:,['Year','Present_Price', 'Kms_Driven',
'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
lis=list()
for i in datas.columns:
    if(datas[i].dtype=='category' or datas[i].dtype=='object'):
        a = st.selectbox(i.title(), options=datas[i].unique())
    else:
        if(len(datas[i].unique())<=6):
            a = st.selectbox(i.title(), options=datas[i].unique())
        else:
            datas[i]=datas[i].astype(float)
            a = st.number_input(i.title(),min_value=datas[i].min(),max_value=datas[i].max(),value=datas[i].median(),step=0.5)
    lis.append(a)


model_path = "models/randomforest.pkl"
labelencoder_path = "models/labelencoder.pkl"
with open(model_path , 'rb') as f:
    rf_reg = pickle.load(f)

with open(labelencoder_path,'rb') as fi:
    fuel_type,seller_type,transmission = pickle.load(fi)



def preprocess(input):
    input['Fuel_Type']=fuel_type.transform(input['Fuel_Type'])
    input['Seller_Type']=seller_type.transform(input['Seller_Type'])
    input['Transmission']=transmission.transform(input['Transmission'])
    return input


if st.button('Predict Price'):
    ss=datas.columns
    dic={}
    for i in range(len(ss)):
        dic[ss[i]]=[lis[i]]
    dff=pd.DataFrame(dic)
    dff=preprocess(dff)
    prediction = rf_reg.predict(dff)

    st.write(f"Prediction of the Enteered Car Price  is {prediction[0]:.2f} Lakhs")