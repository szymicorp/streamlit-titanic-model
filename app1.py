# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
import pandas as pd
import math
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
sex_d = {0:"Female", 1:"Male"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

base_data = pd.read_csv("DSP_1.csv")
age_max = max(base_data["Age"])
age_min = min(base_data["Age"])
sib_max = max(base_data["SibSp"])
sib_min = min(base_data["SibSp"])
parch_max = max(base_data["Parch"])
parch_min = min(base_data["Parch"])
fare_max = max(base_data["Fare"])
fare_min = min(base_data["Fare"])

def main():

	st.set_page_config(page_title="Titanic")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://ichef.bbci.co.uk/news/976/cpsprodpb/8884/production/_98684943__mg_9727.jpg")

	with overview:
		st.title("Titanic was lit, check if you would survived!")

	with left:
		sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x])
		embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x])
		pclass_radio = st.radio("Klasa", list(pclass_d), format_func= lambda x : pclass_d[x])

	with right:
		age_slider = st.slider("Wiek", value=1, min_value=math.ceil(age_min), max_value=math.ceil(age_max))
		sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=sib_min, max_value=sib_max)
		parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=parch_min, max_value=parch_max)
		fare_slider = st.slider("Cena biletu", min_value=math.ceil(fare_min), max_value=math.ceil(fare_max), step=1)

	data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
