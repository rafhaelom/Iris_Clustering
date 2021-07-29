import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Íris Clustering")

st.header("Sobre a base de dados")
st.markdown("""No artigo de Fisher, um clássico na área de reconhecimento de padrões, sendo referênciado 
com frequência até hoje. A base contém 3 classes de 50 instâncias cada, onde cada classe se refere a um 
tipo de planta íris. Uma classe é linearmente separável das outras 2, os últimos não são linearmente 
separáveis uns dos outros.""")

df = pd.read_csv("iris.csv")

st.subheader("Dicionário de dados")
st.markdown("""|Coluna |Descrição |
:-------------: | :-------------:
|sepal.length | Comprimento da sépala|
|sepal.width | Largura da sépala|
|petal.length | Comprimento da pétala|
|petal.width | Largura da pétala|
|variety | Classe|""")


st.markdown("""### Classes que podem ser encontradas:
* Iris Setosa
* Iris Versicolour
* Iris Virginica""")

st.subheader("Dados")
st.dataframe(df)

st.markdown("Verificando a quantidade por tipo de classe, conforme especificado, tem-se 50 instâncias por cada uma das 3 classes nas 4 dimensões do dataset.")
st.write(df["variety"].value_counts())

st.markdown("Plotando gráfico")


#sns.scatterplot(data=df, x="sepal.length", y="sepal.width", hue="variety")
#plt.show()

#st.pyplot(plt)

#############################

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Sépalas e Pétalas')

sns.scatterplot(ax=axes[0], x=df["sepal.length"], y=df["sepal.width"], hue=df["variety"])
#axes[0].set_title(bulbasaur.name)

sns.scatterplot(ax=axes[1], x=df["petal.length"], y=df["petal.width"], hue=df["variety"])
#axes[1].set_title(charmander.name)
st.pyplot(fig)
plt.clf()

X = df.iloc[:, 0:4]