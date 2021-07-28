import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
st.write(df)

st.markdown("Verificando a quantidade por tipo de classe, conforme especificado, tem-se 50 instâncias por cada uma das 3 classes nas 4 dimensões do dataset.")
st.write(df["variety"].value_counts())

col1, col2 = st.beta_columns(2)

with col1:
    sns.scatterplot(data= df, x= "sepal.length", y= "sepal.width", hue= "variety")
    plt.ylabel('sepal.width')
    plt.xlabel('sepal.length')
    plt.show()

with col2:
    sns.scatterplot(data= df, x= "petal.length", y= "petal.width", hue= "variety")
    plt.ylabel('petal.width')
    plt.xlabel('petal.length')
    plt.show()

X = df.iloc[:, 0:4]