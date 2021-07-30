import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Íris Clustering")

st.header("O que é o Íris? :tulip:")
st.markdown("""No artigo de Fisher, o Íris é um clássico dataset na área de reconhecimento de padrões, 
sendo referênciado com frequência até hoje. A base de dados (dataset) contém 3 classes de 50 instâncias 
cada, onde cada classe se refere a um tipo de planta íris, sendo uma espécie(classe) de flor é linearmente 
separável das outras duas, mas as outras duas não são linearmente separáveis ​​uma da outra.""")

st.header("E o Clustering? :dart:")
st.markdown("""*Clustering ou Agrupamento*, é o conjunto de técnicas de **data mining** (mineração de dados no português)
que visa fazer *agrupamentos automáticos* de dados segundo o seu grau de semelhança. O critério de semelhança faz parte 
da *definição do problema* e, dependendo, *do algoritmo*. A cada conjunto de dados resultante do processo da-se o nome 
de **cluster** (grupo, aglomerado ou agrupamento).""")

st.subheader("Bom, já descobrimos o que é Íris e o que é Clustering, mas e agora? :astonished:")
st.markdown("""Agora vamos entender o problema de negócio que pretendemos solucionar, entender qual e o que é o dado que temos! :grin:""")

st.subheader("""Problema de negócio :nerd_face:""")
st.markdown("""A Clusterização pode ser usada tanto na preparação de dados para descobrir padrões ainda desconhecidos quanto para a criação 
de modelos com o objetivo de criar grupos, separando os dados em grupos ainda desconhecidos. 

Utilizando um modelo de clusterização, ou seja, um modelo de aprendizado de máquina (machine learning) não-supervisionado, para classificar/prever 
as espécies da flor da íris em três espécies neste conjunto de dados. Este modelo de machine learning não utiliza a target, e por isso é chamado de 
não-supervisionado, e sendo as colunas conhecidas por dimensões.""")

st.subheader("Dataset :floppy_disk:")
df = pd.read_csv("iris.csv")

if st.checkbox("Mostrar o dataset"):
    st.dataframe(df)    

st.subheader("Dicionário de dados")
st.markdown("""|Coluna |Descrição |
:-------------: | :-------------:
|sepal.length | Comprimento da sépala|
|sepal.width | Largura da sépala|
|petal.length | Comprimento da pétala|
|petal.width | Largura da pétala|
|variety | Classe|""")


st.markdown("""### Classes que podem ser encontradas:""")

st.markdown("Verificando a quantidade por tipo de classe, conforme especificado, tem-se 50 instâncias por cada uma das 3 classes nas 4 dimensões do dataset.")
st.write(df["variety"].value_counts())

st.markdown("Distribuição de cada classe conforme o dataset")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Sépalas e Pétalas')

sns.scatterplot(ax=axes[0], x=df["sepal.length"], y=df["sepal.width"], hue=df["variety"])
axes[0].set_title("Sépala")

sns.scatterplot(ax=axes[1], x=df["petal.length"], y=df["petal.width"], hue=df["variety"])
axes[1].set_title("Pétala")
st.pyplot(fig)
plt.clf()

st.markdown("Testando inclusão de texto!")


st.markdown("Testando inclusão de texto!")
st.markdown("Testando inclusão de texto!")


X = df.iloc[:, 0:4]
