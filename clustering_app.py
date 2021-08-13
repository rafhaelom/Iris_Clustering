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

st.image("images/flores_petal_sepal.png")

st.header("E o Clustering? :dart:")
st.markdown("""*Clustering ou Agrupamento*, é o conjunto de técnicas de **data mining** (mineração de dados no português)
que visa fazer *agrupamentos automáticos* de dados segundo o seu grau de semelhança. O critério de semelhança faz parte 
da *definição do problema* e, dependendo, *do algoritmo*. A cada conjunto de dados resultante do processo da-se o nome 
de **cluster** (grupo, aglomerado ou agrupamento).""")

st.image("images/cluster.png")

st.subheader("Bom, já descobrimos o que é Íris e o que é Clustering, mas e agora? :astonished:")
st.markdown("""Agora vamos entender o problema de negócio que pretendemos solucionar, entender qual e o que é o dado que temos! :grin:""")

st.subheader("""Problema de negócio :nerd_face:""")
st.markdown("""A Clusterização pode ser usada tanto na preparação de dados para descobrir padrões ainda desconhecidos quanto para a criação 
de modelos com o objetivo de criar grupos, separando os dados em grupos ainda desconhecidos. A ideia é a partir dos dados referentes ao tamanho 
da **Pétala** e da **Sépala** de cada flor, ou seja, as variáveis `sepal.length`, `sepal.width`, `petal.length` e `petal.width`, seja possível encontrar e
definir qual é a classe/grupo `variety` de flor íris pertence o dado.""")
st.image("images/petal_sepal.png")   

st.markdown("""Utilizando um modelo de clusterização, ou seja, um modelo de aprendizado de máquina (machine learning) não-supervisionado, para classificar/prever 
as espécies da flor da íris em três espécies neste conjunto de dados. Este modelo de machine learning não utiliza a target, e por isso é chamado de 
não-supervisionado, e sendo as colunas conhecidas por dimensões.""")
st.image("images/k-means-clustering.png")

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

st.markdown("""### Classes que podem ser encontradas:
* Iris Setosa
* Iris Versicolour
* Iris Virginica""")

st.markdown("Verificando a quantidade por tipo de classe, conforme especificado, tem-se 50 instâncias por cada uma das 3 classes nas 4 dimensões do dataset.")
st.write(df["variety"].value_counts())

# Gráfico.
st.markdown("Gráfico com a dispersão dos dados com a separação pela categoria do dataset de origem.")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Sépalas e Pétalas pela classe da planta')

sns.scatterplot(ax=axes[0], x=df["sepal.length"], y=df["sepal.width"], hue=df["variety"])
axes[0].set_title("Sépala")

sns.scatterplot(ax=axes[1], x=df["petal.length"], y=df["petal.width"], hue=df["variety"])
axes[1].set_title("Pétala")
st.pyplot(fig)
plt.clf()

st.subheader("Preparação dos dados :building_construction:")
st.markdown("""Nesta etapa será necessária a separação da coluna `variety` das outras colunas (dimensões) do dataset, afim de 
analisar as variáveis de medida das plantas, sem ser passado ao modelo de ML a `Target` que no caso é a `variety`.""")

X = df.iloc[:, 0:4]

if st.checkbox("Mostrar o dataset preparado"):
    st.dataframe(X) 

st.subheader("Modelo :construction:")
st.markdown("""Utilizaremos o KMeans que não será passado a target para o modelo, e por esse motivo ele é considerado um modelo 
de não-supervisionado. 

A grande questão que o algoritmo KMeans exige é a específicação do número de grupos K. Às vezes, o número de grupos é direcionado pela 
aplicação. Por exemplo, no nosso dataset sabemos que tem-se 3 tipos diferentes de flores íris, portanto o mínimo de clusters ou grupos 
necessário para o modelo é de 3. 
""")

st.image("images/flores_iris.png")

st.markdown("""Mas em alguns momentos não temos essa noção de grupos, e portanto precisamos de uma definição deste, sendo 
então necessária uma abordagem estatística, o mais comum é utilizar de um método para definir o número de clusters para o modelo.""")

st.markdown("""Este método é chamado de **Método do Cotovelo**, basicamente ele é utilizado para identificar quando o conjunto de grupos explica 
a ***maioria*** da variância nos dados. O ***cotovelo*** é o ponto em que a variância explicada cumulativa se estabiliza depois de uma subida brusca, 
por isso o nome do método.""")



st.markdown("""## Referências: 
* https://www.youtube.com/watch?v=EItlUEPCIzM 
* https://medium.com/pursuitnotes/k-means-clustering-model-in-6-steps-with-python-35b532cfa8ad
* https://medium.com/computing-science/using-multilayer-perceptron-in-classification-problems-iris-flower-6fc9fbf36040
* https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset
* http://archive.ics.uci.edu/ml/datasets/Iris
* https://medium.com/greyatom/using-clustering-for-feature-engineering-on-the-iris-dataset-f438366d0b4b
* https://www.kaggle.com/rae385/iris-classification-and-visualization
* https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning""")