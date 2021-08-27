import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics

st.title("Íris Clustering")

st.header("O que é o Íris? :tulip:")
st.markdown("<p align='justify'> No artigo de Fisher, o Íris é um clássico dataset na área de reconhecimento de padrões, sendo referênciado com frequência até hoje. A base de dados (dataset) contém 3 classes de 50 instâncias cada, onde cada classe se refere a um tipo de planta íris, sendo uma espécie(classe) de flor é linearmente separável das outras duas, mas as outras duas não são linearmente separáveis ​​uma da outra. <p align='justify'>", unsafe_allow_html=True)

st.image("images/flores_petal_sepal.png")

st.header("E o Clustering? :dart:")
st.markdown("""<p align='justify'> <b>Clustering ou Agrupamento</b>, é o conjunto de técnicas de <b>data mining</b> (mineração de dados no português)
que visa fazer <i>agrupamentos automáticos</i> de dados segundo o seu grau de semelhança. O critério de semelhança faz parte 
da <i>definição do problema</i> e, dependendo, <i>do algoritmo</i>. A cada conjunto de dados resultante do processo da-se o nome 
de <b>cluster</b> (grupo, aglomerado ou agrupamento). <p align='justify'>""", unsafe_allow_html=True)

st.image("images/cluster.png")

st.subheader("Bom, já descobrimos o que é Íris e o que é Clustering, mas e agora? :astonished:")
st.markdown("""<p align='justify'>Agora vamos entender o problema de negócio que pretendemos solucionar, entender qual e o que é o dado que temos! &#x1F601 <p align='justify'>""", unsafe_allow_html=True)

st.subheader("""Problema de negócio :nerd_face:""")
st.markdown("""<p align='justify'>A Clusterização pode ser usada tanto na preparação de dados para descobrir padrões ainda desconhecidos quanto para a criação 
de modelos com o objetivo de criar grupos, separando os dados em grupos ainda desconhecidos. A ideia é a partir dos dados referentes ao tamanho 
da <b>Pétala</b> e da <b>Sépala</b> de cada flor, ou seja, as variáveis <code>sepal.length</code>, <code>sepal.width</code>, <code>petal.length</code> e <code>petal.width</code>, seja possível encontrar e
definir qual é a classe/grupo <code>variety</code> de flor íris pertence o dado. <p align='justify'>""", unsafe_allow_html=True)

st.image("images/petal_sepal.png")

st.markdown("""<p align='justify'>Utilizando um modelo de clusterização, ou seja, um modelo de aprendizado de máquina (machine learning) não-supervisionado, para classificar/prever 
as espécies da flor da íris em três espécies neste conjunto de dados. Este modelo de machine learning não utiliza a target, e por isso é chamado de 
não-supervisionado, e sendo as colunas conhecidas por dimensões. <p align='justify'>""", unsafe_allow_html=True)
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

st.markdown("""<p align='justify'>Verificando a quantidade por tipo de classe, conforme especificado, tem-se 50 instâncias por cada uma das 3 classes nas 4 dimensões do dataset. <p align='justify'>""", unsafe_allow_html=True)
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
st.markdown("""<p align='justify'>Nesta etapa será necessária a separação da coluna <code>variety</code> das outras colunas (dimensões) do dataset, afim de 
analisar as variáveis de medida das plantas, sem ser passado ao modelo de ML a <code>Target</code> que no caso é a <code>variety</code>. <p align='justify'>""", unsafe_allow_html=True)

X = df.iloc[:, 0:4]

if st.checkbox("Mostrar o dataset preparado"):
    st.dataframe(X) 

st.subheader("Modelo :construction:")
st.markdown("""<p align='justify'> Utilizaremos o KMeans que não será passado a target para o modelo, e por esse motivo ele é considerado um modelo 
de não-supervisionado. 

A grande questão que o algoritmo KMeans exige é a específicação do número de grupos K. Às vezes, o número de grupos é direcionado pela 
aplicação. Por exemplo, no nosso dataset sabemos que tem-se 3 tipos diferentes de flores íris, portanto o mínimo de clusters ou grupos 
necessário para o modelo é de 3. 
<p align='justify'>""", unsafe_allow_html=True)

st.image("images/flores_iris.png")

st.subheader("Método do Cotovelo")
st.markdown("""<p align='justify'>Mas em alguns momentos não temos essa noção de grupos, e portanto precisamos de uma definição deste, sendo 
então necessária uma abordagem estatística, o mais comum é utilizar de um método para definir o número de clusters para o modelo. <p align='justify'>""", unsafe_allow_html=True)

st.markdown("""<p align='justify'>Este método é chamado de <b>Método do Cotovelo</b>, basicamente ele é utilizado para identificar quando o conjunto de grupos explica 
a <i>maioria</i> da variância nos dados. O <i>cotovelo</i> é o ponto em que a variância explicada cumulativa se estabiliza depois de uma subida brusca, 
por isso o nome do método. <p align='justify'>""", unsafe_allow_html=True)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

sns.lineplot(range(1, 11), wcss)
plt.title("Curva do Cotovelo")
plt.xlabel("Número de Clusters")
plt.ylabel("Somatório da variância explicada cumulativa")
st.pyplot(fig)
plt.clf()

st.markdown("""<p align='justify'>Percebe-se que no gráfico da <b>Curva do Cotovelo</b> o melhor valor para o número de cluters 
no modelo KMeans é de 3 clusters. <p align='justify'>""", unsafe_allow_html=True)

st.subheader("Escolha o número de clusters que deseja utilizar para o modelo KMeans :rocket:")
num_clus = st.slider('Qual o número de clusters irá utilizar?', 1, 10, 3)

# Algoritmo KMeans.
kmeans = KMeans(n_clusters = num_clus, max_iter = 300, n_init = 10, random_state = 0)
kmeans.fit(X)
clusters = kmeans.fit_predict(X)

# Coordenadas de cada centroid pela dimensão.
centroids = kmeans.cluster_centers_

# Número de clusters.
st.write("Número de Clusters: ", kmeans.n_clusters)

# Labels preditas.
labels = kmeans.labels_

# Faz a clusterização dos dados usando o modelo criado
grupos = kmeans.predict(X)

if num_clus == 1:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()
elif num_clus == 2:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    plt.scatter(X.iloc[clusters == 1,0], X.iloc[clusters == 1,1], s=50, color='green')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    plt.scatter(centroids[1][0], centroids[1][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()
elif num_clus == 3:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    plt.scatter(X.iloc[clusters == 1,0], X.iloc[clusters == 1,1], s=50, color='green')
    plt.scatter(X.iloc[clusters == 2,0], X.iloc[clusters == 2,1], s=50, color='yellow')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    plt.scatter(centroids[1][0], centroids[1][1], marker="*", s=200, color='black')
    plt.scatter(centroids[2][0], centroids[2][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()
elif num_clus == 4:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    plt.scatter(X.iloc[clusters == 1,0], X.iloc[clusters == 1,1], s=50, color='green')
    plt.scatter(X.iloc[clusters == 2,0], X.iloc[clusters == 2,1], s=50, color='yellow')
    plt.scatter(X.iloc[clusters == 3,0], X.iloc[clusters == 3,1], s=50, color='blue')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    plt.scatter(centroids[1][0], centroids[1][1], marker="*", s=200, color='black')
    plt.scatter(centroids[2][0], centroids[2][1], marker="*", s=200, color='black')
    plt.scatter(centroids[3][0], centroids[3][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()
elif num_clus == 5:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    plt.scatter(X.iloc[clusters == 1,0], X.iloc[clusters == 1,1], s=50, color='green')
    plt.scatter(X.iloc[clusters == 2,0], X.iloc[clusters == 2,1], s=50, color='yellow')
    plt.scatter(X.iloc[clusters == 3,0], X.iloc[clusters == 3,1], s=50, color='blue')
    plt.scatter(X.iloc[clusters == 4,0], X.iloc[clusters == 4,1], s=50, color='orange')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    plt.scatter(centroids[1][0], centroids[1][1], marker="*", s=200, color='black')
    plt.scatter(centroids[2][0], centroids[2][1], marker="*", s=200, color='black')
    plt.scatter(centroids[3][0], centroids[3][1], marker="*", s=200, color='black')
    plt.scatter(centroids[4][0], centroids[4][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()
elif num_clus == 6:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    plt.scatter(X.iloc[clusters == 1,0], X.iloc[clusters == 1,1], s=50, color='green')
    plt.scatter(X.iloc[clusters == 2,0], X.iloc[clusters == 2,1], s=50, color='yellow')
    plt.scatter(X.iloc[clusters == 3,0], X.iloc[clusters == 3,1], s=50, color='blue')
    plt.scatter(X.iloc[clusters == 4,0], X.iloc[clusters == 4,1], s=50, color='orange')
    plt.scatter(X.iloc[clusters == 5,0], X.iloc[clusters == 5,1], s=50, color='violet')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    plt.scatter(centroids[1][0], centroids[1][1], marker="*", s=200, color='black')
    plt.scatter(centroids[2][0], centroids[2][1], marker="*", s=200, color='black')
    plt.scatter(centroids[3][0], centroids[3][1], marker="*", s=200, color='black')
    plt.scatter(centroids[4][0], centroids[4][1], marker="*", s=200, color='black')
    plt.scatter(centroids[5][0], centroids[5][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()
elif num_clus == 7:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    plt.scatter(X.iloc[clusters == 1,0], X.iloc[clusters == 1,1], s=50, color='green')
    plt.scatter(X.iloc[clusters == 2,0], X.iloc[clusters == 2,1], s=50, color='yellow')
    plt.scatter(X.iloc[clusters == 3,0], X.iloc[clusters == 3,1], s=50, color='blue')
    plt.scatter(X.iloc[clusters == 4,0], X.iloc[clusters == 4,1], s=50, color='orange')
    plt.scatter(X.iloc[clusters == 5,0], X.iloc[clusters == 5,1], s=50, color='violet')
    plt.scatter(X.iloc[clusters == 6,0], X.iloc[clusters == 6,1], s=50, color='cyan')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    plt.scatter(centroids[1][0], centroids[1][1], marker="*", s=200, color='black')
    plt.scatter(centroids[2][0], centroids[2][1], marker="*", s=200, color='black')
    plt.scatter(centroids[3][0], centroids[3][1], marker="*", s=200, color='black')
    plt.scatter(centroids[4][0], centroids[4][1], marker="*", s=200, color='black')
    plt.scatter(centroids[5][0], centroids[5][1], marker="*", s=200, color='black')
    plt.scatter(centroids[6][0], centroids[6][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()
elif num_clus == 8:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    plt.scatter(X.iloc[clusters == 1,0], X.iloc[clusters == 1,1], s=50, color='green')
    plt.scatter(X.iloc[clusters == 2,0], X.iloc[clusters == 2,1], s=50, color='yellow')
    plt.scatter(X.iloc[clusters == 3,0], X.iloc[clusters == 3,1], s=50, color='blue')
    plt.scatter(X.iloc[clusters == 4,0], X.iloc[clusters == 4,1], s=50, color='orange')
    plt.scatter(X.iloc[clusters == 5,0], X.iloc[clusters == 5,1], s=50, color='violet')
    plt.scatter(X.iloc[clusters == 6,0], X.iloc[clusters == 6,1], s=50, color='cyan')
    plt.scatter(X.iloc[clusters == 7,0], X.iloc[clusters == 7,1], s=50, color='deeppink')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    plt.scatter(centroids[1][0], centroids[1][1], marker="*", s=200, color='black')
    plt.scatter(centroids[2][0], centroids[2][1], marker="*", s=200, color='black')
    plt.scatter(centroids[3][0], centroids[3][1], marker="*", s=200, color='black')
    plt.scatter(centroids[4][0], centroids[4][1], marker="*", s=200, color='black')
    plt.scatter(centroids[5][0], centroids[5][1], marker="*", s=200, color='black')
    plt.scatter(centroids[6][0], centroids[6][1], marker="*", s=200, color='black')
    plt.scatter(centroids[7][0], centroids[7][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()
elif num_clus == 9:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    plt.scatter(X.iloc[clusters == 1,0], X.iloc[clusters == 1,1], s=50, color='green')
    plt.scatter(X.iloc[clusters == 2,0], X.iloc[clusters == 2,1], s=50, color='yellow')
    plt.scatter(X.iloc[clusters == 3,0], X.iloc[clusters == 3,1], s=50, color='blue')
    plt.scatter(X.iloc[clusters == 4,0], X.iloc[clusters == 4,1], s=50, color='orange')
    plt.scatter(X.iloc[clusters == 5,0], X.iloc[clusters == 5,1], s=50, color='violet')
    plt.scatter(X.iloc[clusters == 6,0], X.iloc[clusters == 6,1], s=50, color='cyan')
    plt.scatter(X.iloc[clusters == 7,0], X.iloc[clusters == 7,1], s=50, color='deeppink')
    plt.scatter(X.iloc[clusters == 8,0], X.iloc[clusters == 8,1], s=50, color='beige')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    plt.scatter(centroids[1][0], centroids[1][1], marker="*", s=200, color='black')
    plt.scatter(centroids[2][0], centroids[2][1], marker="*", s=200, color='black')
    plt.scatter(centroids[3][0], centroids[3][1], marker="*", s=200, color='black')
    plt.scatter(centroids[4][0], centroids[4][1], marker="*", s=200, color='black')
    plt.scatter(centroids[5][0], centroids[5][1], marker="*", s=200, color='black')
    plt.scatter(centroids[6][0], centroids[6][1], marker="*", s=200, color='black')
    plt.scatter(centroids[7][0], centroids[7][1], marker="*", s=200, color='black')
    plt.scatter(centroids[8][0], centroids[8][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()
elif num_clus == 10:
    # Gráfico com os Clusters.
    plt.scatter(X.iloc[clusters == 0,0], X.iloc[clusters == 0,1], s=50, color='red')
    plt.scatter(X.iloc[clusters == 1,0], X.iloc[clusters == 1,1], s=50, color='green')
    plt.scatter(X.iloc[clusters == 2,0], X.iloc[clusters == 2,1], s=50, color='yellow')
    plt.scatter(X.iloc[clusters == 3,0], X.iloc[clusters == 3,1], s=50, color='blue')
    plt.scatter(X.iloc[clusters == 4,0], X.iloc[clusters == 4,1], s=50, color='orange')
    plt.scatter(X.iloc[clusters == 5,0], X.iloc[clusters == 5,1], s=50, color='violet')
    plt.scatter(X.iloc[clusters == 6,0], X.iloc[clusters == 6,1], s=50, color='cyan')
    plt.scatter(X.iloc[clusters == 7,0], X.iloc[clusters == 7,1], s=50, color='deeppink')
    plt.scatter(X.iloc[clusters == 8,0], X.iloc[clusters == 8,1], s=50, color='beige')
    plt.scatter(X.iloc[clusters == 9,0], X.iloc[clusters == 9,1], s=50, color='chocolate')
    # Centroids
    plt.scatter(centroids[0][0], centroids[0][1], marker="*", s=200, color='black')
    plt.scatter(centroids[1][0], centroids[1][1], marker="*", s=200, color='black')
    plt.scatter(centroids[2][0], centroids[2][1], marker="*", s=200, color='black')
    plt.scatter(centroids[3][0], centroids[3][1], marker="*", s=200, color='black')
    plt.scatter(centroids[4][0], centroids[4][1], marker="*", s=200, color='black')
    plt.scatter(centroids[5][0], centroids[5][1], marker="*", s=200, color='black')
    plt.scatter(centroids[6][0], centroids[6][1], marker="*", s=200, color='black')
    plt.scatter(centroids[7][0], centroids[7][1], marker="*", s=200, color='black')
    plt.scatter(centroids[8][0], centroids[8][1], marker="*", s=200, color='black')
    plt.scatter(centroids[9][0], centroids[9][1], marker="*", s=200, color='black')
    # Título e visualização
    plt.title("Clusters por flor e seus centroids")
    st.pyplot(fig)
    plt.clf()


st.subheader("""Avaliando o modelo""")
st.markdown("""Utilizaremos a tabela cruzada para verificar a qualidade das predições feitas pelo modelo.""")

df1 = pd.DataFrame({'labels':labels, "species":df['variety']})
ct = pd.crosstab(df1['labels'], df1['species'])

st.write(ct)

st.markdown("""Para melhorar a visualização da predição do modelo, será feito o gráfico desta tabela cruzada.""")

plt.title("KMeans")
sns.heatmap(ct,annot=True,cbar=False,cmap="Blues") #annot mostra os coeficientes da matriz
st.pyplot(fig)
plt.clf()

st.markdown("""<p align='justify'>Pode-se perceber que seguindo o princípio do método do cotovelo com 3 clusters, o modelo acertou todas as <code>50</code> flores do tipo <b>Setosa</b>, 
mas está errando <code>14</code> flores do tipo <b>Víginica</b> e <code>2</code> do tipo <b>Versicolor</b>. Mas já podemos ver o funcionamento do mesmo. <p align='justify'>""", unsafe_allow_html=True)


results = df[["variety"]].copy()
results["clusters"] = clusters

teste = results.groupby(['clusters']).agg(lambda x:x.value_counts().index[0])
teste.reset_index(inplace=True)

colunas = {'variety': 'flor'}
teste.rename(columns=colunas, inplace=True)

dados = results.merge(teste, how="left", on="clusters")

st.write("Precision_score micro: ", metrics.precision_score(dados["variety"], dados["flor"], average="micro"))

st.write("Precision_score macro: ", metrics.precision_score(dados["variety"], dados["flor"], average="macro"))

st.markdown("""## Referências: 
* https://www.youtube.com/watch?v=EItlUEPCIzM 
* https://medium.com/pursuitnotes/k-means-clustering-model-in-6-steps-with-python-35b532cfa8ad
* https://medium.com/computing-science/using-multilayer-perceptron-in-classification-problems-iris-flower-6fc9fbf36040
* https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset
* http://archive.ics.uci.edu/ml/datasets/Iris
* https://medium.com/greyatom/using-clustering-for-feature-engineering-on-the-iris-dataset-f438366d0b4b
* https://www.kaggle.com/rae385/iris-classification-and-visualization
* https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning
* https://matplotlib.org/stable/gallery/color/named_colors.html
* https://www.w3schools.com/charsets/ref_emoji_smileys.asp
* https://www.tc.df.gov.br/ice4/vordf/outros/html-comandos.html
* https://developer.mozilla.org/pt-BR/docs/Web/HTML/Element/code
* https://www.ti-enxame.com/pt/r/centrar-imagem-e-texto-em-r-markdown-para-um-relatorio-pdf/1046822178/
* https://docs.streamlit.io/en/stable/api.html
* https://rknagao.medium.com/streamlit-101-o-b%C3%A1sico-para-colocar-seu-projeto-no-ar-38a71bd641eb
* https://streamlit.io/""")

st.subheader("Web App feito por Rafhael de Oliveira Martins")
st.markdown("Data 25/08/2021")
"[Repositório](https://github.com/rafhaelom/Iris_Clustering)"
"[![GitHub](https://img.shields.io/badge/-GitHub-333333?style=for-the-badge&logo=github)](https://github.com/rafhaelom)" " " "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafhael-de-oliveira-martins-3bab63138)"