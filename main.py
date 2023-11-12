import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk
import re
from wordcloud import WordCloud


nltk.download('punkt')

# Instalar as bibliotecas necessárias (descomente as linhas abaixo se ainda não tiver instalado)
# !pip install pandas matplotlib nltk

# Carregar CSV para DataFrame
df = pd.read_csv('ChangesPython.csv')

# Exibir as primeiras linhas
print(df.head())

# Estatísticas descritivas
print("\nEstatísticas Descritivas:")
print(df.describe())

# Gráfico de barras para a quantidade de alterações
#df['Quantidade de Alteracoes'].plot(kind='bar')
#plt.title('Distribuição da Quantidade de Alterações')
#plt.xlabel('Repositório')
#plt.ylabel('Quantidade de Alterações')
#plt.show()

# Tokenização dos nomes dos commits e cálculo da frequência de palavras
tokens = word_tokenize(' '.join(df['Nome dos Commits']))
# Filtrar palavras com caracteres não alfabéticos
filtered_tokens = [word for word in tokens if re.match('^[a-zA-Z]+$', word)and word.lower() not in ['the', 'to', 'for', 'and', 'in', 'of', 'a','is','initial','commit']]

fdist = FreqDist(filtered_tokens)

# Exibir as palavras mais frequentes
print("\nPalavras mais frequentes nos Nomes dos Commits (sem caracteres especiais):")
print(fdist.most_common(10))

# Adicione uma coluna para palavras-chave de inconsistência de licença
df['Inconsistency Keywords'] = df['Nome dos Commits'].str.contains('inconsistência|problema de licença|licença inválida|Inconsistency|incon|incompatibility', case=False)

# Análise da frequência de inconsistências de licença por repositório
inconsistency_frequency = df.groupby('Repositorio')['Inconsistency Keywords'].sum()
print("\nFrequência de Inconsistências de Licença por Repositório:")
print(inconsistency_frequency)
# Filtrar comentários relacionados a inconsistências de licença
inconsistency_comments = df[df['Inconsistency Keywords']]['Nome dos Commits']

# Exibir os comentários relacionados a inconsistências de licença
print("\nComentários sobre Inconsistências de Licença:")
print(inconsistency_comments)



# Criar uma nuvem de palavras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fdist)

# Exibir a nuvem de palavras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


