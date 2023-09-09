# Scrap base de Renda Fixa

import requests
import pandas as pd
import json


url = "https://api2.apprendafixa.com.br/vn/get_featured_investments"

payload = {
    "idx": [],
    "corretora": [],
    "emissor": []
}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
   # "g-rcp-tkn": "03AAYGu2RQfMpKPdaecIBoZUDJSnWHnMk-N9XdxfGS0vmBzhEcQDuqL2irjnoSA5Nd-jY6ugkWz63XbcN9oBUFNDkCokuOUDsJv3lYsnLdBsxTaXm0kz-MHo3Ge3eOzCGdxgZOoAaUJplQjRjKJOjpoiJMze9elW0H9YaIUdsW7NIVWq3Ur6lFEwibishqpRY36K7b_feBjU5EYlj-MyKKq9ULYAWeSJoQIsGvgt10Lhfm3apj0n9knI4ohxAItSv2qRi4QcIGt4v0Fe6uCCy_TxgEqOJQIXlBoKOhpZm1pCupkNoH2gyA7kc_s4n_XobB8JTOCSkKZhp1RYVLXx7bz2KbFse7HmXlQsQdXrTO_Z6xxQX6wQMabqOwasCVH2kgNJmtkQQmcA7yYsr37hJfJdAFWjWrXdSJGFliyhCisMt4d1U1mPK69t6AJyw4q1BPIM05dVQqqTnm8He59d3ySzjVzP0U6nWKbQ8z5yKPuZS9yAfhIiAotiaihqgOmOiiZ0l6uGDoeNEvp7zUYfc8mbHjZLgCfPRe1RPSuuLy1b7D_8Aad4Mi9BE",
    "Content-Type": "application/json",
    "Origin": "https://apprendafixa.com.br",
    "Connection": "keep-alive",
    "Referer": "https://apprendafixa.com.br/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "TE": "trailers"
}

response = requests.request("POST", url, json=payload, headers=headers)
data = response.json()

# Convert the list of dictionaries into a DataFrame
df = pd.json_normalize(data)

#Exploração do Dataset
#df.info()
#Verificando que a base precida de tratamento
#df.agencia.value_counts()
# function limpeza base

def clean_agencia(agencia):
    # Check if rating is a string
    if isinstance(agencia, str):
        # Remove '.br' suffix if present
        #if rating.endswith('.br'):
        #    rating = rating[:-3]
        # Correct the name of the rating agency
        if agencia in ['MOODYS',"Moodys","Moody´s","MOODY´S",""]:
            agencia = "Moody's"
        if agencia in ['Standard & Poors']:
            agencia = 'S&P'
        if agencia in ['FITCH','']:
           agencia = 'Fitch'
        # Replace agency names with None
        if agencia not in ['S&P', 'Fitch',"Moody's"]:
            agencia = None
        # Replace empty strings with None
        if agencia == '':
            agencia = None
    else: agencia = None
    return agencia

df.agencia =  df.agencia.apply(clean_agencia)
#df.agencia.unique()
#Verificando situação base
#df.rating.unique()
#function limpeza rating

def clean_rating(rating):
    # Check if rating is a string
    if isinstance(rating, str):
        # Remove '.br' suffix if present
        if rating.endswith('.br'):
            rating = rating[:-3]
        # Correct the name of the rating agency
        #if rating == 'Moody´s':
        #    rating = 'Moody\'s'
        # Replace agency names with None
        if rating in ['S&P', 'Fitch','Moody´s',"S&P","",]:
            rating = None
        # Replace empty strings with None
        if rating == '':
            rating = None
    return rating

df.rating = df.rating.apply(clean_rating)
#df.rating.unique()
#df.agencia.value_counts()
import pandas as pd
# Define the rating scales for each agency
rating_scales = {
    "Moody's": {
        'Muito Baixo': ['Aaa', 'Aa1', 'Aa2', 'Aa3'],
        'Baixo': ['A1', 'A2', 'A3'],
        'Médio': ['Baa1', 'Baa2', 'Baa3'],
        'Alto': ['Ba1', 'Ba2', 'Ba3'],
        'Muito Alto': ['B1', 'B2', 'B3', 'Caa1', 'Caa2', 'Caa3', 'Ca', 'C']
    },
    'S&P': {
        'Muito Baixo': ['AAA', 'AA+', 'AA', 'AA-'],
        'Baixo': ['A+', 'A', 'A-'],
        'Médio': ['BBB+', 'BBB', 'BBB-'],
        'Alto': ['BB+', 'BB', 'BB-'],
        'Muito Alto': ['B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']
    },
    'Fitch': {
        'Muito Baixo': ['AAA', 'AA+', 'AA', 'AA-'],
        'Baixo': ['A+', 'A', 'A-'],
        'Médio': ['BBB+', 'BBB', 'BBB-'],
        'Alto': ['BB+', 'BB', 'BB-'],
        'Muito Alto': ['B+', 'B', 'B-', 'CCC', 'CC', 'C', 'RD', 'D']
    }
}

# Create a DataFrame from the rating scales
data = []
for agency, scales in rating_scales.items():
    for risk, ratings in scales.items():
        for rating in ratings:
            data.append([agency, rating, risk])
rating_df = pd.DataFrame(data, columns=['agencia_risco', 'rating', 'grau_risco'])

#rating_df

# Create a mapping dictionary from rating_df
rating_map  = dict(zip(zip(rating_df['agencia_risco'], rating_df['rating']), rating_df['grau_risco']))

# Map the 'grau de risco' to VLR
df['grau_risco'] = df.apply(lambda row: rating_map.get((row['agencia'], row['rating'])), axis=1)


df['grau_risco'].value_counts()
import warnings
warnings.filterwarnings('ignore')

df['nr_grau_risco'] = ""
df['nr_grau_risco'][df['grau_risco'] == 'Muito Baixo'] = 1
df['nr_grau_risco'][df['grau_risco'] == 'Baixo'] = 2
df['nr_grau_risco'][df['grau_risco'] == 'Médio'] = 3
df['nr_grau_risco'][df['grau_risco'] == 'Alto'] = 4
df['nr_grau_risco'][df['grau_risco'] == 'Muito Alto'] = 5
df['nr_grau_risco'].value_counts()
df = df[(df.agencia != None) &(df.rating != None) & (df.grau_risco != None) & (df.nr_grau_risco != None) & (df.nr_grau_risco != "")]
df.reset_index(drop=True, inplace=True)
#df.sample(5).T
#salvando backup
df_bkp = df.copy()
#preparando atributos para treinamento
df['titulo_ajust'] = df.tipo + " " +  df.emissor + "-" + df.taxa # nome do título passado para o modelo.
#df['rla'] #retorno líquido anual
df['duration_ano'] = round(df.dc/360) #duration
df.drop_duplicates(subset=['titulo_ajust'], keep='last',inplace=True)
#df['nr_grau_risco']

# optimize_portfolio(df, 25, 10, 3, 100, 20000) #restrica_indiv, restricao_duration, restricao_risk, restricao_total, valor_disponivel)
########################
#####Versao 03
########################
#!pip install ipython-autotime
#%load_ext autotime

import pandas as pd
from scipy.optimize import linprog


#simplex
def optimize_portfolio_simplex(df, rest_indiv, rest_duration, rest_risk, rest_total, valor_disponivel):
    # Convertendo os dados para listas
    retornos = df["rla"].tolist()
    maturidade = df["duration_ano"].tolist()
    risco = df["nr_grau_risco"].tolist()

    # Coeficientes da função objetivo
    c = [-r for r in retornos]  # Negando os retornos para maximizar a função

    # Matriz de coeficientes das restrições
    A = [[1] * len(df)]  # Restrição total de investimento
    A += [[0] * i + [1] + [0] * (len(df) - i - 1) for i in range(len(df))]  # Restrições individuais de investimento
    A.append([-1 if m > rest_duration else 0 for m in maturidade])  # Restrição de maturidade
    A.append([1 if r > rest_risk else 0 for r in risco])  # Restrição de risco
    A.append([1] * len(df))  # Restrição de valor disponível

    # Vetor de limitantes das restrições
    b = [rest_total] + [rest_indiv] * len(df) + [-50, 50] + [valor_disponivel]

    # Limites para as variáveis
    bounds = [(0, rest_indiv)] * len(df)

    # Resolvendo o problema
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='simplex')

    Result_otim = -res.fun
    #list_titulos_idx =
    #print('Resultado ótimo:', -res.fun)

    #for i, x in enumerate(res.x):
    #    if x > 0 :
    #      print(f'{i} - Percentual investido em {df["titulo_ajust"][i]}: {x}%')

    return res

#highs
def optimize_portfolio_highs(df_func, rest_indiv, rest_duration, rest_risk, rest_total, valor_disponivel):
    # Convertendo os dados para listas
    retornos = df_func["rla"].tolist()
    maturidade = df_func["duration_ano"].tolist()
    risco = df_func["nr_grau_risco"].tolist()

    # Coeficientes da função objetivo
    c = [-r for r in retornos]  # Negando os retornos para maximizar a função

    # Matriz de coeficientes das restrições
    A = [[1] * len(df_func)]  # Restrição total de investimento
    A += [[0] * i + [1] + [0] * (len(df_func) - i - 1) for i in range(len(df_func))]  # Restrições individuais de investimento
    A.append([-1 if m > rest_duration else 0 for m in maturidade])  # Restrição de maturidade
    A.append([1 if r > rest_risk else 0 for r in risco])  # Restrição de risco
    A.append([1] * len(df_func))  # Restrição de valor disponível

    # Vetor de limitantes das restrições
    b = [rest_total] + [rest_indiv] * len(df_func) + [-50, 50] + [valor_disponivel]

    # Limites para as variáveis
    bounds = [(0, rest_indiv)] * len(df_func)

    # Resolvendo o problema
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    #print('Resultado ótimo:', -res.fun)

    #for i, x in enumerate(res.x):
    #  if x > 0 :
    #    print(f'{i} - Percentual investido em {df["titulo_ajust"][i]}: {x}%')

    return res
#resultados
#simplex =  optimize_portfolio_simplex(df, 25, 10, 3, 100, 20000) #restrica_indiv, restricao_duration, restricao_risk, restricao_total, valor_disponivel)
#simplex =  optimize_portfolio_simplex(df,  10, 8, 40, 100, 20000) #restrica_indiv, restricao_duration, restricao_risk, restricao_total, valor_disponivel)
#resultados 2
#highs = optimize_portfolio_highs(df, 25, 10, 3, 100, 20000) #restrica_indiv, restricao_duration, restricao_risk, restricao_total, valor_disponivel)
#highs = optimize_portfolio_highs(df, 10, 8, 40, 100, 20000) #restrica_indiv, restricao_duration, restricao_risk, restricao_total, valor_disponivel)
#df2= df.filter(items = [0,1,17,21,22,332,333,853,854,964], axis=0)

#df2.T
#df2.to_csv('result1.csv')
#df.loc[17].reset_index().T


#!pip install flask-ngrok pandas
import streamlit as st
from scipy.optimize import linprog
import pandas as pd
import io
import os
import streamlit as st
from streamlit.components.v1 import html


def main():
  st.title('Otimização de Portfólio de Investimentos em Renda Fixa ')

  st.subheader('Insira os parâmetros para otimizar a alocação.')

  #valor_disponivel = st.number_input('Valor a ser alocado (em R$)', min_value=100, max_value=1000000, value=50000, step=500)
  restricao_indiv = st.slider('Percentual(%) Máximo por Ativo', min_value=5, max_value=50, value=20, step=5)
  restricao_duration = st.slider('Percentual(%) em títulos com duration maior que 10 anos', min_value=1, max_value=50, value=5, step=5)
  restricao_risk = st.slider('Percentual(%) Máximo de Títulos com Grau de Risco Alto e Muito Alto', min_value=0, max_value=100, value=20, step=5)
  restricao_total=100 #(%)
  #isento_ir = st.checkbox('Isento de IR?')
  valor_disponivel = 100000

  if st.button('Otimizar Alocação'):
      # Note: optimize_portfolio_highs and df need to be defined
      #if isento_ir:
     #   res = optimize_portfolio_highs(df[df.tir <= 0], restricao_indiv, restricao_duration, restricao_risk, restricao_total, valor_disponivel)
      #else:
      res = optimize_portfolio_highs(df, restricao_indiv, restricao_duration, restricao_risk, restricao_total, valor_disponivel)
            
      st.write(' ----------------------------------------------------- Resultado da Otimização -----------------------------------------------------')
   
      try:
          st.write(f'Retorno ótimo: {round(-res.fun)} %')
      except Exception as e:
            st.error('Não foi possível identificar combinações no formato solicitado. Tente novamente.')

      
      #st.write(f'Retorno ótimo: {round(-res.fun)} %')
      st.write(f'Total de títulos avaliados: {round(len(df))} ')


      df_optm = pd.DataFrame(columns=['idx','titulo_perc'])
      try:
        for i, x in enumerate(res.x):
          if x > 0 :
            df_optm = df_optm.append({'idx': i, 'titulo_perc': x}, ignore_index=True)
      
      except Exception as e:
         st.error('Não foi possível identificar combinações no formato solicitado. Tente novamente.')
      
      # Note: optimized_return needs to be calculated
      # Convert 'idx' to int for the merge operation
      df_optm['idx'] = df_optm['idx'].astype(int)

      st.write('Alocação Otimizada:')
      # Reset index in df for the merge operation
      df_reset = df.reset_index()
      # Merge the dataframes
      df_final = pd.merge(df_optm, df_reset, left_on='idx', right_index=True)
      df_final = df_final[['titulo_perc','titulo_ajust','duration_ano','tir','rba','rla','grau_risco','nr_grau_risco']]
      df_final.rename(columns={'titulo_perc':'% alocado','titulo_ajust':'Título','duration_ano':'Duration(anos)','tir':'IR(%)','rba':'Retorno Bruto Anual(%)','rla':'Retorno Líquido Anual(%)','grau_risco':'Grau de Risco'}, inplace=True)
      #df_final.to_csv('final.csv')
      # Display the merged dataframe
      print(df.info())
      st.table(df_final.drop(columns=['nr_grau_risco']))

      import plotly.graph_objects as go
      # Suponha que df_final é o seu DataFrame e que ele tem colunas 'retorno_liquido_anual', 'grau_de_risco', e 't
      x = df_final['Retorno Líquido Anual(%)']
      y = df_final['nr_grau_risco']
      text = df_final[['Título','Retorno Líquido Anual(%)']]  # This will be displayed when a point is hovered

      # Create a scatter plot
      fig = go.Figure(data=go.Scatter(x=x, y=y, text=text, mode='markers', hovertemplate='%{text}<extra></extra>'))

      # Update layout for dark background
      fig.update_layout(
          title='Retorno Líquido Anual x Grau de Risco',
          xaxis_title='Retorno Líquido Anual',
          yaxis_title='Grau de Risco',
          autosize=False,
          width=500,
          height=500,
          plot_bgcolor='black',
          paper_bgcolor='black',
          font=dict(color='white')
        )
      st.plotly_chart(fig)  # Envia o gráfico para o Streamlit

if __name__ == '__main__':
    main()
