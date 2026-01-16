import os
import pandas as pd
from datetime import date
from groq import Groq

client = Groq(
    api_key="GROQ_API_KEY",
)

df = pd.read_csv('desafio_dio.csv')


def calcular_idade(data_nascimento):
    hoje = date.today()
    idade = hoje.year - data_nascimento.year
    if (hoje.month, hoje.day) < (data_nascimento.month, data_nascimento.day):
        idade -= 1
    return idade

df["data_nascimento"] = pd.to_datetime(df["data_nascimento"])
df["idade"] = df["data_nascimento"].apply(calcular_idade)


def gerar_recomendacao(user):
    nome = user["nome"]
    idade = user["idade"]
    genero = user["genero_favorito"]

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um assistente de recomendação de filmes e séries. "
                    "Sugira conteúdos existentes de forma personalizada. "
                    "Regras: Máximo 2 títulos; Apenas obras conhecidas; "
                    "Sem propaganda de streamings; Linguagem clara e objetiva; "
                    "Fale diretamente com o usuário pelo nome."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Crie uma recomendação para: Nome: {nome}, Idade: {idade}, "
                    f"Gênero favorito: {genero}. Responda diretamente a {nome}, "
                    "recomende 2 títulos e explique brevemente o porquê."
                ),
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content.strip('\"')


recomendacoes = []

for _, user in df.iterrows(): 
    recomendacao = gerar_recomendacao(user) 
    recomendacoes.append(recomendacao) 
    
df['recomendacao'] = recomendacoes

df.to_csv("desafio_dio_final.csv", index=False)