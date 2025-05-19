# app.py

import streamlit as st
import os
import genai as google_genai # Renomeado para evitar conflito com 'from google import genai'
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.generativeai import types as genai_types # Para criar conteúdos (Content e Part)
from datetime import datetime
import textwrap
import requests
import warnings
import re
import pandas as pd

warnings.filterwarnings("ignore")

# --- Configuração Inicial e Funções Auxiliares ---

# Configura a API Key do Google Gemini (DEVE SER FEITO ANTES DE QUALQUER CHAMADA À API)
# Para deploy no Streamlit Cloud, adicione GOOGLE_API_KEY aos segredos da app.
# Para desenvolvimento local, crie um arquivo .streamlit/secrets.toml com:
# GOOGLE_API_KEY = "SUA_API_KEY_AQUI"
try:
    if "GOOGLE_API_KEY" not in os.environ: # Para não sobrescrever se já estiver no ambiente
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    
    # Configura o cliente da SDK do Gemini
    # Renomeei 'google.genai' para 'google_genai' no import para evitar conflito
    # e 'types' para 'genai_types'
    google_genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    client = google_genai.GenerativeModel(model_name="gemini-1.5-flash") # Usando o modelo mais recente para GenerativeModel

except Exception as e:
    st.error(f"Erro ao configurar a API Key do Google Gemini: {e}")
    st.stop()


MODEL_ID_AGENT = "gemini-1.5-flash" # Modelo para os agentes ADK (verificar se o nome está correto para ADK)
# Nota: O ADK pode ter requisitos específicos de nome de modelo. 
# "gemini-2.0-flash" não é um nome de modelo padrão conhecido para a API Gemini.
# "gemini-1.5-flash" é mais comum. Ajuste se necessário.


# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final
def call_agent(agent: Agent, message_text: str) -> str:
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message_text)])

    final_response = ""
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text is not None:
                    final_response += part.text
                    final_response += "\n"
    return final_response

# Função auxiliar para formatar texto em Markdown (simplificada para Streamlit)
def to_streamlit_markdown(text):
    text = text.replace('•', '  *')
    # O textwrap.indent pode ser muito agressivo para o layout do Streamlit.
    # Se precisar de indentação, considere usar blockquotes ou formatação manual.
    return textwrap.indent(text, '> ', predicate=lambda _: True) # Mantido como no original

# --- Definições dos Agentes ---

##########################################
# --- Agente 1: Analisador de Nascimento --- #
##########################################
#@st.cache_resource # Cacheia a criação do agente para não recriar a cada rerun
def criar_agente_analisador():
    return Agent(
        name="agente_analisador",
        model=MODEL_ID_AGENT,
        instruction="""
        Você é um analista de personalidade e propósito de vida com base na data de nascimento.
        Sua tarefa é fornecer análises profundas e precisas sobre a personalidade, padrões emocionais,
        caminhos de carreira e desafios pessoais com base na data de nascimento fornecida.
        Use a ferramenta de busca do Google (google_search) para obter informações relevantes e
        garantir que as análises sejam fundamentadas e úteis.
        """,
        description="Agente que analisa a personalidade e o propósito de vida com base na data de nascimento",
        tools=[google_search]
    )

def rodar_agente_analisador(data_nascimento_str: str):
    agente = criar_agente_analisador()
    entrada_do_agente_analisador = f"""
    Data de Nascimento: {data_nascimento_str}

    Realize as seguintes análises:
    1. Decodificador de Personalidade pela Data de Nascimento
    “Com base na data de nascimento {data_nascimento_str}, descreva meus pontos fortes naturais, padrões emocionais e como me comporto em relacionamentos — que seja profundo, específico e psicologicamente preciso.”
    2. Roteiro da Infância
    “Usando a data de nascimento {data_nascimento_str}, escreva um perfil psicológico de como minha infância moldou minha personalidade, hábitos e tomada de decisões hoje — seja gentil, mas revelador.”
    3. Analisador de Propósito Profissional
    “Dada a data de nascimento {data_nascimento_str}, quais caminhos de carreira combinam com meus traços de personalidade, valores e talentos naturais? Sugira áreas, funções e ambientes de trabalho.”
    4. Detector de Auto-Sabotagem
    “Com base na data {data_nascimento_str}, quais são meus hábitos de auto-sabotagem mais prováveis e como eles aparecem no dia a dia? Dê soluções práticas com base na psicologia.”
    5. Mapa de Gatilhos Emocionais
    “Usando a data de nascimento {data_nascimento_str}, explique o que geralmente me desencadeia emocionalmente, como eu costumo reagir e como posso desenvolver resiliência emocional em torno desses padrões.”
    6. Escaneamento de Energia nos Relacionamentos
    “Com base na data de nascimento {data_nascimento_str}, descreva como eu dou e recebo amor, o que preciso de um parceiro e que tipo de pessoa eu naturalmente atraio.”
    """
    return call_agent(agente, entrada_do_agente_analisador)

################################################
# --- Agente 2: Identificador de Melhorias --- #
################################################
#@st.cache_resource
def criar_agente_melhorias():
    return Agent(
        name="agente_melhorias",
        model=MODEL_ID_AGENT,
        instruction="""
        Você é um consultor de desenvolvimento pessoal. Sua tarefa é analisar as análises fornecidas
        pelo Agente 1 (analisador de nascimento) e identificar áreas de melhoria em cada uma das seis
        categorias. Seja específico e forneça sugestões práticas para o desenvolvimento pessoal.
        """,
        description="Agente que identifica pontos de melhoria nas análises do Agente 1",
        tools=[google_search]
    )

def rodar_agente_melhorias(data_nascimento_str: str, analises_agente1: str):
    agente = criar_agente_melhorias()
    entrada_do_agente_melhorias = f"""
    Data de Nascimento: {data_nascimento_str}
    Análises do Agente 1: {analises_agente1}

    Para cada uma das seis análises fornecidas pelo Agente 1, identifique áreas de melhoria e
    forneça sugestões práticas para o desenvolvimento pessoal.
    """
    return call_agent(agente, entrada_do_agente_melhorias)

######################################
# --- Agente 3: Buscador de Pessoas de Sucesso --- #
######################################
#@st.cache_resource
def criar_agente_buscador_sucesso():
    # Nota: "gemini-2.5-flash-preview-04-17" pode não ser um nome válido.
    # Usando MODEL_ID_AGENT como padrão, ajuste se necessário.
    # O nome do modelo no ADK é geralmente mais simples, como "gemini-pro" ou "gemini-1.5-flash".
    return Agent(
        name="agente_buscador_sucesso",
        model=MODEL_ID_AGENT, # Usando o modelo padrão, ajuste se "gemini-2.5-flash-preview-04-17" for específico e correto para ADK
        instruction="""
            Você é um pesquisador de pessoas de sucesso brasileiras. Sua tarefa é buscar na internet 5 homens e 5 mulheres
            que nasceram na mesma data fornecida e que alcançaram sucesso em suas áreas de atuação, e que sejam brasileiros.
            Monte uma tabela com as seguintes colunas: nome, profissão, no que a pessoa tem sucesso e site da informação.
            A tabela deve ser formatada em Markdown.
            Ao realizar a busca no Google, certifique-se de incluir o termo "brasileiro" ou "brasileira" para garantir que os resultados sejam apenas de pessoas do Brasil.
            Use a ferramenta de busca do Google (google_search) para encontrar as informações e o site de onde tirou a informação.
            Retorne APENAS a tabela em formato Markdown, começando com o cabeçalho | Nome | Profissão | Sucesso | Site da Informação |.
            """,
        description="Agente que busca pessoas de sucesso nascidas na mesma data",
        tools=[google_search]
    )

def rodar_agente_buscador_sucesso(data_nascimento_str: str):
    agente = criar_agente_buscador_sucesso()
    entrada_do_agente_buscador_sucesso = f"""
    Data de Nascimento: {data_nascimento_str}

    Busque na internet 5 homens e 5 mulheres que nasceram na mesma data e que alcançaram sucesso
    em suas áreas de atuação e que sejam brasileiros. Monte uma tabela com as seguintes colunas:
    nome, profissão, no que a pessoa tem sucesso e site da informação. Ao realizar a busca no Google, certifique-se de incluir o termo "brasileiro" ou "brasileira" para garantir que os resultados sejam apenas de pessoas do Brasil. Seja claro e objetivo.
    Retorne APENAS a tabela em formato Markdown, começando com o cabeçalho | Nome | Profissão | Sucesso | Site da Informação |.
    """
    tabela_markdown_sucesso = call_agent(agente, entrada_do_agente_buscador_sucesso)

    # Tentativa de converter para HTML para melhor controle no Streamlit,
    # mas se o LLM já fornecer Markdown, podemos usar isso diretamente.
    # A conversão para DataFrame e depois HTML é mais robusta se o Markdown do LLM variar.
    try:
        lines = tabela_markdown_sucesso.strip().split('\n')
        header = [h.strip() for h in lines[0].strip('|').split('|')]
        
        data_rows = []
        for line in lines[2:]: # Pula a linha de separador do Markdown (---|---|...)
            if not line.strip() or not line.startswith("|"): # Ignora linhas vazias ou que não parecem de tabela
                continue
            values = [v.strip() for v in line.strip('|').split('|')]
            if len(values) == len(header):
                data_rows.append(dict(zip(header, values)))
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            return df.to_html(index=False, escape=False, classes=["table", "table-striped", "table-hover"])
        else: # Se não conseguiu parsear, retorna o markdown original
            return tabela_markdown_sucesso

    except Exception as e:
        st.warning(f"Erro ao parsear tabela de sucesso para HTML, mostrando Markdown: {e}")
        return tabela_markdown_sucesso # Retorna o markdown cru em caso de erro no parsing

##########################################
# --- Agente 4: Gerador de Relatório Final --- #
##########################################
#@st.cache_resource
def criar_agente_relatorio_final():
    return Agent(
        name="agente_relatorio",
        model=MODEL_ID_AGENT,
        instruction="""
        Você é um gerador de relatórios finais. Sua tarefa é combinar as análises do Agente 1,
        os pontos de melhoria do Agente 2 e a tabela de pessoas de sucesso do Agente 3 em um
        relatório final otimista e motivador. Formate bem o relatório usando Markdown.
        Conclua o relatório com uma mensagem de incentivo.
        """,
        description="Agente que gera o relatório final",
        # tools=[] # Não precisa de busca aqui
    )

def rodar_agente_relatorio_final(data_nascimento_str: str, analises: str, melhorias: str, tabela_sucesso_html_ou_md: str):
    agente = criar_agente_relatorio_final()
    entrada_do_agente_relatorio = f"""
    Data de Nascimento: {data_nascimento_str}

    Análises Detalhadas (Insights do Agente 1):
    {analises}

    Áreas de Desenvolvimento e Sugestões Práticas (Insights do Agente 2):
    {melhorias}

    Inspiração: Pessoas de Sucesso Nascidas na Mesma Data (Dados do Agente 3):
    {tabela_sucesso_html_ou_md}

    Combine as informações acima em um relatório final coeso, bem estruturado, otimista e motivador.
    Use títulos e subtítulos em Markdown para organizar o relatório.
    Conclua o relatório com uma mensagem de incentivo poderosa e personalizada.
    """
    return call_agent(agente, entrada_do_agente_relatorio)


# --- Interface do Streamlit ---
st.set_page_config(page_title="Análise de Personalidade e Propósito", layout="wide")
st.title("🔮 Analisador de Personalidade e Propósito de Vida 🔮")
st.markdown("Descubra insights sobre você com base na sua data de nascimento!")

# --- Obter a Data de Nascimento do Usuário ---
data_nascimento_dt = st.date_input(
    "📅 Por favor, selecione sua DATA DE NASCIMENTO:",
    value=None, # datetime.date(2000, 1, 1), # Data padrão opcional
    min_value=datetime(1900, 1, 1),
    max_value=datetime.now(),
    format="DD/MM/YYYY"
)

if data_nascimento_dt:
    data_nascimento_str = data_nascimento_dt.strftime('%d/%m/%Y')
    st.info(f"Data selecionada: {data_nascimento_str}")

    if st.button("🔍 Gerar Análise Completa", type="primary"):
        with st.spinner("Processando sua análise... Isso pode levar alguns minutos... ⏳"):
            try:
                # --- Execução do Sistema de Agentes ---
                st.subheader("🌟 Iniciando o Sistema de Análise 🌟")

                with st.status("Agente 1: Analisando personalidade...", expanded=True) as status1:
                    analises_agente1 = rodar_agente_analisador(data_nascimento_str)
                    st.markdown("---")
                    st.markdown("### 📝 Resultado do Agente 1 (Analisador de Nascimento)")
                    st.markdown(to_streamlit_markdown(analises_agente1))
                    status1.update(label="Análise de personalidade concluída!", state="complete")

                with st.status("Agente 2: Identificando melhorias...", expanded=True) as status2:
                    pontos_de_melhoria = rodar_agente_melhorias(data_nascimento_str, analises_agente1)
                    st.markdown("---")
                    st.markdown("### 🌱 Resultado do Agente 2 (Identificador de Melhorias)")
                    st.markdown(to_streamlit_markdown(pontos_de_melhoria))
                    status2.update(label="Identificação de melhorias concluída!", state="complete")

                with st.status("Agente 3: Buscando pessoas de sucesso...", expanded=True) as status3:
                    tabela_sucesso_html_ou_md = rodar_agente_buscador_sucesso(data_nascimento_str)
                    st.markdown("---")
                    st.markdown("### 🏆 Resultado do Agente 3 (Buscador de Pessoas de Sucesso)")
                    if "<table>" in tabela_sucesso_html_ou_md: # Se for HTML
                         st.markdown(tabela_sucesso_html_ou_md, unsafe_allow_html=True)
                    else: # Se for Markdown (fallback)
                         st.markdown(tabela_sucesso_html_ou_md)
                    status3.update(label="Busca por pessoas de sucesso concluída!", state="complete")

                with st.status("Agente 4: Gerando relatório final...", expanded=True) as status4:
                    relatorio_final = rodar_agente_relatorio_final(data_nascimento_str, analises_agente1, pontos_de_melhoria, tabela_sucesso_html_ou_md)
                    st.markdown("---")
                    st.markdown("### 📜 Resultado do Agente 4 (Gerador de Relatório Final)")
                    st.markdown(relatorio_final) # O agente 4 já deve formatar em Markdown
                    status4.update(label="Relatório final gerado!", state="complete")

                st.success("🎉 Análise Completa Gerada com Sucesso! 🎉")

            except Exception as e:
                st.error(f"Ocorreu um erro durante a análise: {e}")
                st.exception(e) # Mostra o traceback para debug
else:
    st.warning("Por favor, insira sua data de nascimento para começar.")

st.markdown("---")
st.markdown("Desenvolvido com 🧠 por [Seu Nome/Organização] usando Google Gemini e Streamlit.")