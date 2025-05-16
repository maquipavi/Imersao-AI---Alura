# Projeto de Análise de Personalidade e Propósito de Vida com Agentes de IA

## Descrição

Este projeto utiliza agentes de Inteligência Artificial, impulsionados pelo modelo Gemini do Google, para realizar análises de personalidade e identificar pontos de melhoria com base na data de nascimento do usuário. O sistema busca também exemplos de brasileiros de sucesso que compartilham a mesma data de nascimento, oferecendo um relatório final otimista e motivador.

## Funcionalidades

*   **Análise de Personalidade Detalhada:**  Com base na data de nascimento, o sistema fornece insights sobre pontos fortes, padrões emocionais, comportamento em relacionamentos, e influências da infância.
*   **Identificação de Áreas de Melhoria:**  Um agente especializado analisa os resultados da análise inicial e sugere áreas específicas para desenvolvimento pessoal.
*   **Busca de Modelos de Sucesso:**  O sistema pesquisa e apresenta uma tabela de brasileiros de sucesso que compartilham a data de nascimento do usuário, servindo como inspiração e prova de potencial.
*   **Relatório Final Otimista:**  Todas as informações coletadas são combinadas em um relatório final que visa motivar e inspirar o usuário.
*   **Validação da Data:** Garante que a data de nascimento seja inserida no formato correto (DD/MM/AAAA).
*   **Busca Focada no Brasil:** As buscas de pessoas de sucesso são focadas em brasileiros, tornando os exemplos mais relevantes para o usuário.

## Tecnologias Utilizadas

*   **Python:** Linguagem de programação principal.
*   **Google Gemini API:**  Modelo de linguagem utilizado pelos agentes de IA.
*   **GoogleADK (Agent Development Kit):** Framework do Google para criação e gerenciamento de agentes de IA.
*   **Google GenAI Python SDK:** SDK do Google para interagir com os modelos Gemini.
*   **Pandas:** Biblioteca para criação e manipulação de tabelas de dados.
*   **IPython.display:** Exibição formatada de HTML e Markdown no ambiente Colab.
*   **Google Colab:** Ambiente de desenvolvimento e execução.

## Pré-requisitos

*   Uma conta Google com acesso à API Gemini.
*   Uma chave de API Gemini válida (configurada como `GOOGLE_API_KEY` no ambiente).
*   As seguintes bibliotecas Python instaladas:
    *   `google-genai`
    *   `google-adk`
    *   `pandas`
    *   `requests`

## Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [URL do seu repositório]
    cd [nome do diretório do repositório]
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Crie um arquivo `requirements.txt` com as dependências listadas abaixo, ou instale diretamente)*

    *   `google-genai`
    *   `google-adk`
    *   `pandas`
    *   `requests`
    *   `ipython`  *(se você não estiver usando o Google Colab)*

3.  **Configure a chave da API Gemini:**
    *   No Google Colab, use o recurso `userdata` para armazenar sua chave de API Gemini com o nome `GOOGLE_API_KEY`.
    *   Localmente, defina a variável de ambiente `GOOGLE_API_KEY` com sua chave.

## Utilização

1.  **Abra o projeto no Google Colab:** Carregue o arquivo `.ipynb` no Google Colab.
2.  **Execute as células:** Execute as células do notebook em sequência.
3.  **Siga as instruções:** O sistema solicitará a data de nascimento no formato `DD/MM/AAAA`.
4.  **Analise os resultados:** Os resultados da análise serão exibidos em formato Markdown e HTML.

## Arquitetura

O sistema é composto por quatro agentes de IA, cada um com uma função específica:

1.  **Agente Analisador de Nascimento:**  Realiza a análise inicial da personalidade com base na data de nascimento.
2.  **Agente Identificador de Melhorias:** Identifica áreas de melhoria com base na análise do Agente 1.
3.  **Agente Buscador de Pessoas de Sucesso:**  Pesquisa e apresenta uma tabela de brasileiros de sucesso nascidos na mesma data.
4.  **Agente Gerador de Relatório Final:** Combina as informações dos agentes anteriores em um relatório final otimista.

## Observações

*   A precisão das análises depende da qualidade dos dados retornados pela API Gemini e das buscas na internet.
*   A identificação de pessoas de sucesso pode variar dependendo da disponibilidade de informações online.

## Contribuição

Contribuições são bem-vindas! Se você tiver sugestões de melhorias, correções de bugs ou novas funcionalidades, sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença

[Free License, MIT License]
