import streamlit as st
import requests
from PIL import Image
import io
import random

# --- Configuração da Página e Constantes ---
st.set_page_config(
    page_title="AI Baby Generator",
    page_icon="👶",
    layout="wide"
)

# --- AVISO DE SEGURANÇA ---
# A chave de API está exposta aqui por solicitação. 
# Para um aplicativo real, NUNCA exponha chaves no código. Use st.secrets.
API_TOKEN = "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" 

# URL da API do Hugging Face para o modelo Stable Diffusion XL
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Prompts base para garantir a qualidade e o estilo da imagem
BASE_PROMPT_TEMPLATE = "ultra realistic 8k photo, {age_desc}, {skin_tone}, cinematic lighting, professional photography, sharp focus, incredibly detailed"
NEGATIVE_PROMPT = "hands, fingers, deformed hands, mutated hands, arms, blurry, deformed, ugly, disfigured, cartoon, anime, 3d render, painting, text, watermark, signature, extra limbs, missing limbs, body"

# Descrições específicas para cada fase da vida
AGE_PROMPTS = {
    'Bebê': 'extreme close-up face portrait of a cute newborn baby, 1 month old, sleeping peacefully, perfect smooth skin',
    'Criança': 'cute happy child, 6 years old, smiling, headshot portrait',
    'Adolescente': 'portrait of a teenager, 16 years old, natural look, high school photo style',
    'Adulto': 'professional headshot portrait of a young adult, 28 years old, confident expression'
}

# --- Funções ---

def query_api(payload):
    """Envia um prompt para a API do Hugging Face e retorna o conteúdo da imagem."""
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        try:
            error_message = response.json().get('error', 'Erro desconhecido.')
            st.error(f"Erro na API: {error_message} (Status: {response.status_code})")
        except:
            st.error(f"Erro na API (Status: {response.status_code}): {response.text}")
        return None

def create_image_grid(images):
    """Cria uma imagem única a partir de 4 imagens em uma grade 2x2."""
    image_size = 512
    grid_image = Image.new('RGB', (image_size * 2, image_size * 2))
    
    grid_image.paste(images[0].resize((image_size, image_size)), (0, 0))
    grid_image.paste(images[1].resize((image_size, image_size)), (image_size, 0))
    grid_image.paste(images[2].resize((image_size, image_size)), (0, image_size))
    grid_image.paste(images[3].resize((image_size, image_size)), (image_size, image_size))
    
    return grid_image

# --- Interface do Usuário (UI) ---

st.title("👶 AI Baby Generator")
st.markdown("Crie imagens realistas do seu futuro filho(a) em diferentes fases da vida. Siga os passos abaixo.")

# --- PASSO 1: UPLOAD DE FOTOS (SIMULADO) ---
st.subheader("1. Carregue as Fotos dos Pais (Simulação Visual)")
st.info("ℹ️ **Atenção:** O upload das fotos é apenas para simular a interface original. **O modelo de IA não usa as fotos para gerar o resultado.** A imagem é criada com base nas opções de texto na barra lateral.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file_1 = st.file_uploader("Pai 1 / Mãe 1", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    if uploaded_file_1:
        st.image(uploaded_file_1, caption='Pai 1 / Mãe 1', use_column_width=True)

with col2:
    uploaded_file_2 = st.file_uploader("Pai 2 / Mãe 2", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    if uploaded_file_2:
        st.image(uploaded_file_2, caption='Pai 2 / Mãe 2', use_column_width=True)

st.markdown("---")

# --- PASSO 2: GERAÇÃO ---
st.subheader("2. Gere a Imagem")
st.markdown("Use as opções na barra lateral para personalizar e clique em um dos botões abaixo para gerar.")

# Coluna para os botões de geração
gen_col1, gen_col2 = st.columns(2)

# Placeholder para exibir a imagem gerada
result_placeholder = st.empty()


# Barra lateral com as opções de personalização
with st.sidebar:
    st.header("🎨 Personalize a Geração")
    
    gender_input = st.radio("Gênero:", ('Menino', 'Menina', 'Aleatório'), horizontal=True)
    
    skin_tone_options = {
        'Automático': 'beautiful diverse heritage',
        'Muito Claro': 'very light, pale caucasian skin',
        'Claro': 'light, fair caucasian skin',
        'Moreno Claro': 'light brown, mediterranean skin',
        'Moreno Escuro': 'dark brown, south asian skin',
        'Negro': 'deep dark, african skin'
    }
    skin_tone_selection = st.selectbox("Tom de Pele:", options=list(skin_tone_options.keys()))
    
    st.markdown("---")
    
    st.subheader("🖼️ Imagem Única")
    age_selection = st.selectbox("Selecione a Idade:", options=['Bebê', 'Criança', 'Adolescente'])
    
    st.info("Clique nos botões na página principal para gerar a imagem.")
    
    st.markdown("---")
    st.sidebar.info("Este aplicativo usa IA para gerar imagens. Os resultados são artísticos e não uma previsão científica.")


# Lógica dos botões
with gen_col1:
    if st.button("Gerar Imagem Única", use_container_width=True):
        with st.spinner("Gerando sua imagem... ⏳"):
            gender = gender_input if gender_input != 'Aleatório' else random.choice(['Menino', 'Menina'])
            skin_tone = skin_tone_options[skin_tone_selection]
            
            age_desc = AGE_PROMPTS[age_selection].replace('newborn baby', f'newborn baby {gender.lower()}').replace('child', f'{gender.lower()} child').replace('teenager', f'{gender.lower()} teenager')
            final_prompt = BASE_PROMPT_TEMPLATE.format(age_desc=age_desc, skin_tone=skin_tone)
            
            payload = {"inputs": final_prompt, "parameters": {"negative_prompt": NEGATIVE_PROMPT}}
            image_bytes = query_api(payload)
            
            if image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
                result_placeholder.image(image, caption=f"Resultado: {gender} - {age_selection}", use_column_width=True)

with gen_col2:
    if st.button("Gerar Todas as Fases", use_container_width=True, type="primary"):
        with st.spinner("Gerando a progressão de idade... Isso pode levar um minuto!"):
            gender = gender_input if gender_input != 'Aleatório' else random.choice(['Menino', 'Menina'])
            skin_tone = skin_tone_options[skin_tone_selection]
            phases_to_generate = ['Bebê', 'Criança', 'Adolescente', 'Adulto']
            generated_images = []

            progress_bar = st.progress(0, text="Iniciando geração...")
            
            for i, phase in enumerate(phases_to_generate):
                progress_bar.progress((i) / len(phases_to_generate), text=f"Gerando fase: {phase}...")
                
                age_desc = AGE_PROMPTS[phase]
                if gender.lower() == 'menino':
                    age_desc = age_desc.replace('child', 'boy').replace('teenager', 'male teenager').replace('adult', 'man')
                else:
                    age_desc = age_desc.replace('child', 'girl').replace('teenager', 'female teenager').replace('adult', 'woman')
                age_desc = age_desc.replace('newborn baby', f'newborn baby {gender.lower()}')
                
                final_prompt = BASE_PROMPT_TEMPLATE.format(age_desc=age_desc, skin_tone=skin_tone)
                payload = {"inputs": final_prompt, "parameters": {"negative_prompt": NEGATIVE_PROMPT}}
                image_bytes = query_api(payload)
                
                if image_bytes:
                    image = Image.open(io.BytesIO(image_bytes))
                    generated_images.append(image)
                else:
                    st.error(f"Falha ao gerar a fase '{phase}'. Abortando.")
                    break
            
            if len(generated_images) == 4:
                progress_bar.progress(1.0, text="Montando a grade final...")
                grid_image = create_image_grid(generated_images)
                result_placeholder.image(grid_image, caption=f"Progressão de Idade - {gender}", use_column_width=True)
                progress_bar.empty()
