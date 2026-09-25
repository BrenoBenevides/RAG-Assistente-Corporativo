# Projeto 1 - Assistente Corporativo de Perguntas e Respostas com RAG Sobre Documentos Internos
# Módulo da App

# Imports
import os
import tempfile
import streamlit as st
from dsa_rag_engine import rag_engine

# Variável de ambiente
# Evita um comportamento indesejado do Hugging Face Tokenizers quando usado dentro de aplicações que executam várias threads em paralelo, 
# como Streamlit, FastAPI ou Jupyter.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuração Inicial da Aplicação Streamlit
st.set_page_config(
    page_title="Data Science Academy",  # Título que aparece na aba do navegador
    page_icon=":100:",                  # Ícone (emoji) que aparece na aba do navegador
    layout="wide",                      # Define o layout da página para usar a largura total da tela
    initial_sidebar_state="expanded",   # Garante que a sidebar (menu lateral) comece aberta
)

# Títulos
st.title("Data Science Academy")
st.title("🤖 Assistente de RH Corporativo")
st.markdown("""
Este sistema utiliza **RAG (Retrieval-Augmented Generation)** para responder perguntas 
baseadas nas políticas internas da empresa.
""")

# --- Sidebar: Área de Gestão de Documentos (CRUD e Ingestão) ---
with st.sidebar:

    st.header("📂 Gestão de Conhecimento")
    
    # Upload de arquivos
    uploaded_files = st.file_uploader(
        "Carregar Novas Políticas (PDF)", 
        type = ["pdf"], 
        accept_multiple_files = True
    )
    
    if st.button("Processar Documentos"):
        
        if uploaded_files:
            
            with st.spinner("Processando (Lendo, Vetorizando e Indexando)..."):
                
                # Cria diretório temporário para salvar os arquivos para o Loader ler
                with tempfile.TemporaryDirectory() as temp_dir:
                    
                    for uploaded_file in uploaded_files:
                        
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Chama a engine para processar
                    result_msg = rag_engine.ingest_documents(temp_dir)

                    st.success(result_msg)
        else:
            st.warning("Por favor, faça upload de arquivos PDF.")

    st.markdown("---")
    
    if st.button("🗑️ Limpar Banco de Dados"):
        msg = rag_engine.clear_database()
        st.warning(msg)

    st.markdown("---")

    st.sidebar.markdown(
        """
        <div style="background-color:#00CC96; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 15px;">
            <h3 style="color:white; margin:0; font-weight:bold;">Dúvidas?</h3>
            <h3 style="color:white; margin:0; font-weight:bold;">suporte@datascienceacademy.com.br</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Área Principal: Chat ---

# Inicializa histórico de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Pergunte sobre férias, benefícios, home office..."):
    
    # 1. Adiciona pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Gera resposta usando o RAG Engine
    with st.chat_message("assistant"):
        
        with st.spinner("Consultando base vetorial..."):
            
            # Obtém a resposta completa
            response_payload = rag_engine.get_response(prompt)
            
            # Extrai a resposta
            answer = response_payload['result']
            
            # Extrai os metadados
            sources = response_payload['source_documents']
            
            # Exibe a resposta
            st.markdown(answer)
            
            # --- Exibindo Metadados ---
            with st.expander("📚 Fontes Consultadas (Metadados)"):
                
                for doc in sources:
                    
                    # Extrai metadados do documento recuperado
                    source_name = doc.metadata.get('source', 'Desconhecido')
                    page_num = doc.metadata.get('page', 'N/A')
                    preview = doc.page_content[:150] + "..."
                    
                    st.markdown(f"**Fonte:** `{os.path.basename(source_name)}` | **Página:** `{page_num}`")
                    st.caption(f"Trecho: {preview}")

    # 3. Adiciona resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": answer})


# Fim
