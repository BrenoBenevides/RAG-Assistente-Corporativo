# Projeto 1 - Assistente Corporativo de Perguntas e Respostas com RAG Sobre Documentos Internos
# Módulo de RAG

# Imports
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# Carrega variáveis de ambiente
load_dotenv()

# Configurações de Caminhos para Persistência
# Aqui aplicamos o conceito de PERSISTÊNCIA: os dados não somem quando a app fecha.
PERSIST_DIRECTORY = "./chroma_db_data"

# Classe do módulo de RAG
class DSARAGEngine:
    
    def __init__(self):
        
        # 1. Inicializa o Modelo de Embeddings
        # Responsável por transformar texto em vetores numéricos (Embeddings)
        self.embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        
        # 2. Inicializa o LLM (Groq)
        # self.llm = ChatGroq(temperature = 0, model_name = "llama-3.3-70b-versatile")
        self.llm = ChatGroq(temperature = 0, model_name = "openai/gpt-oss-120b")
        
        # 3. Inicializa/Carrega o Banco de Dados Vetorial
        # Se a pasta existir, ele carrega os dados persistidos.
        self.vector_store = Chroma(
            persist_directory = PERSIST_DIRECTORY,
            embedding_function = self.embedding_model,
            collection_name = "rh_policies" 
        )

    def ingest_documents(self, temp_dir_path):
        """
        Lê PDFs, cria chunks, gera embeddings e salva no banco vetorial.
        """
        
        # Carregamento 
        loader = PyPDFDirectoryLoader(temp_dir_path)
        documents = loader.load()
        
        if not documents:
            return "Nenhum documento encontrado."

        # Divisão em Chunks (Para caber no contexto e melhorar a busca)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        
        chunks = text_splitter.split_documents(documents)

        # Inserção no Banco Vetorial 
        # O Chroma gera os Embeddings e salva os Metadados (origem, página)
        self.vector_store.add_documents(documents = chunks)
        
        return f"Processado com sucesso! {len(chunks)} fragmentos de texto adicionados à Collection."

    def get_response(self, query):
        """
        Executa a busca semântica e gera a resposta.
        """
        
        # Cria a Chain de QA (Retrieval Augmented Generation)
        qa_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff",
            retriever = self.vector_store.as_retriever(
                search_kwargs = {"k": 3} # Retorna os 3 chunks mais similares
            ),
            return_source_documents = True # Para mostrar Metadados
        )

        # Executa a query
        response = qa_chain.invoke({"query": query})

        return response
    
    def clear_database(self):
        """
        Deleta a coleção para reiniciar 
        """
        self.vector_store.delete_collection()
        
        # Recria a instância vazia
        self.vector_store = Chroma(
            persist_directory = PERSIST_DIRECTORY,
            embedding_function = self.embedding_model,
            collection_name = "rh_policies"
        )
        return "Banco de dados limpo com sucesso."

# Instância global para ser usada na App
rag_engine = DSARAGEngine()


