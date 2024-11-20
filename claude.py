import streamlit as st
from typing import List, Dict, Any
from pathlib import Path
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings
import os

from typing import List

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.faiss import FAISS

from langchain_ollama import OllamaEmbeddings



def load_pdfs(pdf_directory: str) -> List:

    """

    Carrega todos os arquivos PDF de um diretório

    """

    documents = []

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    

    for pdf_file in pdf_files:

        file_path = os.path.join(pdf_directory, pdf_file)

        loader = PyPDFLoader(file_path)

        documents.extend(loader.load())

    

    return documents



def create_faiss_index(

    pdf_directory: str,

    output_dir: str = "./faiss_index",

    chunk_size: int = 1000,

    chunk_overlap: int = 200

):

    """

    Cria um índice FAISS a partir de documentos PDF

    """

    # Criar diretório de saída se não existir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    

    # Carregar documentos

    print("Carregando documentos...")

    documents = load_pdfs(pdf_directory)

    

    # Dividir documentos em chunks

    print("Dividindo documentos em chunks...")

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=chunk_size,

        chunk_overlap=chunk_overlap,

        length_function=len,

    )

    texts = text_splitter.split_documents(documents)

    

    # Criar embeddings

    print("Criando embeddings...")

    embeddings = OllamaEmbeddings(model="llama3.2:latest")

    

    # Criar e salvar índice FAISS

    print("Criando índice FAISS...")

    vector_store = FAISS.from_documents(texts, embeddings)

    

    # Salvar índice

    print(f"Salvando índice em {output_dir}...")

    vector_store.save_local(output_dir)

    

    print("Índice criado com sucesso!")

    return vector_store



if __name__ == "__main__":

    # Configurar diretórios

    PDF_DIR = "D:\\5- Asimov acaemy\\novo_chat_pdf\\"  # Diretório onde estão seus PDFs

    OUTPUT_DIR = "./faiss_index"  # Diretório onde o índice será salvo

    

    try:

        index = create_faiss_index(

            pdf_directory=PDF_DIR,

            output_dir=OUTPUT_DIR,

            chunk_size=1000,

            chunk_overlap=200

        )

        print(f"Índice FAISS criado com sucesso em: {OUTPUT_DIR}")

    except Exception as e:

        print(f"Erro ao criar índice: {str(e)}")
        
# [Previous imports remain the same...]

class FAISSRetriever:
    def __init__(
        self,
        index_path: str,
        embedding_model_name: str = "llama3.2:latest",
        llm_model_name: str = "llama3.2:latest",
        search_kwargs: Dict[str, Any] = {"k": 4},
        use_memory: bool = True
    ):
        self.index_path = Path(index_path)
        self.embedding_model = OllamaEmbeddings(model=embedding_model_name)
        self.search_kwargs = search_kwargs
        
        # Carregar o índice FAISS
        self.vector_store = FAISS.load_local(
            str(self.index_path),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Configurar o LLM com contexto profissional médico
        self.llm = ChatOllama(
            model=llm_model_name,
            system_prompt="""Você é um assistente especializado em cuidados paliativos, 
            fornecendo informações baseadas no Manual de Cuidados Paliativos para profissionais de saúde.
            Forneça informações técnicas precisas sobre medicamentos, procedimentos e recomendações conforme
            documentado no manual, incluindo dosagens e diretrizes específicas quando disponíveis.
            Sempre cite a seção específica do manual de onde a informação foi extraída.
            
            Lembre-se:
            1. Você está se comunicando com profissionais de saúde
            2. Forneça informações técnicas completas sobre medicamentos, incluindo dosagens quando disponíveis
            3. Cite especificamente as seções do manual
            4. Mantenha o foco nas diretrizes e recomendações oficiais do manual
            5. Se uma informação específica não estiver no manual, indique isso claramente
            """
        )
        
        # Configurar memória
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        ) if use_memory else None
        
        # Criar a chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs=self.search_kwargs
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
    
    def simple_search(self, query: str) -> List[Document]:
        return self.vector_store.similarity_search(
            query,
            k=self.search_kwargs.get("k", 4)
        )
    
    def query_with_context(self, query: str) -> Dict[str, Any]:
        result = self.chain.invoke({"question": query})
        return {
            "answer": result["answer"],
            "source_documents": result["source_documents"],
            "chat_history": self.memory.chat_memory.messages if self.memory else None
        }

def initialize_retriever():
    if 'retriever' not in st.session_state:
        st.session_state.retriever = FAISSRetriever(
            index_path="./faiss_index",
            embedding_model_name="llama3.2:latest",
            llm_model_name="llama3.2:latest",
            search_kwargs={"k": 4},
            use_memory=True
        )

def display_chat_message(role: str, content: str, key: str):
    """Exibe mensagens do chat com estilo personalizado"""
    if role == "user":
        st.markdown(f"🧑‍💻 **Você**: {content}")
    elif role == "assistant":
        st.markdown(f"🤖 **Assistente**: {content}")
    else:
        st.markdown(f"📚 **{role}**: {content}")

def main():
    # Exemplo de uso
    index_path = "./faiss_index"
    st.set_page_config(
        page_title="Chat com Manual de Cuidados Paliativos - Exclusivo para Profissionais de Saúde",
        page_icon="🏥",
        layout="wide"
    )

    st.title("💬 Manual de Cuidados Paliativos - Consulta Profissional")
    st.markdown("---")
    
    # Adicionar aviso de uso profissional
    st.warning("""
        ⚕️ **ATENÇÃO**: Este sistema é destinado exclusivamente para uso por profissionais de saúde.
        As informações fornecidas são baseadas no Manual de Cuidados Paliativos e incluem detalhes técnicos
        sobre medicamentos e procedimentos.
    """)

    # Inicializar o retriever
    initialize_retriever()

    # Inicializar histórico do chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Layout em duas colunas
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Chat")
        # Área de chat
        for message in st.session_state.chat_history:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("key", "default")
            )

        # Input do usuário
        user_input = st.text_input(
            "Digite sua pergunta:",
            key="user_input",
            placeholder="Ex: Qual é o objetivo dos cuidados paliativos?"
        )

        if st.button("Enviar", key="send_button"):
            if user_input:
                # Adicionar pergunta ao histórico
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "key": f"user_{len(st.session_state.chat_history)}"
                })

                try:
                    # Buscar resposta
                    with st.spinner("Pesquisando resposta..."):
                        result = st.session_state.retriever.query_with_context(user_input)
                        
                        # Adicionar resposta ao histórico
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "key": f"assistant_{len(st.session_state.chat_history)}"
                        })
                        
                        # Adicionar fontes ao histórico
                        sources_text = "\n\n**Fontes consultadas:**\n"
                        for i, doc in enumerate(result["source_documents"], 1):
                            sources_text += f"\n**Documento {i}:**\n"
                            sources_text += f"```\n{doc.page_content[:2500]}...\n```\n"
                        
                        st.session_state.chat_history.append({
                            "role": "system",
                            "content": sources_text,
                            "key": f"sources_{len(st.session_state.chat_history)}"
                        })

                    # Recarregar a página para mostrar as novas mensagens
                    st.rerun()  # Changed from st.experimental_rerun()

                except Exception as e:
                    st.error(f"Erro ao processar a pergunta: {str(e)}")

    with col2:
        st.subheader("Configurações")
        
        # Configurações do modelo
        st.slider(
            "Número de documentos a consultar (k)",
            min_value=1,
            max_value=10,
            value=4,
            key="k_value"
        )
        
        # Botão para limpar histórico
        if st.button("Limpar Histórico", key="clear_history"):
            st.session_state.chat_history = []
            st.rerun()  # Changed from st.experimental_rerun()

        # Exibir informações
        st.markdown("---")
        st.markdown("### Sobre")
        st.markdown("""
        📚 **Fonte**: Manual de Cuidados Paliativos  
        ⚕️ **Uso**: Exclusivo para profissionais de saúde  
        ⚠️ **Aviso**: As informações fornecidas devem ser utilizadas em conjunto com o julgamento clínico profissional
        """)

if __name__ == "__main__":
    main()
