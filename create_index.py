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