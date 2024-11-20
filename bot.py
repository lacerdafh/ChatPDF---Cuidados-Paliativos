from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings
# Import Document class
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Correct import for Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import text splitter


file_path = "D:/5- Asimov acaemy/manual-cuidados-paliativos (1).pdf"
loader = PyPDFLoader(str(file_path))
documentos = loader.load()  # This should already return Document objects, but verify its structure.

batch_size = 8

# File path to the PDF
file_path = "D:/5- Asimov acaemy/manual-cuidados-paliativos (1).pdf"
loader = PyPDFLoader(str(file_path))
raw_documents = loader.load()

# Split documents into chunks using RecursiveCharacterTextSplitter
def split_documents(documents):
    """
    Split raw documents into smaller chunks for better indexing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Maximum size of each chunk
        chunk_overlap=200,  # Overlap between chunks
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

documents = split_documents(raw_documents)

# Function to create FAISS index in batches and save to disk
def create_and_save_faiss_index(documents, batch_size, embedding_model, save_path):
    """
    Create a FAISS index in batches and save it to disk.

    :param documents: List of Document objects to index.
    :param batch_size: Number of documents to process per batch.
    :param embedding_model: Embedding model to use.
    :param save_path: Path to save the FAISS index.
    """
    vector_store = None

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]  # Get a batch of documents
        print(f"Processing batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")
        
        # Create FAISS index for the batch
        temp_vector_store = FAISS.from_documents(batch, embedding_model)
        
        # Merge batch into the main index
        if vector_store is None:
            vector_store = temp_vector_store
        else:
            vector_store.merge_from(temp_vector_store)
    
    # Save the final index to disk
    print(f"Saving FAISS index to {save_path}")
    vector_store.save_local(save_path)
    print("FAISS index saved successfully.")

# Function to load FAISS index from disk
def load_faiss_index(load_path, embedding_model):
    """
    Load a FAISS index from disk.

    :param load_path: Path where the FAISS index is stored.
    :param embedding_model: Embedding model used for the index.
    :return: Loaded FAISS vector store.
    """
    print(f"Loading FAISS index from {load_path}")
    vector_store = FAISS.load_local(
        load_path, 
        embedding_model, 
        allow_dangerous_deserialization=True  # Enable deserialization
    )
    print("FAISS index loaded successfully.")
    return vector_store


# Example usage
if __name__ == "__main__":
    # Embedding model
    embedding_model = OllamaEmbeddings(model="llama3.2:latest")
    
    # Define paths
    save_path = "./faiss_index"
    load_path = save_path

    # Create and save the index
    create_and_save_faiss_index(documents, batch_size=8, embedding_model=embedding_model, save_path=save_path)

    # Load the index
    vector_store = load_faiss_index(load_path, embedding_model)

    # Example query
    query = "What is the first document about?"
    results = vector_store.similarity_search(query, k=2)
    print("Search Results:", results)

    for i in range(0, len(documentos), batch_size):
        batch = documentos[i:i + batch_size]  # Get a batch of documents
        print(f"Processing batch {i // batch_size + 1}/{(len(documentos) + batch_size - 1) // batch_size}")
        
        # Create FAISS index for the batch
        temp_vector_store = FAISS.from_documents(batch, embedding_model)
        
        # Merge batch into the main index
        if vector_store is None:
            vector_store = temp_vector_store
        else:
            vector_store.merge_from(temp_vector_store)
    
    # Save the final index to disk
    print(f"Saving FAISS index to {save_path}")
    vector_store.save_local(save_path)
    print("FAISS index saved successfully.")

# Function to load FAISS index from disk
# Function to load FAISS index from disk
def load_faiss_index(load_path, embedding_model):
    """
    Load a FAISS index from disk.

    :param load_path: Path where the FAISS index is stored.
    :param embedding_model: Embedding model used for the index.
    :return: Loaded FAISS vector store.
    """
    print(f"Loading FAISS index from {load_path}")
    vector_store = FAISS.load_local(
        load_path, 
        embedding_model, 
        allow_dangerous_deserialization=True  # Enable deserialization
    )
    print("FAISS index loaded successfully.")
    return vector_store


# Example usage:
if __name__ == "__main__":
    # If the documentos list is not already a list of Document objects, convert it:
    if not isinstance(documentos[0], Document):
        documentos = [
            Document(page_content=d['page_content'], metadata=d['metadata'])
            for d in documentos
        ]
    
    # Embedding model (Replace with your embedding model)
    embedding_model = OllamaEmbeddings(model="llama3.2:latest")
    
    # Define paths
    save_path = "./faiss_index"
    load_path = save_path

    # Create and save the index
    create_and_save_faiss_index(documentos, batch_size=8, embedding_model=embedding_model, save_path=save_path)

    # Load the index
    vector_store = load_faiss_index(load_path, embedding_model)

    # Example query
    query = "What is the first document about?"
    results = vector_store.similarity_search(query, k=2)
    print("Search Results:", results)
