import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough

# Configuração do LLM
llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.1,
    num_predict=256,
)

# Exemplos de perguntas e respostas
examples = [
    {
        "pergunta": "Como posso utilizar morfina para dor?",
        "sistema": "Pergunta de acompanhamento necessária: Sim. ...",
        "resposta": "Segundo o Manual de Cuidados Paliativos, 2ª ed.: ..."
    },
    {
        "pergunta": "Quais são os efeitos colaterais da morfina?",
        "sistema": "Pergunta de acompanhamento necessária: Sim. ...",
        "resposta": "Segundo o Manual de Cuidados Paliativos, 2ª ed.: ..."
    },
]

# Template Base
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{pergunta}"),
    ("system", "{sistema}"),
    ("assistant", "{resposta}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        '''
        Você é um Chatbot que auxilia profissionais de saúde em cuidados paliativos com base apenas no Manual ...
        '''
    ),
    few_shot_prompt,
    ("human", "{pergunta}"),
])

# Funções para carregar e processar documentos
def importacao_documentos(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_de_documentos(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documentos = splitter.split_documents(documentos)
    for i, doc in enumerate(documentos):
        doc.metadata['doc_id'] = i
    return documentos

def cria_vector_store(documentos):
    embedding_model = OllamaEmbeddings(model="llama3.2:latest")
    return FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )

# Interface do Streamlit
st.title("Chatbot para Cuidados Paliativos")
file_path = st.text_input("Caminho do arquivo PDF:", "D:/5- Asimov acaemy/manual-cuidados-paliativos (1).pdf")

if st.button("Carregar e Processar Documento"):
    with st.spinner("Carregando documento..."):
        documentos = importacao_documentos(file_path)
        documentos = split_de_documentos(documentos)
        vector_store = cria_vector_store(documentos)
        st.success("Documento processado com sucesso!")

# Configuração do retriever e do chain
retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k": 5, "fetch_k": 20})
persist_directory = "./chroma_vectorstore"
embeddings = OllamaEmbeddings(model="llama3.2:latest")
vectorstore = Chroma.from_documents(
    documents=documentos,
    embedding=embeddings,
    persist_directory=persist_directory,
)
chain = final_prompt | llm | {"context": retriever, "pergunta": RunnablePassthrough()}

# Entrada do usuário
pergunta = st.text_input("Digite sua pergunta:")

if st.button("Enviar"):
    with st.spinner("Gerando resposta..."):
        resposta = chain.invoke({"pergunta": pergunta})
        resposta_texto = resposta.content if isinstance(resposta, AIMessage) else str(resposta)
        st.write("**Resposta:**")
        st.write(resposta_texto)
