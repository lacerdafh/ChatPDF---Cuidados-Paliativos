import streamlit as st
import requests
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
def baixar_manual(url, local_path):
    """Baixa o manual PDF da URL especificada."""
    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path

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

# Botão para carregar o manual
if st.button("Baixar e Carregar Manual"):
    manual_url = "https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/guias-e-manuais/2023/manual-de-cuidados-paliativos-2a-edicao/@@download/file"
    local_file_path = "manual-cuidados-paliativos.pdf"
    st.info("Baixando o manual...")
    try:
        file_path = baixar_manual(manual_url, local_file_path)
        documentos = importacao_documentos(file_path)
        documentos = split_de_documentos(documentos)
        vector_store = cria_vector_store(documentos)
        st.success("Manual carregado e processado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao baixar ou processar o manual: {str(e)}")

# Entrada do usuário
pergunta = st.text_input("Digite sua pergunta:")

if st.button("Enviar"):
    if 'vector_store' not in locals():
        st.warning("Por favor, carregue o manual antes de enviar perguntas.")
    else:
        with st.spinner("Gerando resposta..."):
            retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k": 5, "fetch_k": 20})
            chain = final_prompt | llm | {"context": retriever, "pergunta": RunnablePassthrough()}
            resposta = chain.invoke({"pergunta": pergunta})
            resposta_texto = resposta.content if isinstance(resposta, AIMessage) else str(resposta)
            st.write("**Resposta:**")
            st.write(resposta_texto)
