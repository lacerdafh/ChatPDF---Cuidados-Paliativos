from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

#from dotenv import load_dotenv, find_dotenv

#from configs import *

#_ = load_dotenv(find_dotenv())

llm=ChatOllama(
    model="llama3.2:latest",
    temperature = 0.1,
    num_predict = 256,
    # other params ...
)


### Exemplos de pergunta resposta
# Definição de exemplos
examples = [
            {
                
                    "pergunta": "Como posso utilizar morfina para dor?",
                    "sistema": """
                    Pergunta de acompanhamento necessária: Sim.
                    1. Pergunta de acompanhamento: Existe referência a algum capítulo no texto?
                    Resposta intermediária: Sim.
                    2. Pergunta de acompanhamento: Qual capítulo é relevante?
                    Resposta intermediária: Dor.
                    3. Pergunta de acompanhamento: Dentro do capítulo 'Dor', existe referência a algum subtítulo?
                    Resposta intermediária: Sim.
                    4. Pergunta de acompanhamento: Qual subtítulo?
                    Resposta intermediária: Morfina.
                    """,
                "resposta": """
                    Segundo o Manual de Cuidados Paliativos, 2ª ed.:
                    Para dor, a morfina pode ser utilizada da seguinte forma:
                    Morfina simples:
                    - Dose inicial: 5 mg a cada 4 horas (VO), com necessidade de avaliar doses mais baixas em pacientes idosos, com disfunção renal ou hepática;
                    - Dose máxima: Não possui dose teto; o limite é o efeito colateral, devendo ser titulado cuidadosamente;
                    - Frequência de administração: A cada 4 horas. Em casos específicos (idosos, disfunções), considerar a cada 6 horas;
                    - Vias de administração: Oral, sonda nasoenteral, gastrostomia, endovenosa, subcutânea, hipodermóclise;
                    - Equipotência: Morfina endovenosa é três vezes mais potente que a oral;
                    - Particularidades: Metabolizada no fígado e excretada pelo rim. Usar com cautela em pacientes com doença hepática ou renal;
                    - Disponibilidade no SUS: Constante na Rename 2022."""                    
            },
        
            {
                
                "pergunta": "Quais são os efeitos colaterais da morfina?",
                "sistema": """
                Pergunta de acompanhamento necessária: Sim.
                1. Existe um capítulo relacionado? Sim.
                Resposta intermediária: Dor.
                2. Algum subtítulo é relevante? Sim.
                Resposta intermediária: Efeitos colaterais.""",

                "resposta": 
                """Segundo o Manual de Cuidados Paliativos, 2ª ed.:
                Os principais efeitos colaterais da morfina incluem náuseas, vômitos, 
                constipação, sonolência e depressão respiratória. Esses efeitos devem ser monitorados e manejados adequadamente.""" 
            },

            {
                
                "pergunta": "Quem são os autores do Capitulo de Dor?",
                "sistema": """
            Pergunta de acompanhamento necessária: Sim.
            1. Existe um capítulo relacionado? Sim.
            Resposta intermediária: Dor.
            2. Algum subtítulo é relevante? Não.
            Resposta intermediária: Efeitos colaterais.
            3. Foi perguntado sobre nomes de pessoas? Sim.
            Resposta intermediária: Sim.
            4. É sobre o livro ou capítulo?
            Resposta intermediária: capítulo.""",


            "resposta": """
                        
                        Os autores do Capítulo de Dor do Manual de Cuidados Paliativos, 2ª ed., são:
                        1- Daniel Felgueiras Rolo
                        2- Maria Perez Soares D'Alessandro
                        3- Gustavo Cassefo
                        4- Sergio Seiki Anagusko
                        5- Ana Paula Mirarchi Vieira Maiello
                        """
                         
            },

        {
            "pergunta": "Quem são os autores do livro?", 
            "sistema": """
            Pergunta de acompanhamento necessária: Sim.
            1. Existe um capítulo relacionado? 
            Resposta intermediária: Não.
            2. Algum subtítulo é relevante? 
            Resposta intermediária: Não.
            3. Foi perguntado sobre nomes de pessoas? Sim.
            Resposta intermediária: Sim.
            4. É sobre o livro ou capítulo?
            Resposta intermediária: livro.""",


            "resposta": 
                        """
           
                A equipe responsável pelo Manual de Cuidados Paliativos, 2ª ed., foram:

               ## Editores ##
                Maria Perez Soares D’Alessandro, Lara Cruvinel Barbosa, Sergio Seiki Anagusko, Ana Paula Mirarchi Vieira
                Maiello, Catherine Moreira Conrado, Carina Tischler Pires e Daniel Neves Forte.

               ## Autores ##
                Aline de Almada Messias, Ana Cristina Pugliese de Castro, Ana Paula Mirarchi Vieira Maiello, Caroline
                Freitas de Oliveira, Catherine Moreira Conrado, Daniel Felgueiras Rolo, Fábio Holanda Lacerda, Fernanda
                Pimentel Coelho, Fernanda Spiel Tuoto, Graziela de Araújo Costa , Gustavo Cassefo, Heloisa Maragno,
                Hieda Ludugério de Souza, Lara Cruvinel Barbosa, Leonardo Bohner Hoffmann, Lícia Maria Costa Lima,
                Manuele de Alencar Amorim, Marcelo Oliveira Silva, Maria Perez Soares D’Alessandro, Mariana Aguiar
                Bezerra, Nathalia Maria Salione da Silva, Priscila Caccer Tomazelli, Sergio Seiki Anagusko e Sirlei Dal Moro.

                ## Equipe da Secretaria de Atenção Especializada em Saúde do Ministério da Saúde ##
                Nilton Pereira Junior, Mariana Borges Dias, Taís Milene Santos de Paiva, Cristiane Maria Reis Cristalda e
                Lorayne Andrade Batista.

                ## Equipe do CONASS ##
                René José Moreira dos Santos, Eliana Maria Ribeiro Dourado e Luciana Toledo.

                ##Equipe de apoio HSL ##
                Guilherme Fragoso de Mello, Juliana de Lima Gerarduzzi e Luiz Felipe Monteiro Correia."""
            
            }
        ]

### Template Base
prompt_template = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{sistema}

Think carefully about the above context. 

Now, review the user question:

{pergunta}

Provide an answer to this questions using only the above context. 

Awnser:"""

# Template base para cada exemplo
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{pergunta}"),
    ("system", "{sistema}"),
    ("assistant", "{resposta}")
])

# Integração dos exemplos
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
Você é um Chatbot que auxilia profissionais de saúde em cuidados paliativos com base apenas no Manual de Cuidados Paliativos, 2ª ed., São Paulo: Hospital Sírio-Libanês; Ministério da Saúde, 2023.
Responda apenas com informações documentadas no manual e, caso não saiba a resposta, pergunte se pode buscar em outras fontes.
Sempre forneça uma resposta completa e detalhada.
Estruture as respostas de forma clara, mencionando capítulos e subtítulos do manual quando relevante.
'''
        ),
        few_shot_prompt,
        ("human", "{pergunta}"),
    ]
)

file_path = "D:/5- Asimov acaemy/manual-cuidados-paliativos (1).pdf"

def importacao_documentos():
    documentos = []
    loader = PyPDFLoader(file_path)
    documentos = loader.load()
    
    return documentos



def split_de_documentos(documentos):
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["/n\n", "\n", ".", " ", ""]
    )
    documentos = recur_splitter.split_documents(documentos)

    for i, doc in enumerate(documentos):
        doc.metadata['source'] = doc.metadata['source'].split('/')[-1]
        doc.metadata['doc_id'] = i
    return documentos

def cria_vector_store(documentos):
    embedding_model = OllamaEmbeddings(model="llama3.2:latest") 
    vector_store = FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )
    return vector_store


documentos = importacao_documentos()
documentos = split_de_documentos(documentos)
vector_store = cria_vector_store(documentos)

retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 5, "fetch_k": 20},
    )

persist_directory = "./chroma_vectorstore"
embeddings = OllamaEmbeddings(model="llama3.2:latest") 
vectorstore = Chroma.from_documents(
            documents=documentos,
            embedding=embeddings,
            persist_directory=persist_directory,
            )
chain = final_prompt|llm |{"context": retriever , "pergunta": RunnablePassthrough()}

resposta = chain.invoke({"pergunta": "quem são os autores do capitulo de dor?"})
resposta_texto = resposta.content if isinstance(resposta, AIMessage) else str(resposta)
print(resposta_texto)

