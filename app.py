# Imports
import os
from langchain import hub
from pathlib import Path
import streamlit as st
from langchain_core.messages import AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
import torch

# Corrigir caminho do PDF
file_path = "D:/5- Asimov acaemy/manual-cuidados-paliativos (1).pdf"



pipe2 = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.2-3B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
  )


model_id = "meta-llama/Meta-Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

hf = HuggingFacePipeline.from_model_id(
    "text-generation",
    model_id="meta-llama/Meta-Llama-3.2-3B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf = HuggingFacePipeline(pipeline=pipe)

# Carregar documentos do PDF
#documents = load_and_display_pdf(file_path)
loader = PyPDFLoader(file_path)
documents = loader.load()

# Dividir o texto em pedaços menores
print("Dividindo o texto em pedaços menores para embeddings...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"Quantidade de textos divididos: {len(texts)}")
persist_directory = "./chroma_vectorstore"
embeddings = OllamaEmbeddings(model="Llama3.2-3B-Instruct") 
vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory,
            )

llm=ChatOllama(
    model="Llama3.2-3B-Instruct",
    temperature=0,
    # other params...
)
print("Configurando o modelo LLaMA 3.2 com Ollama...")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


print("Vector configurado:")        
retriever = vectorstore.as_retriever()
    
print("Retriever configurado:", retriever)

print("Construindo a cadeia de QA com LangChain...")
    # Carregar o vectorstore
    


### Exemplos de pergunta resposta
examples = [
    {
        "pergunta": "Como posso utilizar morfina para dor?",
        "resposta": """
Pergunta de acompanhamento necessária: Sim.
1. Pergunta de acompanhamento: Existe referência a algum capítulo no texto?
   Resposta intermediária: Sim.
2. Pergunta de acompanhamento: Qual capítulo é relevante?
   Resposta intermediária: Dor.
3. Pergunta de acompanhamento: Dentro do capítulo 'Dor', existe referência a algum subtítulo?
   Resposta intermediária: Sim.
4. Pergunta de acompanhamento: Qual subtítulo?
   Resposta intermediária: Morfina.

Resposta final:
Segundo o Manual de Cuidados Paliativos, 2ª ed.:
Para dor, a morfina pode ser utilizada da seguinte forma:
Morfina simples:
- Dose inicial: 5 mg a cada 4 horas (VO), com necessidade de avaliar doses mais baixas em pacientes idosos, com disfunção renal ou hepática;
- Dose máxima: Não possui dose teto; o limite é o efeito colateral, devendo ser titulado cuidadosamente;
- Frequência de administração: A cada 4 horas. Em casos específicos (idosos, disfunções), considerar a cada 6 horas;
- Vias de administração: Oral, sonda nasoenteral, gastrostomia, endovenosa, subcutânea, hipodermóclise;
- Equipotência: Morfina endovenosa é três vezes mais potente que a oral;
- Particularidades: Metabolizada no fígado e excretada pelo rim. Usar com cautela em pacientes com doença hepática ou renal;
- Disponibilidade no SUS: Constante na Rename 2022.
""",
    },
    {
        "pergunta": "Quais são os efeitos colaterais da morfina?",
        "resposta": """
Pergunta de acompanhamento necessária: Sim.
1. Existe um capítulo relacionado? Sim.
   Resposta intermediária: Dor.
2. Algum subtítulo é relevante? Sim.
   Resposta intermediária: Efeitos colaterais.

Resposta final:
Segundo o Manual de Cuidados Paliativos, 2ª ed.:
Os principais efeitos colaterais da morfina incluem náuseas, vômitos, constipação, sonolência e depressão respiratória. Esses efeitos devem ser monitorados e manejados adequadamente.
"""
    },
    {
        "pergunta": "Quem são os autores do Capitulo de Dor?",
        "resposta": """
Pergunta de acompanhamento necessária: Sim.
1. Existe um capítulo relacionado? Sim.
   Resposta intermediária: Dor.
2. Algum subtítulo é relevante? Não.
   Resposta intermediária: Efeitos colaterais.
3. Foi perguntado sobre nomes de pessoas? Sim.
   Resposta intermediária: Sim.
4. É sobre o livro ou capítulo?
   Resposta intermediária: capítulo.


Resposta final:
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
        "resposta": """
Pergunta de acompanhamento necessária: Sim.
1. Existe um capítulo relacionado? 
   Resposta intermediária: Não.
2. Algum subtítulo é relevante? 
   Resposta intermediária: Não.
3. Foi perguntado sobre nomes de pessoas? Sim.
   Resposta intermediária: Sim.
4. É sobre o livro ou capítulo?
   Resposta intermediária: livro.


Resposta final:
A equipe responsável pelo Manual de Cuidados Paliativos, 2ª ed., foram:

Editores
Maria Perez Soares D’Alessandro, Lara Cruvinel Barbosa, Sergio Seiki Anagusko, Ana Paula Mirarchi Vieira
Maiello, Catherine Moreira Conrado, Carina Tischler Pires e Daniel Neves Forte.

Autores
Aline de Almada Messias, Ana Cristina Pugliese de Castro, Ana Paula Mirarchi Vieira Maiello, Caroline
Freitas de Oliveira, Catherine Moreira Conrado, Daniel Felgueiras Rolo, Fábio Holanda Lacerda, Fernanda
Pimentel Coelho, Fernanda Spiel Tuoto, Graziela de Araújo Costa , Gustavo Cassefo, Heloisa Maragno,
Hieda Ludugério de Souza, Lara Cruvinel Barbosa, Leonardo Bohner Hoffmann, Lícia Maria Costa Lima,
Manuele de Alencar Amorim, Marcelo Oliveira Silva, Maria Perez Soares D’Alessandro, Mariana Aguiar
Bezerra, Nathalia Maria Salione da Silva, Priscila Caccer Tomazelli, Sergio Seiki Anagusko e Sirlei Dal Moro.

Equipe da Secretaria de Atenção Especializada em Saúde do Ministério da Saúde
Nilton Pereira Junior, Mariana Borges Dias, Taís Milene Santos de Paiva, Cristiane Maria Reis Cristalda e
Lorayne Andrade Batista.

Equipe do CONASS
René José Moreira dos Santos, Eliana Maria Ribeiro Dourado e Luciana Toledo.

Equipe de apoio HSL:
Guilherme Fragoso de Mello, Juliana de Lima Gerarduzzi e Luiz Felipe Monteiro Correia.
"""
    }
]


# Template para os exemplos
example_prompt = ChatPromptTemplate.from_messages([
                                                    ("human", '''Pergunta: {pergunta} 
                                                    Contexto: {context} 
                                                    Resposta:'''),
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
Estruture as respostas de forma clara, mencionando capítulos e subtítulos do manual quando relevante.
'''
        ),
        few_shot_prompt,
        ("human", "{pergunta}"),
    ]
)

    
print("Retriever configurado:", retriever)



qa_chain = (
    {"context": retriever | format_docs, "pergunta": RunnablePassthrough()}
    | final_prompt
    | llm
    | StrOutputParser()
    )
print("QA Chain configurada:", qa_chain)
    
resposta = qa_chain.invoke({"pergunta":"o que é cuidados paliativos?"})
print(resposta.content)



