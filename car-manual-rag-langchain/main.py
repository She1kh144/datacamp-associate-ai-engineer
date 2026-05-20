# Import the required packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os

# Load environment variables from .env file
load_dotenv()

# Load the models required to complete the exercise (api_key is already set up)
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the HTML as a LangChain document loader
loader = UnstructuredHTMLLoader(file_path="data/mg-zs-warning-messages.html")
car_docs = loader.load()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Split car_docs
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
chunks = splitter.split_documents(car_docs)

# Create index if it doesn't exist
index_name = "car-manual-rag"
if not pc.has_index(name=index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,  # text-embedding-3-small dimension
        metric="cosine",  # by default           
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Embed and ingest documents to the index once
# Then you can just use .from_existing_index(index_name, embeddings)
"""
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=index_name
)
"""
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

# Create a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

system_prompt = (
    "You are a helpful assistant for MG ZS car owners. "
    "Answer questions about dashboard warning lights and what to do. "
    "Use only the provided context. Be clear, concise, and actionable. "
    "If the answer is not in the context, say 'I don't have information about that in the manual.'\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

query = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"
response = rag_chain.invoke({"input": query})
answer = response["answer"]
print(answer)