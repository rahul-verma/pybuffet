from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

class Retriever:
    def __init__(self, file_path):
        loader = PyPDFLoader(file_path)    
        pages = loader.load_and_split()
        db = Chroma.from_documents(pages, OpenAIEmbeddings())
        self.__retriever = db.as_retriever()
        
    def retrieve(self, query):
        chunks = self.__retriever.invoke(query)
        return '--\n'.join([chunk.page_content for chunk in chunks])
