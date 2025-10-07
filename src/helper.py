from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


#Extract Data From the PDF File
def load_pdf_files(data):
    """Try to load PDF files from `data` folder. Prefers PyMuPDFLoader, falls back to PyPDFLoader.
    Returns a list of langchain Documents if successful, otherwise raises an informative exception.
    """
    loader_cls = None
    # Prefer PyMuPDFLoader if available
    try:
        from langchain.document_loaders import PyMuPDFLoader
        loader_cls = PyMuPDFLoader
        print('Using PyMuPDFLoader')
    except Exception:
        try:
            from langchain.document_loaders import PyPDFLoader
            loader_cls = PyPDFLoader
            print('PyMuPDFLoader unavailable — using PyPDFLoader')
        except Exception:
            loader_cls = None
    if loader_cls is None:
        raise RuntimeError('No suitable PDF loader found. Install pymupdf or pypdf and ensure langchain is updated.')
    from langchain.document_loaders import DirectoryLoader
    try:
        loader = DirectoryLoader(data, glob='*.pdf', loader_cls=loader_cls)
        documents = loader.load()
        print(f'Loaded {len(documents)} documents from {data}')
        return documents
    except Exception as e:
        # Provide helpful debugging info
        print('Failed to load PDFs using', getattr(loader_cls, '__name__', loader_cls))
        raise



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings