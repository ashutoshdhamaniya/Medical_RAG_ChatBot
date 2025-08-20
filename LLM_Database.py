from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "../data_files/"

def load_all_pdf_files(data):
    """
    Loads all PDF files from the given directory path.
    Args:
        data (str): Path to the directory containing PDF files.
    Returns:
        list: A list of document objects loaded from all PDF files.
    """

    # Initialize a DirectoryLoader to scan the given directory (`data`)
    # `glob='*.pdf'` ensures only files with a `.pdf` extension are matched.
    # `loader_cls=PyPDFLoader` specifies that each PDF will be loaded using the PyPDFLoader class.
    loader = DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )

    # Load all matching PDF files into a list of documents
    documents = loader.load()

    # Return the list of loaded documents
    return documents


def create_text_chunks(document_data):
    """
    Splits the given document data into smaller text chunks for easier processing.
    Args:
        document_data (list): A list of document objects (e.g., from PDFs).
    Returns:
        list: A list of smaller text chunks obtained by splitting the documents.
    """

    # Initialize a RecursiveCharacterTextSplitter
    # - chunk_size=500: Each text chunk will contain up to 500 characters.
    # - chunk_overlap=50: Each chunk will overlap the next one by 50 characters,
    #   which helps retain context across chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Split the input documents into smaller chunks using the defined splitter
    text_chunks = text_splitter.split_documents(document_data)

    # Return the resulting list of text chunks
    return text_chunks

def get_embedding_model_object():
    """
    Creates and returns a HuggingFace sentence embedding model.

    Key details about this model:
    - Embedding size: 384 dimensions (each text input is mapped to a 
      384-dimensional vector).
    - Usage: Well-suited for tasks like semantic search, clustering, 
      and Retrieval-Augmented Generation (RAG).

    Returns:
        HuggingFaceEmbeddings: An embedding model object that can be used 
        to convert text into numerical vector representations.
    """

    # Initialize the HuggingFaceEmbeddings object
    # 'all-MiniLM-L6-v2' maps each text input into a 384-dimensional vector.
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    # Return the embedding model instance
    return embedding_model



documents = load_all_pdf_files(DATA_PATH)
print(f"Length of the documments: {len(documents), type(documents)}")

chunked_text = create_text_chunks(documents)
print(f"Length of the documments: {len(chunked_text), type(chunked_text)}")
print(chunked_text[0])

embedding_model_object = get_embedding_model_object()