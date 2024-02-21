import streamlit as st

import os, shutil ,tempfile, time
import pandas as pd
import pinecone
from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from chromadb.errors import InvalidDimensionException


TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')
st.set_page_config(page_title="RAG")
# st.session_state["rag"] = {}
# st.session_state.rag = {}

def local_css(file_name):
    # file_path = str(os.path.join(os.path.dirname(__file__), file_name))
    with open(file_name, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles/style.css")


# # DATABASE 
# @st.cache_resource
# def init_connection():
#     return psycopg2.connect(**st.secrets["postgres"])

# def insert_query(query, tuple):
#     with conn.cursor() as cur:
#         cur = conn.cursor()
#         cur.execute("ROLLBACK")
#         return cur.execute(query, tuple)

# conn = init_connection()

# # conn.autocommit = True
# insert_sql = """INSERT INTO rag (chat_model, embedding_model, document, question, answer, elapsed_time) VALUES (%s,%s,%s,%s,%s, %s);"""



@st.cache_resource
def convert_df(messages):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = pd.DataFrame(messages)
    return df.to_csv(index=False).encode('utf-8')



def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.txt')
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size):
    # text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    # docs = text_splitter.split_text(documents)
    # st.write(docs)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=100)
    docs = text_splitter.create_documents([documents])
    # num_chunks = len(texts)
    return docs

# st.write(LOCAL_VECTOR_STORE_DIR.as_posix())
def embeddings_on_local_vectordb(texts):
    Chroma().delete_collection()
    # default: text-embedding-ada-002
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # vectordb = Chroma.from_documents(texts, embedding=embeddings,
    #                                  persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    # vectordb.persist()

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    try:
        vector = Chroma.from_documents(documents=texts, embedding=embeddings)
    except InvalidDimensionException:
        # Clear data in Chroma DB to avoid dimension matching error
        Chroma().delete_collection()
        vector = Chroma.from_documents(documents=texts, embedding=embeddings)                 
    vector.persist()  

    retriever = vector.as_retriever(search_kwargs={'k': len(texts)})
    return retriever

def embeddings_on_pinecone(texts):
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=pinecone_index)
    retriever = vectordb.as_retriever()
    return retriever

def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(model=selected_chat_model, openai_api_key=openai_api_key),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.texts})
    result = result['answer']

    st.session_state.texts.append((query, result))
    return result


def process_documents(chunk_size):
    if not openai_api_key:
        st.warning(f"Please enter openai api key.")
    
    elif pinecone_db:
        if not pinecone_api_key or not pinecone_env or not pinecone_index or not uploaded_files:
            st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in uploaded_files:

                # (TMP_DIR)
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.txt') as tmp_file:
                    tmp_file.write(source_doc.read())
                
                documents = load_documents()

                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                # (type(documents))split_documents

                texts = split_documents(documents[0].page_content, chunk_size)
                # (texts)
                # #
                
                if not pinecone_db:
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                else:
                    st.session_state.retriever = embeddings_on_pinecone(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")


def read_file(file):
    with tempfile.NamedTemporaryFile(mode="wb") as temp:
        with st.spinner('This may take a while. Wait for it...'):
            bytes_data = file.getvalue()
            
    return bytes_data


def run():
    
    # input_fields()
    # if "texts" not in st.session_state:
    #     st.session_state.texts = []
    
    for text in st.session_state.texts:
        st.chat_message('user').write(text[0])
        st.chat_message('assistant').write(text[1])   
        # with st.chat_message(text["role"]):
        #     st.write(text["content"])

    if query := st.chat_input(placeholder="Enter your question"):
        st.chat_message("user").write(query)
        # st.session_state.texts.append({"role": "user", "content": query})
        # with st.chat_message("user"):
        #     st.write(query)


        # if st.session_state.texts[-1]["role"] != "assistant":
        #     with st.chat_message("assistant"):
        #         with st.spinner("Generating..."):
            
    # with st.spinner("Generating..."):
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("assistant").write(response)


        
    # st.session_state.texts.append(response)
    # st.session_state.texts.append({"role": "assistant", "content": response})





# Main Content
st.title("Retrieval Augmented Generation")

#
pinecone_db = st.toggle('Use Pinecone Vector DB')

# def input_fields():
#     #
with st.sidebar:
    #
    if "openai_api_key" in st.secrets:
        openai_api_key = st.secrets.openai_api_key
    else:
        openai_api_key = st.text_input("OpenAI API key", type="password")
        # os.environ.get("OPEN_AI_KEY")
    
    selected_chat_model = st.selectbox(
            "",
            ("gpt-3.5-turbo-0125", "gpt-4"),
            label_visibility="collapsed",
            placeholder="Choose a model",
            # index=None,
        )
    
    selected_embedding_model = st.selectbox(
            "",
            (["text-embedding-ada-002"]),
            label_visibility="collapsed",
            placeholder="Choose a model",
            # index=None,
        )
    if pinecone_db:
        #
        if "pinecone_api_key" in st.secrets:
            pinecone_api_key = st.secrets.pinecone_api_key
        else: 
            pinecone_api_key = st.text_input("Pinecone API key", type="password")
        #
        if "pinecone_env" in st.secrets:
            pinecone_env = st.secrets.pinecone_env
        else:
            pinecone_env = st.text_input("Pinecone environment")
        #
        if "pinecone_index" in st.secrets:
            pinecone_index = st.secrets.pinecone_index
        else:
            pinecone_index = st.text_input("Pinecone index name")

    # testing = load_documents()
    # (testing)

    uploaded_files = st.file_uploader(label="Upload Documents", type=["txt"], accept_multiple_files=True)

    # if uploaded_files is None:
    #     st.write("hi")
            # (bytes_data)
    chunk_size_selected = st.selectbox(
        "Chunk Size",
        (512, 1024, 16385),
        # label_visibility="collapsed"
    )

    # top_k_selected = st.number_input(
    #         label="Top K",
    #         min_value=1,
    #         step=1, 
    #         placeholder="Type a number...", 
    #         # label_visibility="collapsed"
    #     )
        
    
    # texts, num_chunks = split_documents(load_documents(), chunk_size_selected)

    # text = []

    # for file in uploaded_files:
    #     # st.write(uploaded_files)
    #     text.append(file)
    # join_string = "/n".join(text)

    if uploaded_files != []:
        
        content_list = []
        for file in uploaded_files:
            file_content = str(file.getvalue())
            
            content_list.append(str(file.getvalue()))
            
        full_content = "/n".join(content_list)
        word_count = len(full_content.split())
        chunk_count = len(split_documents(full_content, chunk_size_selected))

        st.write("Word Count: " + str(word_count))

        st.write("Chunks: " + str(chunk_count))
            
            # docs, num_chunks = split_documents(uploaded_files.getvalue(), chunk_size_selected)
            # st.write("Chunks: " + str(num_chunks))
        # def start_capture():
        #     st.write("# CLICKED")   

        # st.button("Submit Documents", on_click=start_capture)

        submit_document = st.button("Submit Documents", on_click=lambda: process_documents(chunk_size=chunk_size_selected), key="process_btn")
    
    st.markdown("---")
    if "texts" not in st.session_state:
        st.session_state.texts = []


    reset_button = st.button("Reset Chat",key="reset_button")
    if reset_button:
        del st.session_state.texts
        st.session_state.texts = []


    # if "messages"  in st.session_state.keys():
    # # st.session_state.messages = []
    #     # download_chat = st.download_button("Download chat history", data=st.session_state.messages)
    #     csv = convert_df(st.session_state.texts)

    #     download_chat = st.download_button(
    #         label="Download chat as CSV",
    #         data=csv,
    #         file_name='large_df.csv',
    #         mime='text/csv',
    #     )

# if uploaded_files != [] and openai_api_key and submit_document:
#     run()
    
    # st.write("Chunk Size")


#

#

if __name__ == '__main__':

    run()
