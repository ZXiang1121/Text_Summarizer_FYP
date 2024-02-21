import streamlit as st
from st_pages import Page, show_pages, add_page_title

from hugchat import hugchat
from hugchat.login import Login
import pandas as pd
import base64
import time

import os
import shutil
import sqlalchemy as sa
import psycopg2


import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from  langchain import LLMChain, HuggingFacePipeline, PromptTemplate
import torch

from langchain.chat_models import ChatOpenAI


# Specify what pages should be shown in the sidebar, and what their titles 
# and icons should be
show_pages(
    [
        Page("streamlit_app.py", "Summarizer", "💬"),
        Page("pages/diarization.py", "Diarization", "🎤"),
        Page("pages/database.py", "Database", "🗄️"),
        Page("pages/rag.py", "RAG", "🗒️"),
    ]
)



# CSS
def local_css(file_name):
    # file_path = str(os.path.join(os.path.dirname(__file__), file_name))
    with open(file_name, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles/style.css")


# DATABASE 
@st.cache_resource
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()

# conn.autocommit = True
select_sql = "SELECT text, summary, model, elapsed_time FROM summary;"
insert_sql = """INSERT INTO summary (text, summary, model, elapsed_time) VALUES (%s,%s,%s,%s);"""


# INSERT DATA INTO TABLE
# @st.cache_data(ttl=600)
def insert_query(query, tuple):
    with conn.cursor() as cur:
        cur = conn.cursor()
        cur.execute("ROLLBACK")
        return cur.execute(query, tuple)

# run_query(f"INSERT INTO test (summary_id, text, summary, elapsed_time) VALUES (2, 'Hi its testings', 'testing', 100);")

# # App Logo
# def add_logo():
#     st.markdown(
#         """
#         <style>
#             [data-testid="stSidebarNav"] {
#                 background-image: url("images/nyc logo.png);
#                 background-repeat: no-repeat;
#                 padding-top: 120px;
#                 background-position: 20px 20px;
#             }
#             [data-testid="stSidebarNav"]::before {
#                 content: "NYC 💬 Summarizer";
#                 margin-left: 20px;
#                 margin-top: 20px;
#                 font-size: 30px;
#                 position: relative;
#                 top: 100px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
# add_logo()

# def get_table_download_link(messages):
#     df = pd.DataFrame(messages)
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#     return f'<a href="data:file/csv;base64,{b64}" download="chat_history.csv">Download here</a>'

@st.cache_resource
def convert_df(messages):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = pd.DataFrame(messages)
    return df.to_csv(index=False).encode('utf-8')


def load_llm(model):
    if model == "bart-large":
        model_name = "philschmid/bart-large-cnn-samsum"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, return_dict=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_length = tokenizer.model_max_length

        pipeline = transformers.pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=max_length,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
     
        llm = HuggingFacePipeline(pipeline = pipeline)

        return llm, max_length
    



# Function for generating LLM response
def generate_summary_response(model, prompt_input, prompt_template=None):
    start_time = time.time()
    if model == "bart-large":
        llm, max_token = load_llm(model=model)
            
        summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", token_max=max_token)
        
    elif model == "gpt-3.5-turbo":
        map_template = """
        The following is a set of documents
        {text}
        Based on this list of docs, generate summary for each speaker.
        Helpful Answer:
        """
        combine_template = f"""{prompt_template}"""

        map_prompt = PromptTemplate(template=map_template, input_variables=["text"])

        combine_prompt = PromptTemplate(template=combine_template, input_variables=["text"])

        model_name = "gpt-3.5-turbo-0125"
        llm = ChatOpenAI(temperature=temperature, model_name=model_name, openai_api_key=api_key)
        max_token = 16385

        summary_chain = load_summarize_chain (
            llm=llm,
            chain_type='map_reduce',
            map_prompt = map_prompt,
            combine_prompt=combine_prompt,
            verbose=False
        )
        
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=max_token-100, chunk_overlap=100)
    docs = text_splitter.create_documents([prompt_input])
    
    summary = summary_chain.run(docs)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # INSERT DATA INTO POSTGRESQL
    insert_query(insert_sql, (prompt_input, summary,model, elapsed_time))

    return summary

# Function for generating LLM response
def generate_hugchat_response(prompt_input, email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot                        
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)


# SIDEBAR

# Hugging Face Credentials
with st.sidebar:
    # st.image('nyc logo.png')

    st.title('NYC 💬 Summarizer')
    if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
        hf_email = st.secrets['EMAIL']
        hf_pass = st.secrets['PASS']
    else:
        hf_email = st.text_input('Enter E-mail:', type='password')
        hf_pass = st.text_input('Enter password:', type='password')
        if not (hf_email and hf_pass):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    # st.write("Fine-tuned LLM model for video conference/webinar transcripts summarization.")
    # st.write("Built with 🤗 Transformers, Streamlit, and AWS")
    # st.write("######")


    # st.markdown("---")

    st.write("Choose Summary Type")
    selected_summary = st.selectbox(
        "",
        ("Overall", "Multi-Speaker"),
        label_visibility="collapsed"
    )

    st.write("Model Details")

    if selected_summary == "Overall":
        selected_model = st.selectbox(
            "",
            ("bart-large", "gpt-3.5-turbo"),
            label_visibility="collapsed",
            placeholder="Choose a model",
            # index=None,
        )
    else:
        selected_model = st.selectbox(
            "",
            (["gpt-3.5-turbo"]),
            placeholder="Choose a model",
            label_visibility="collapsed",
            # index=None,
        )
         


    if selected_model == "gpt-3.5-turbo":
        api_key = st.text_input("Enter OpenAI API key", type="password")
        temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, step=0.1)
        st.caption("- Lower Temperature More Informative")
        st.caption("- Higher Temperature More Creative")
    elif selected_model == "palm-2":
        api_key = st.text_input("Enter GCP API key", type="password")
        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, step=0.1)
        st.caption("- Lower Temperature More Informative")
        st.caption("- Higher Temperature More Creative")

    st.markdown("---")




with st.sidebar:
    st.write("Summarize a file")
    uploaded = st.file_uploader("", type=["txt"], label_visibility="collapsed", key="upload_file_btn")
    if uploaded is not None:
        send_file_btn = st.button("send")

    st.markdown("---")

    # View Current DB button


# st.markdown('<style>section[data-testid="stSidebar"] div.stButton button {width: 200px;}</style>', unsafe_allow_html=True)

with st.sidebar:

    reset_button = st.button("Reset Chat",key="reset_button")
    if reset_button:
        del st.session_state.messages


    if "messages"  in st.session_state.keys():
    # st.session_state.messages = []
        # download_chat = st.download_button("Download chat history", data=st.session_state.messages)
        csv = convert_df(st.session_state.messages)

        download_chat = st.download_button(
            label="Download chat as CSV",
            data=csv,
            file_name='large_df.csv',
            mime='text/csv',
        )

    # download chat history
    # download_db = st.button("Download chat history")
    # if download_db:
    #     st.markdown(get_table_download_link(st.session_state.messages), unsafe_allow_html=True)
    # st.markdown("---")


# from streamlit_modal import Modal
#     modal = Modal(key="Demo Key",title="test")
#     for col in st.columns(8):
#         with col:
#             open_modal = st.button(label='button')
#             if open_modal:
#                 with modal.container():
#                     st.markdown('testtesttesttesttesttesttesttest')

    # st.sidebar.write("Prompt Formatting")
    # prompt_format = st.sidebar.text_area(
    #     label = "Prompt Formatting",
    # value ="It was the best of times, it was the worst of times, it was the age of "
    # )
    # st.sidebar.write(f'You wrote {len(prompt_format)} characters.')

    # from streamlit_modal_input import modal_input

    # prompt_engineer = modal_input(
    #      "hi",

    # )

    # st.sidebar.write(prompt_engineer)


# SUMMARIZER CONTENT

if selected_model == "bart-large" or selected_model == None:
    selected_prompt = None

else:
    if selected_summary == "Overall":

        prompt_options = [
    """Write a concise summary of the following: {text}
    """, 
"""Generate a summary of the following text that includes the following elements:

* A title that accurately reflects the content of the text.
* An introduction paragraph that provides an overview of the topic.
* Bullet points that list the key points of the text.
* A conclusion paragraph that summarizes the main points of the text.

Text:`{text}`
""",
    # """Transcript:
    # {text}

    # Given a transcripts between multiple speakers above, provide a concise summary within 200 words for each speaker highlighting their key points or contributions with the given format below:

    # Speaker 1 Name: (Summary for speaker 1)


    # Speaker 2 Name: (Summary for speaker 2)
    # """
    ]

    else:
        prompt_options = [
        """Transcript:
        {text}

        Given a transcripts between multiple speakers above, provide a concise summary within 200 words for each speaker highlighting their key learning points or contributions with the given format below:

        Also add a "Overall Summary" section to summarise the entire discussion.
        """,

        """Transcript:
{text}

Given a transcripts between multiple speakers above
- List out all Speaker Name In the discussion.
- Provide a concise summary within 200 words for each speaker highlighting their key learning points or contributions with the given format below:
- Add a "Overall Summary" section to summarise the entire discussion at the end.
""",

"""Transcript:
{text}

Given a transcripts between multiple speakers above, provide a concise summary within 200 words for each speaker highlighting their key points or contributions with the given format below:

Speaker 1 Name: (Summary for speaker 1)


Speaker 2 Name: (Summary for speaker 2)
""",
        ]

    st.subheader("Prompt")
    with st.expander("Prompt Engine"):
        selected_prompt = st.selectbox(
            label="Choose a Prompt",
            options=prompt_options,
            index=None,
            label_visibility="collapsed",
        )
        # selected_prompt = st.radio(key="radio_prompt",
        # label="Select a prompt:",
        # options=prompt_options,
        # captions = ["Laugh out loud.", "Get the popcorn.", "Never stop learning."])

        st.markdown("---")

        selected_prompt = st.text_area(key="text_area_prompt",
                label="Customize your prompt:",
                height = 300,
                # label_visibility="collapsed",
                placeholder="E.g. Write a concise summary of the following: {text}",
                value= selected_prompt
            )
        st.caption("""Remember to add "{text}" which will be where transcript will be filled.""")

        print(selected_prompt)

        st.markdown("---")



info = """
Here are some things you can ask me to do:\n
Tell me to summarize something:  
    \tsummarize: <enter text here>  
    \tsummarize: The quick brown fox jumped over the lazy dog.\n
You can also select a different model and upload a transcript as a file to summarize.
"""

def addUserPrompt(prompt):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

def addBotPrompt(response, help=False):
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        full_response = ""
        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        if help == True:
            st.info(info)
            st.session_state.messages.append({"role": "ai", "content": response + "<<<" + info + ">>>"})
        else:
            st.session_state.messages.append({"role": "ai", "content": response})
        message_placeholder.markdown(full_response)


st.subheader("Summarize 💬")
st.write("Ask me to summarize anything!")

# Store LLM generated responses

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Enter Transcript Below ..."}]




# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])



# User-provided prompt
if prompt := st.chat_input( placeholder="Enter your transcript here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

elif uploaded is not None:
    prompt = uploaded.read().decode("utf-8")
    if send_file_btn:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            if selected_model == None:
                response = generate_hugchat_response(prompt, hf_email, hf_pass)
            else:
                try:
                    response = generate_summary_response(model=selected_model, prompt_input=prompt, prompt_template=selected_prompt)
                except ValueError:
                    if api_key:
                        response = "Please enter a valid API KEY."
                        if prompt_options:
                            response = "Please select a prompt."
                    else:
                        response = "Please enter your API KEY."


            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

