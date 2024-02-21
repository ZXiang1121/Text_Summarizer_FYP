import streamlit as st
import pandas as pd
import psycopg2

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

def select_query(query):
    with conn.cursor() as cur:
        cur.execute("ROLLBACK")
        cur.execute(query)
        return cur.fetchall()




# Retrieve Summary DT
select_summary_sql = "SELECT text, summary, model, elapsed_time FROM summary;"
rows = select_query(select_summary_sql)

summary_dt=pd.DataFrame(rows)
summary_dt.columns=['text','summary','model','elapsed_time']

# Retrieve Diarization DT
select_diarization_sql = "SELECT model, audio_file, num_speaker, transcript, elapsed_time FROM diarization;"
rows = select_query(select_diarization_sql)

diarization_dt=pd.DataFrame(rows)
diarization_dt.columns=['model','audio_file', 'num_speaker' ,'transcript','elapsed_time']


# # Retrieve RAG DT
# select_rag_sql = "SELECT chat_model, embedding_model, document, question, answer, elapsed_time FROM rag;"
# rows = select_query(select_rag_sql)

# rag_dt=pd.DataFrame(rows)
# rag_dt.columns=['chat_model','embedding_model', 'document' ,'question','answer', 'elapsed_time']


st.subheader("PostgreSQL Database 🗄️")

st.text("\n")

st.subheader("Summary")
st.dataframe(summary_dt)


st.subheader("Diarization")
st.dataframe(diarization_dt)

# st.subheader("RAG")
# st.dataframe(rag_dt)


