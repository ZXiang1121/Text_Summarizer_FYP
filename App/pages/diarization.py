import streamlit as st
import psycopg2

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import torch

from sklearn.cluster import AgglomerativeClustering

from simple_diarizer.diarizer import Diarizer




import whisper
import speech_recognition as sr

import math
import time
import datetime
import contextlib
import wave
import numpy as np
import pandas as pd
import soundfile as sf
import io
import os
import ffmpeg


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

def insert_query(query, tuple):
    with conn.cursor() as cur:
        cur = conn.cursor()
        cur.execute("ROLLBACK")
        return cur.execute(query, tuple)

conn = init_connection()

# conn.autocommit = True
insert_sql = """INSERT INTO diarization (model, audio_file, transcript, elapsed_time, num_speaker) VALUES (%s,%s,%s,%s,%s);"""


# def rttm_to_dataframe(rttm_file_path):
#   columns = ["Type", "File ID", "Channel", "Start Time", "Duration", "Orthography", "Confidence", "Speaker", "x", "y"]

#   with open(rttm_file_path, "r") as rttm_file:
#     lines = rttm_file.readlines()

#   data = []

#   for line in lines:
#     line = line.strip().split()
#     data.append(line)

#   df = pd.DataFrame(data, columns=columns)
#   df = df.drop(["x", "y", "Orthography", "Confidence"], axis=1)
#   return df



def extract_text_from_audio(audio_file, start_time, end_time):

    r = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        # audio = r.listen(source, duration=end_time, offset=start_time, timeout=2)
        audio = r.record(source, duration=end_time, offset=start_time)

    # text = r.recognize_google_cloud(audio_data=audio, credentials_json="./assets/nyctextsummarizer-9ebeb27831a8.json", language="en-US")
    text = r.recognize_whisper(audio)

    return text

@st.cache_resource
def load_pyannote():
    audio_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    # Access Token from HuggingFace
    use_auth_token=st.secrets['diarization_access_key']
)
    return audio_pipeline

audio_pipeline = load_pyannote()


# @st.cache_resource
# def load_whisper_model(size):
#     model = whisper.load_model(size)
#     return model

# @st.cache_resource
# def speaker_embedding():
#     embedding_model = PretrainedSpeakerEmbedding(
#         "speechbrain/spkrec-ecapa-voxceleb",
#         device=torch.device("cuda"))    
#     return embedding_model


# whisper_model = load_whisper_model("large")
# audio = Audio()
# embedding_model = speaker_embedding()

# def segment_embedding(segment):
#     start = segment["start"]
#     # Whisper overshoots the end timestamp in the last segment
#     end = min(duration, segment["end"])
#     clip = Segment(start, end)
#     waveform, sample_rate = audio.crop(path, clip)

#     # Convert waveform to single channel
#     waveform = waveform.mean(dim=0, keepdim=True)

#     return embedding_model(waveform.unsqueeze(0))

def time_taken(secs):
  return datetime.timedelta(seconds=round(secs))

def generate_transcript(audio_file, min_speakers):

    if selected_model == "pyannote":
        diarization = audio_pipeline(audio_file, min_speakers=min_speakers)
        transcripts = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_start = turn.start
            speaker_end = turn.end
            
            # st.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            try:
                transcription = extract_text_from_audio(audio_file, speaker_start, speaker_end)
            except:
                transcription = "Not Found"
            finally:
                transcripts.append(f"{speaker} [{speaker_start:.1f}s - {speaker_end:.1f}s]: {transcription}")
                st.write(f"{speaker} [{speaker_start:.1f}s - {speaker_end:.1f}s]: {transcription}")
                st.markdown("---")

                # transcripts.append(st.write(f"{speaker} [{speaker_start:.1f}s - {speaker_end:.1f}s]: \n{transcription}"))
        full_transcript = '\n'.join(transcripts)
    
    # elif selected_model == "clustering":
    #     st.audio(audio_file, format='audio/wav')
    #     st.write(os.path.abspath(audio_file))
    #     result = whisper_model.transcribe(audio_file)
    #     ('stop')
    #     segments = result["segments"]

    #     embeddings = np.zeros(shape=(len(segments), 192))
    #     for i, segment in enumerate(segments):
    #         embeddings[i] = segment_embedding(segment)
        
    #     embeddings = np.nan_to_num(embeddings)

    #     clustering = AgglomerativeClustering(min_speakers).fit(embeddings)
    #     labels = clustering.labels_
    #     for i in range(len(segments)):
    #         segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    #     transcript = ""
    #     for (i, segment) in enumerate(segments):
    #         if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
    #             transcript += "\n" + segment["speaker"] + ' ' + str(time_taken(segment["start"])) + '\n'
    #         transcript += segment["text"][1:] + ' '
    #     (transcript)

    # elif selected_model == "simple_diarizer":
    #     diarization = Diarizer(embed_model='xvec', cluster_method='sc')
    #     segments = diarization.diarize(audio_file, num_speakers=min_speakers)
    #     signal, fs = sf.read(audio_file)
    #     (signal, fs)
    #     # segments = diarization.diarize("${AUDIO_FILE_PATH}", threshold=THRESHOLD)



    return full_transcript



# SIDEBAR
with st.sidebar:

    st.write("Model Details")

    selected_model = st.selectbox(
            "model_selected",
            # , "clustering","simple_diarizer"
            (["pyannote"]),
            label_visibility="collapsed",
            placeholder="Choose a model",
            key="diarization_model"
            # index=None,
        )
    
    # Number of Speaker
    st.write("Number of Speaker")

    num_speaker = st.number_input(
                label=" ",
                min_value=2,
                step=1, 
                placeholder="Type a number...", 
                label_visibility="collapsed"
            )

    
    # Upload File
    st.markdown("---")
    st.write("Audio File")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"], label_visibility="collapsed", key="upload_file_btn")
        
    if uploaded_file is not None:
        with contextlib.closing(wave.open(uploaded_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

            seconds = round(duration)
            minutes = math.floor(seconds / 60)
            hours = math.floor(seconds / 3600)

            if seconds < 60:
                st.caption(f"Duration: {seconds} seconds")
            elif seconds < 3600:
                st.caption(f"Duration: {minutes} minutes, {seconds % 60} seconds")
            else:
                st.caption(f"Duration: {hours} hours, {minutes % 60} minutes, {seconds % 60} seconds")
        send_file_btn = st.button("send")

                      
    st.markdown("---")



# MAIN BODY
st.subheader("Diarization 🎤")
st.write("Ask me transcript multi-speaker audio for you!")

from pathlib import Path
if uploaded_file is not None:

    if send_file_btn:
        audio = st.audio(uploaded_file, format='audio/wav')
        print(type(uploaded_file))
        audio_folder_path = "./assets/audio/"
        save_path = audio_folder_path + uploaded_file.name

        if uploaded_file.name.endswith('wav'):
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            # audio = pydub.AudioSegment.from_wav(uploaded_file)
            # file_type = 'wav'
        elif uploaded_file.name.endswith('mp3'):
            with open(save_path, "wb") as f:
    
                f.write(uploaded_file.getvalue())
        # st.write(save_path)
        # transcript = model.transcribe()
        # st.write(os.path.exists(save_path))
        st.success("Transcribing Audio ... ...")
        # os.path.abspath(save_path)

        start_time = time.time()
        transcript = generate_transcript(save_path, num_speaker)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        insert_query(insert_sql, (selected_model, uploaded_file.name,transcript, elapsed_time, num_speaker))

        st.success("Transcription Completed")


            # print(audio)
            # # print(type(audio))

            # audio_src = audio._st_level[0].data.url
            # print(audio_src)
            # uploaded_file = "./new.mp3"
            # temp = uploaded_file.getvalue()
            # temp.seek(0)
            # with NamedTemporaryFile(suffix="mp3") as temp:
            #     temp.write(uploaded_file.getvalue())
            #     temp.seek(0)
            
        # transcript = generate_transcript(save_path, num_speaker)
        # st.write(transcript)
        # st.write('hi')
        # uploaded_file = open(uploaded_file, 'rb')
        # audio_bytes = uploaded_file.read()
        # audio = st.audio(audio_bytes, format='audio/wav')
        # sample_rate = 44100  # 44100 samples per second
        # seconds = seconds  # Note duration of 2 seconds

        # frequency_la = 10000
        # t = np.linspace(0, seconds, seconds * sample_rate, False)

        # # Generate a 440 Hz sine wave
        # note_la = np.sin(frequency_la * t * 2 * np.pi)
        # st.audio(note_la, format='audio/wav', sample_rate=sample_rate)

        # transcript = generate_transcript(audio, num_speaker)
    # if send_file_btn is not None:
    #     st.write(transcript)