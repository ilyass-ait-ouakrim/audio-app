import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.title("Audio Transcription & Analysis Tool")
st.write("Upload an audio file to extract text, summarize, and get key points.")


audio_file = st.file_uploader("Upload Audio File")

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("Process Audio"):
        with st.spinner("Transcribing and Analyzing in the cloud..."):
            try:
                transcription = client.audio.transcriptions.create(
                  file=(audio_file.name, audio_file.getvalue()),
                  model="whisper-large-v3-turbo",
                  response_format="json"
                )
                transcript_text = transcription.text
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant. Summarize the provided text briefly and list the most important points as bullet points."
                        },
                        {
                            "role": "user",
                            "content": f"Please analyze this text:\n\n{transcript_text}"
                        }
                    ],
                    model="openai/gpt-oss-20b",
                )
                analysis_result = chat_completion.choices[0].message.content
                
                st.subheader("Summary & Important Points:")
                st.write(analysis_result)
                
                st.subheader("Full Transcription:")
                st.write(transcript_text)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
