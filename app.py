import gradio as gr
import numpy as np
import torch
import librosa
import time
import os
from dotenv import load_dotenv
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from chatbot import ChatbotOperations
from fastapi import FastAPI
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


global file_path
file_path = None

# Load environment variables from .env file
load_dotenv()
# Access your secrets
gemini_api_key = os.getenv('gemini_api_key')
pc_api = os.getenv('Pinecone_API')
bot = ChatbotOperations(gemini_api_key, pc_api)

# Audio transcription function
def get_transcription(audio, message):  
    time.sleep(5)
    if audio is None:
        return message
    sample_rate, audio_data = audio
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = None

    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

    input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    message['text'] += "\n" + transcription
    return message

def add_message(message, history):
    for x in message["files"]:
        global file_path 
        file_path = x
    if message["text"] is not None:
        history.append((message["text"], None))

    message['text'] = ''
    message['files'] = ''
    return message, history

def collect_and_process(history, audio, text_input): 
    pdf_added = False
    
    if file_path:
        pdf_added = True
        bot.process_pdf(file_path)  

    combined_query = history[-1][0] 
    response = bot.get_response(question=combined_query, pdf_added = pdf_added)
    
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history 

 
# Gradio app layout with chatbot, text input, audio, file upload, and clear button
with gr.Blocks() as app_layout:
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            # Header Section with custom CSS
            gr.HTML("""
            <style>
                #header {
                    background-color: #0d5a7a;
                    padding: 10px;
                    text-align: center;
                }
                #about {
                    background-color: #0d5a7a;
                    padding: 10px;
                }
                #chatbot {
                    background-color: #07656e;
                    padding: 10px;
                    border-radius: 5px;
                }
                #text_input {
                    padding: 5px;
                    border-radius: 5px;
                }

            </style>
            <div id="header">
                <h1 "style = text-align: center";>Clinical GPT</h1>
                <p>AI Powered Medical Assistant.</p>
            </div>
            <div id="about">  
                            <h2>About</h2>
                            <p>The Clinical-GPT can analyze medical reports, perfrom diagnosis and
                    suggest mediactions. Provides quick and accurate clinical recommendations tailored to your needs.
                     It processes inputs in various formats, including text, audio,
                and medical reports in pdf format.</p>
                <br><br><br><br>
                        </div>
        """)

        with gr.Column(scale=3, elem_id="chatbox"):
                chatbot = gr.Chatbot(label="Chat", height=400, show_label= False, avatar_images= (r'assets\images\chatbot.png',r'assets\images\user.jpg'))

    # Bottom Section
    with gr.Row():
        with gr.Column():
                text_input = gr.MultimodalTextbox(
                    file_types= ['.pdf'],
                    file_count= "single",
        interactive=True,
        placeholder="Enter message or upload file...",
        show_label=False,
    )


    with gr.Row():
        mic_button = gr.Audio(sources="microphone", type="numpy", label="Record Audio", elem_id="mic_button",
                              interactive=True, editable= False)


    text_input.submit(get_transcription, [mic_button,text_input], text_input, queue= False).then(add_message, [text_input, chatbot], [text_input, chatbot]).then(
        collect_and_process, [chatbot, mic_button, text_input], chatbot
    ).then(lambda: gr.MultimodalTextbox(interactive=True, value=None), None, [text_input])


app_layout.launch()