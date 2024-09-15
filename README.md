# ClinicalGPT - A Medical Assistant Chatbot

ClinicalGPT is an advanced AI-powered medical assistant chatbot designed to analyze medical documents, diagnose diseases based on patient symptoms, and provide medical advice. It leverages state-of-the-art language models and vector databases to process text, audio, and PDF inputs, making it a versatile tool for clinical applications.

## Features
- *Medical Document Analysis*: Upload medical documents (text, audio, or PDF) for analysis.
- *Symptom-based Diagnosis*: Provide patient symptoms, and ClinicalGPT offers potential diagnoses and medical advice.
- *Multi-Input Support*: Accepts text, audio, and PDF documents as input.
- *Advanced Embedding & Search*: Uses ClinicalBERT to compute embeddings and PineCon as a vector database for efficient search and retrieval.
- *Powerful Language Model*: Powered by GeminiPro, a sophisticated large language model (LLM) designed for medical applications.
- *User-friendly Interface*: Built with Gradio for an easy-to-use, web-based user interface.

## Technology Stack
- *Embeddings*: [ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT)
- *Audio Transcrption*: [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
- *Vector Database*: [PineCone](https://www.pinecone.io/)
- *Large Language Model*: [GeminiPro](https://gemini.google.com/app)
- *UI Framework*: [Gradio](https://gradio.app)

## Installation

### Prerequisites
- Python 3.8+
- Ensure you have pip installed on your system.

### Steps
1. *Clone the repository:*
   bash
   git clone https://github.com/your-username/ClinicalGPT.git
   cd ClinicalGPT
   

2. *Install the required dependencies:*
   bash
   pip install -r requirements.txt
   

3. *Create an environment file:*
   You will need to set up a .env file in the root of the project. The file should contain the necessary environment variables such as API keys and other credentials. You can follow the template provided in the repository (.env.example).

4. *Run the application:*
   bash
   python app.py
   

   This will launch the Gradio-based UI where you can interact with the chatbot.

## Usage
1. Open the Gradio web interface.
2. Choose the input format: text, audio, or PDF.
3. Upload your medical document or describe the patient's symptoms in the text input box.
4. Click on "Submit" to receive a diagnosis or medical advice based on the input provided.

## Environment Setup
Make sure to configure your environment properly using a .env file. This file should include the necessary API keys for:
- *PineCone*: To manage the vector database.
- *GeminiPro*: To access the large language model for generating responses.

Refer to the .env.example for guidance on setting up these variables.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
