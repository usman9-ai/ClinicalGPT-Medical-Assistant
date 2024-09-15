import warnings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from transformers import BertModel, BertTokenizer
import torch
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings 
import PyPDF2
warnings.filterwarnings("ignore")
import pinecone


class ClinicalBERTEmbeddings(Embeddings):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.model = BertModel.from_pretrained("medicalai/ClinicalBERT")

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token is at position 0
                embeddings.append(cls_embedding.cpu().numpy())  # Append as ndarray
        # Convert embeddings to list of lists
        return [embedding.flatten().tolist() for embedding in embeddings]

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = self.model(**inputs)
            return self.mean_pooling(outputs, inputs['attention_mask']).cpu().numpy().tolist()  # Convert to list

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class ChatbotOperations(ClinicalBERTEmbeddings):
    def __init__(self, gemini_api, pinecone_api):
        super().__init__()
        self.gemini_api_key = gemini_api
        self.pc_api = pinecone_api
        self.pc = pinecone.Pinecone(api_key=self.pc_api)
        self.index_name = "clinical-bert-index"
        self.index = self.connect_to_index()
        self.empty_database()


    def empty_database(self):
        if self.index_name in self.pc.list_indexes():
            # Delete the entire index
            self.pc.delete_index(self.index_name)
            self.pc.create_index(
                        name= self.index_name,
                        dimension=768,  
                        metric='cosine',  
                        spec=pinecone.ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'  
                        )
                    ) 
        else:
            return
        

    def get_response(self, question=None, pdf_added = False):
        result = ''
        sys = """You are now acting as a qualified medical doctor working as a medical assistant, 
  use the following pieces of context to answer the question at the end. 
  The question may be related to getting diagnosis of the disease based on syptoms, in that case you have to identify 
  the disease with high match with the sysmtoms and recommend user to get tested for that disease. 
  If the user asks about the pros and cons of any medication for any particuar disease, give him precise answer and 
  tell user to consult with the doctor if he wants a more professional advice.
  If the user greets or ask you about your self, greet back and tell them properly that you are a medical assistant and 
  tell  them the way you will help them.
  If you want any further information, you can ask the user. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.\n
  {context}
        Question: {question}
        Doctor:
        """


        sys_2 = """You are now acting as a qualified medical doctor working as a medical assistant, 
  use the following pieces of context to answer the question at the end. 
  The question may be related to getting diagnosis of the disease based on syptoms, in that case you have to identify 
  the disease with high match with the sysmtoms and recommend user to get tested for that disease. 
  If the user asks about the pros and cons of any medication for any particuar disease, give him precise answer and 
  tell user to consult with the doctor if he wants a more professional advice.
  If the user greets or ask you about your self, greet back and tell them properly that you are a medical assistant and 
  tell  them the way you will help them.
  If you want any further information, you can ask the user. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.\n
  Question: {question}
        Doctor:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key= self.gemini_api_key,
                                    temperature=0.2, convert_system_message_to_human=True)
        if pdf_added:
            # Query with Pinecone
            query_vector = self.embed_query(question)
            results = self.index.query(vector=query_vector, top_k=3, namespace="medical_reports",include_metadata=True)
            
            # Construct context from results
            context = ""
            if 'matches' in results:
                for match in results['matches']:
                    if 'metadata' in match:
                        text = match['metadata'].get('text', '')
                        if text:
                            context += text + " "
            
            # Generate response using Gemini
            QA_CHAIN_PROMPT = PromptTemplate.from_template(sys)
            qa_chain = LLMChain(
                llm=model,
                prompt=QA_CHAIN_PROMPT
            )
            result = qa_chain({"question": question, "context": context})
            result = result['text']
        
        else:
            qa_chain = LLMChain(
                llm=model,
                prompt=PromptTemplate.from_template(sys_2)
            )
            result = qa_chain({"question": question})
            result = result['text']
        return result
        
    def upsert_embeddings(self,data):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            texts = text_splitter.split_text(data)
            vectors = self.embed_documents(texts)
            upsert_data = [(str(i), vec, {"text": text}) for i, (vec, text) in enumerate(zip(vectors, texts))]
            self.index.upsert(vectors=upsert_data, namespace="medical_reports")
            
    def connect_to_index(self):
        try:
            if self.index_name not in self.pc.list_indexes():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  
                    metric='cosine', 
                    spec=pinecone.ServerlessSpec(
                        cloud='aws',
                        region='us-east-1' 
                    )
                )
        except pinecone.exceptions.PineconeException as e:
            if 'ALREADY_EXISTS' in str(e):
                pass
                #print(f"Index '{self.index_name}' already exists. Using the existing index.")
            else:
                raise
        # Connect to the index
        index = self.pc.Index(self.index_name)
        return index

    def process_pdf(self, file_path):
        try:
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                pages = pdf_reader.pages
                for page in pages:
                    text = page.extract_text()
                    self.upsert_embeddings(text)
        except Exception as e:
            return f"Error: {str(e)}"
