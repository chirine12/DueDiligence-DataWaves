import os
import tempfile
import fitz
import re
import gc
from pathlib import Path
from openai import OpenAI

import numpy as np
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# Simplified embeddings class that can be replaced with actual embedding model later
class SimpleEmbeddings:
    def __init__(self):
        pass
        
    def embed_documents(self, texts):
        # Return random vectors for now - will be replaced with actual embeddings in production
        return [np.random.rand(384) for _ in texts]

    def embed_query(self, text):
        # Return random vector for now - will be replaced with actual embedding in production
        return np.random.rand(384)


# Simplified FAISS store
class SimpleFAISS:
    def __init__(self, embeddings, documents, doc_embeddings=None):
        self.documents = documents
        self.embeddings = embeddings
        
    @classmethod
    def from_documents(cls, documents, embedding):
        embeddings = embedding.embed_documents([doc.page_content for doc in documents])
        return cls(embedding, documents, embeddings)
        
    def similarity_search(self, query, k=3):
        # Simple implementation that returns first k documents 
        # This is just a placeholder until FAISS is properly configured
        return self.documents[:min(k, len(self.documents))]
    
    def save_local(self, folder_path, index_name="faiss_index"):
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, f"{index_name}.pkl"), "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load_local(cls, folder_path, index_name="faiss_index"):
        with open(os.path.join(folder_path, f"{index_name}.pkl"), "rb") as f:
            return pickle.load(f)


# PDF processing functions
def extract_sections_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    current_section = {"title": "Introduction", "content": []}
    
    for page in doc:
        text = page.get_text()
        
        # Very simple section detection based on line breaks and text patterns
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Simple heuristic to detect headers (all caps, short, ends with colon)
            if (line.isupper() and len(line) < 50) or line.endswith(':'):
                # Save previous section if it has content
                if current_section["content"]:
                    sections.append((
                        current_section["title"], 
                        "\n".join(current_section["content"])
                    ))
                    
                # Start new section
                current_section = {"title": line.rstrip(':'), "content": []}
            else:
                current_section["content"].append(line)
                
    # Add final section
    if current_section["content"]:
        sections.append((
            current_section["title"], 
            "\n".join(current_section["content"])
        ))
        
    return sections


# Question generation function
def gen_questions(passage, api_key, k=5):
    if not api_key:
        return []
    
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
    
    client = OpenAI()
    prompt = ("Act like a senior compliance officer, understand deeply this document "
              "and generate the questions that I need to ask to generate a due diligence report:\n\n"
              f"{passage}\n\nPlease provide {k} critical questions, one per line.")

    try:
        rsp = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300, temperature=0.7
        )
        raw = rsp.choices[0].message.content
        qs = [re.sub(r"^[\-\d\.\)\s]*","",l).strip()
               for l in raw.splitlines() if l.strip()]
        return [q.rstrip(".")+("?" if not q.endswith("?") else "") for q in qs][:k]
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


# Categorize questions
def bucket(qs, api_key):
    if not qs or not api_key:
        return {}
    
    LABELS = ["AML / KYC", "Fund Regulation", "Market Manipulation",
              "Stablecoins", "Custody & Wallet Security",
              "Liquidity", "Tokenomics", "Tax & Reporting"]
    
    results = {l:[] for l in LABELS}
    
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
    client = OpenAI()
    
    for q in qs:
        prompt = (f"Act like a senior compliance officer. Choose ONLY ONE of these categories for the following question: "
                  f"{', '.join(LABELS)}.\n\nQuestion: {q}\n\n"
                  f"Reply with just the category name, nothing else.")

        try:
            rsp = client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=50
            )

            category = rsp.choices[0].message.content.strip()
            
            # Find best matching category
            matched_category = None
            for label in LABELS:
                if label.lower() in category.lower():
                    matched_category = label
                    break

            # If no match found, use closest match
            if not matched_category:
                for label in LABELS:
                    if any(word.lower() in category.lower() for word in label.split()):
                        matched_category = label
                        break

            # If still no match, assign to first category
            if not matched_category and LABELS:
                matched_category = LABELS[0]

            # Add question to the matched category
            if matched_category:
                results[matched_category].append(q)

        except Exception as e:
            print(f"Error categorizing question: {e}")

    return {k:v for k,v in results.items() if v}


# Process documents for RAG
def process_documents_for_rag(file_path, file_type):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    try:
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type == 'txt':
            loader = TextLoader(file_path)
        elif file_type == 'docx':
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_type}")
        
        file_documents = loader.load()
        chunks = text_splitter.split_documents(file_documents)
        documents.extend(chunks)
        return documents, len(chunks)
    except Exception as e:
        print(f"Error processing document: {e}")
        return [], 0


# Create vector store from documents
def create_vectorstore(documents):
    try:
        embedding = SimpleEmbeddings()
        vectorstore = SimpleFAISS.from_documents(documents, embedding)
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


# Generate answer
def generate_answer(question, vectorstore, api_key=None):
    if not vectorstore:
        return "No vector store available. Please upload documents first.", []
    
    try:
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # If no API key is provided, fallback to simple snippet-based response
        if not api_key:
            answer = f"Based on the documents I've analyzed, I found information related to '{question}'.\n\n"
            answer += "The key points from the relevant sections include:\n"
            
            for i, doc in enumerate(docs, 1):
                content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                answer += f"\n{i}. {content}"
                
            answer += "\n\nFor a more detailed analysis, please consult the original documents."
            
            return answer, docs
        
        # Use Gemma 2 9B with Groq API for proper answer generation
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
        client = OpenAI()
        
        prompt = f"""<|system|>
You are a senior compliance officer, highly experienced in the Due Diligence process for crypto asset funds.
Use ONLY the provided documentation context below to answer the user's question.
Keep the answer structured, professional, and comprehensive.
If the information is not in the context, say you don't have that information.

Context:
{context}
</|system|>

<|user|>
{question}
</|user|>

<|assistant|>"""
        
        try:
            response = client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Add source information at the end
            answer += "\n\nSources:"
            for i, doc in enumerate(docs, 1):
                metadata = getattr(doc, 'metadata', {})
                source = metadata.get('source', f'Document {i}')
                page = metadata.get('page', '')
                page_info = f" (page {page})" if page else ''
                answer += f"\n{i}. {source}{page_info}"
                
            return answer, docs
            
        except Exception as e:
            print(f"Error with Groq API: {e}")
            # Fallback to simple snippet-based response
            answer = f"Error generating AI response. Here are the relevant sections:\n\n"
            for i, doc in enumerate(docs, 1):
                answer += f"\n{i}. {doc.page_content[:200]}...\n"
            return answer, docs
            
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"I'm sorry, I couldn't generate an answer due to an error: {str(e)}", []


# Function to use in place of split_by_titles from original notebook
def split_by_titles(pdf_path):
    return extract_sections_from_pdf(pdf_path) 