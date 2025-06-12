from fasthtml.common import *
from starlette.datastructures import UploadFile
import openai
import PyPDF2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import faiss
import pickle
import os
from datetime import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken
import io
import base64
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Add this for newer NLTK versions

class PDFKnowledgeBot:
    def __init__(self, openai_api_key: str):
        """Initialize the PDF Knowledge Bot with OpenAI API key."""
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.embeddings = []
        self.text_chunks = []
        self.chunk_metadata = []
        self.index = None
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def extract_text_from_pdf(self, pdf_bytes) -> str:
        """Extract text content from PDF bytes."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            return text
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess the extracted text."""
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        page_pattern = r'--- Page (\d+) ---'
        current_page = 1
        
        for sentence in sentences:
            page_match = re.search(page_pattern, sentence)
            if page_match:
                current_page = int(page_match.group(1))
                continue
                
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': current_page,
                    'chunk_id': len(chunks),
                    'tokens': current_tokens
                })
                
                overlap_text = self._get_overlap_text(current_chunk, overlap_tokens)
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(self.encoding.encode(current_chunk))
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'page': current_page,
                'chunk_id': len(chunks),
                'tokens': current_tokens
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str, max_tokens: int) -> str:
        """Get the last part of text for overlap, within token limit."""
        sentences = sent_tokenize(text)
        overlap_text = ""
        tokens_count = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = len(self.encoding.encode(sentence))
            if tokens_count + sentence_tokens <= max_tokens:
                overlap_text = sentence + " " + overlap_text
                tokens_count += sentence_tokens
            else:
                break
                
        return overlap_text.strip()
    
    def create_embeddings(self, text_chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for text chunks using OpenAI's embedding model."""
        embeddings = []
        failed_chunks = []
        
        for i, chunk in enumerate(text_chunks):
            try:
                if not chunk['text'].strip():
                    continue
                    
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk['text']
                )
                
                embedding = response.data[0].embedding
                
                if len(embedding) == 0 or all(x == 0 for x in embedding):
                    failed_chunks.append(i)
                    continue
                    
                embeddings.append(embedding)
                
            except Exception as e:
                failed_chunks.append(i)
                continue
        
        if not embeddings:
            raise ValueError("No valid embeddings were created. Please check your PDF content and API key.")
        
        return np.array(embeddings)
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for efficient similarity search."""
        if embeddings.size == 0:
            raise ValueError("Cannot build index with empty embeddings array")
            
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings.shape}")
            
        dimension = embeddings.shape[1]
        
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.index = faiss.IndexFlatIP(dimension)
        embeddings_f32 = embeddings.astype('float32')
        
        norms = np.linalg.norm(embeddings_f32, axis=1)
        zero_norm_indices = np.where(norms == 0)[0]
        
        if len(zero_norm_indices) > 0:
            for idx in zero_norm_indices:
                embeddings_f32[idx, 0] = 1e-8
        
        try:
            faiss.normalize_L2(embeddings_f32)
            self.index.add(embeddings_f32)
        except Exception as e:
            raise
    
    def process_pdf(self, pdf_bytes):
        """Complete PDF processing pipeline."""
        try:
            raw_text = self.extract_text_from_pdf(pdf_bytes)
            
            if not raw_text or not raw_text.strip():
                return False, "No text could be extracted from the PDF."
            
            cleaned_text = self.clean_text(raw_text)
            
            if not cleaned_text or not cleaned_text.strip():
                return False, "No usable text after cleaning."
                
            self.text_chunks = self.chunk_text(cleaned_text)
            
            if not self.text_chunks:
                return False, "No text chunks were created."
                
            self.text_chunks = [chunk for chunk in self.text_chunks if chunk['text'].strip()]
            
            if not self.text_chunks:
                return False, "All text chunks were empty after filtering."
                
            self.chunk_metadata = self.text_chunks.copy()
            
            embeddings = self.create_embeddings(self.text_chunks)
            
            if embeddings.size == 0:
                return False, "No valid embeddings were created."
                
            self.embeddings = embeddings
            self.build_faiss_index(embeddings)
            
            return True, "PDF processed successfully!"
            
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Search for most similar chunks to the query."""
        if self.index is None or not self.chunk_metadata:
            return []
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
            
            if np.any(np.isnan(query_embedding)) or np.any(np.isinf(query_embedding)):
                return []
            
            norm = np.linalg.norm(query_embedding)
            if norm == 0:
                return []
                
            faiss.normalize_L2(query_embedding)
            k = min(k, len(self.chunk_metadata))
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.chunk_metadata):
                    chunk_data = self.chunk_metadata[idx].copy()
                    chunk_data['similarity_score'] = float(score)
                    results.append(chunk_data)
            
            return results
            
        except Exception as e:
            return []
    
    def generate_answer(self, question: str, context_chunks: List[Dict]) -> Dict:
        """Generate answer using OpenAI GPT model with retrieved context."""
        if not context_chunks:
            return {
                'answer': "I couldn't find relevant information in the PDF to answer your question.",
                'sources': [],
                'confidence': 0.0
            }
        
        context_text = ""
        sources = []
        
        for i, chunk in enumerate(context_chunks):
            context_text += f"\n[Context {i+1} - Page {chunk['page']}]:\n{chunk['text']}\n"
            sources.append({
                'page': chunk['page'],
                'chunk_id': chunk['chunk_id'],
                'similarity': chunk['similarity_score']
            })
        
        prompt = f"""Based on the following context from a PDF document, answer the user's question accurately and comprehensively. 

Context:
{context_text}

Question: {question}

Instructions:
1. Answer based ONLY on the information provided in the context
2. If the context doesn't contain enough information, say so
3. Cite specific page numbers when referencing information
4. Be thorough but concise
5. If multiple perspectives are presented, acknowledge them

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Changed to more available model
                messages=[
                    {"role": "system", "content": "You are a helpful AI tutor that answers questions based on PDF content. Always cite page numbers and be accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            avg_similarity = np.mean([chunk['similarity_score'] for chunk in context_chunks])
            confidence = min(avg_similarity * 100, 95)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': sources,
                'confidence': 0.0
            }
    
    def suggest_questions(self, num_questions: int = 5) -> List[str]:
        """Generate suggested questions based on PDF content."""
        if not self.text_chunks:
            return []
        
        sample_text = "\n".join([chunk['text'] for chunk in self.text_chunks[:3]])
        
        prompt = f"""Based on this PDF content, suggest {num_questions} thoughtful questions that would help someone learn and understand the material better. Make the questions diverse - include factual, analytical, and application-based questions.

Content sample:
{sample_text[:2000]}...

Generate exactly {num_questions} questions, each on a new line starting with a number:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Changed to more available model
                messages=[
                    {"role": "system", "content": "You are an educational assistant that creates insightful study questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and any(char.isdigit() for char in q[:3])]
            
            cleaned_questions = []
            for q in questions:
                clean_q = re.sub(r'^\d+\.?\s*', '', q).strip()
                if clean_q and clean_q.endswith('?'):
                    cleaned_questions.append(clean_q)
            
            return cleaned_questions[:num_questions]
            
        except Exception as e:
            return []

# Global variables to store state
bot_instance = None
chat_history = []
suggested_questions = []
processing_status = ""
pdf_info = {}

# Initialize FastHTML app
app = FastHTML(
    hdrs=(
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"),
        Script(src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"),
        Style("""
            .chat-message { margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; }
            .question { background-color: #e3f2fd; }
            .answer { background-color: #f3e5f5; }
            .suggested-question { cursor: pointer; transition: background-color 0.2s; }
            .suggested-question:hover { background-color: #e8f5e8; }
            .processing { color: #1976d2; font-weight: bold; }
            .error { color: #d32f2f; }
            .success { color: #388e3c; }
            .file-upload { border: 2px dashed #ccc; padding: 2rem; text-align: center; margin-bottom: 1rem; }
            .metrics { display: flex; gap: 1rem; margin: 1rem 0; }
            .metric { text-align: center; padding: 1rem; background-color: #f5f5f5; border-radius: 0.5rem; }
        """)
    )
)

def render_chat_message(chat):
    """Render a single chat message."""
    return Div(
        Div(
            H6("Question:", cls="mb-2"),
            P(chat['question'], cls="mb-3"),
            H6("Answer:", cls="mb-2"),
            P(chat['answer'], cls="mb-3"),
            Div(
                Div(f"Confidence: {chat['confidence']:.1f}%", cls="text-muted"),
                Div(f"Time: {chat['timestamp']}", cls="text-muted"),
                cls="d-flex justify-content-between"
            ),
            *([
                H6("Sources:", cls="mb-2 mt-3"),
                Ul(*[Li(f"Page {source['page']} (Similarity: {source['similarity']:.2f})") 
                     for source in chat['sources']])
            ] if chat['sources'] else []),
            cls="chat-message answer"
        ),
        cls="mb-3"
    )

@app.get("/")
def home():
    """Main page."""
    global processing_status, pdf_info, suggested_questions, chat_history
    
    sidebar_content = Div(
        H4("Configuration", cls="mb-3"),
        Form(
            Div(
                Label("OpenAI API Key:", for_="api_key", cls="form-label"),
                Input(type="password", id="api_key", name="api_key", cls="form-control", placeholder="Enter your OpenAI API key"),
                cls="mb-3"
            ),
            Button("Set API Key", type="submit", cls="btn btn-primary"),
            hx_post="/set_api_key",
            hx_target="#status",
            cls="mb-4"
        ),
        
        H5("How to Use", cls="mb-3"),
        Ol(
            Li("Enter your OpenAI API key"),
            Li("Upload a PDF file"),
            Li("Wait for processing"),
            Li("Ask questions about the content"),
            Li("Use suggested questions for inspiration")
        ),
        
        H5("Question Types", cls="mb-3 mt-4"),
        Ul(
            Li(Strong("Factual: "), "\"What is...?\""),
            Li(Strong("Analytical: "), "\"Compare X and Y\""),
            Li(Strong("Application: "), "\"How would you apply...?\""),
            Li(Strong("Evaluation: "), "\"What evidence supports...?\"")
        ),
        
        *([
            H5("Debug Info", cls="mb-3 mt-4"),
            P(f"Chunks: {pdf_info.get('chunks', 0)}"),
            P(f"Pages: {pdf_info.get('pages', 0)}"),
            P(f"Index: {'✓' if pdf_info.get('index_ready', False) else '✗'}")
        ] if pdf_info else []),
        
        cls="col-md-3 p-3 bg-light"
    )
    
    main_content = Div(
        H1("🤖 PDF Knowledge Bot", cls="mb-2"),
        H5("Your Personal AI Tutor - Upload a PDF and ask questions!", cls="mb-4 text-muted"),
        
        Div(id="status", cls="mb-3"),
        
        # File upload section
        Div(
            H4("Upload PDF", cls="mb-3"),
            Form(
                Div(
                    Input(type="file", name="pdf_file", accept=".pdf", cls="form-control", required=True),
                    cls="file-upload"
                ),
                Button("Process PDF", type="submit", cls="btn btn-success"),
                method="post",
                action="/upload_pdf",
                enctype="multipart/form-data",
                hx_post="/upload_pdf",
                hx_target="#status",
                hx_encoding="multipart/form-data",
                cls="mb-4"
            ),
            cls="mb-4"
        ),
        
        # PDF info metrics
        *([
            Div(
                Div(
                    H6("File"),
                    P(pdf_info['filename']),
                    cls="metric flex-fill"
                ),
                Div(
                    H6("Chunks"),
                    P(str(pdf_info['chunks'])),
                    cls="metric flex-fill"
                ),
                Div(
                    H6("Pages"),
                    P(str(pdf_info['pages'])),
                    cls="metric flex-fill"
                ),
                cls="metrics"
            )
        ] if pdf_info else []),
        
        # Suggested questions
        *([
            Div(
                Button("🎯 Generate Study Questions", 
                       cls="btn btn-info mb-3",
                       hx_post="/generate_questions",
                       hx_target="#suggested-questions"),
                Div(id="suggested-questions"),
                cls="mb-4"
            )
        ] if pdf_info else []),
        
        # Questions from suggestions
        *([
            H4("📚 Suggested Study Questions", cls="mb-3"),
            Div(*[
                Div(
                    f"❓ {question}",
                    cls="suggested-question p-2 mb-2 border rounded",
                    hx_post="/ask_question",
                    hx_vals=json.dumps({"question": question}),
                    hx_target="#chat-area"
                )
                for question in suggested_questions
            ], cls="mb-4")
        ] if suggested_questions else []),
        
        # Question input
        *([
            H4("💬 Ask Questions", cls="mb-3"),
            Form(
                Div(
                    Input(type="text", name="question", placeholder="Ask a question about the PDF...", 
                          cls="form-control", required=True),
                    cls="mb-3"
                ),
                Button("🔍 Ask Question", type="submit", cls="btn btn-primary"),
                hx_post="/ask_question",
                hx_target="#chat-area",
                cls="mb-4"
            )
        ] if pdf_info else []),
        
        # Chat area
        Div(
            *([
                H4("💭 Chat History", cls="mb-3"),
                *[render_chat_message(chat) for chat in reversed(chat_history)],
                Button("🗑️ Clear Chat History", 
                       cls="btn btn-danger",
                       hx_post="/clear_chat",
                       hx_target="#chat-area")
            ] if chat_history else []),
            id="chat-area"
        ),
        
        id="main-content",
        cls="col-md-9 p-3"
    )
    
    return Html(
        Head(
            Title("PDF Knowledge Bot"),
            *app.hdrs
        ),
        Body(
            Div(
                sidebar_content,
                main_content,
                cls="row"
            ),
            cls="container-fluid"
        )
    )

@app.post("/set_api_key")
def set_api_key(api_key: str):
    """Set the OpenAI API key."""
    global bot_instance
    if api_key and api_key.strip():
        try:
            bot_instance = PDFKnowledgeBot(api_key.strip())
            return Div("✅ API key set successfully!", cls="alert alert-success")
        except Exception as e:
            return Div(f"❌ Error setting API key: {str(e)}", cls="alert alert-danger")
    else:
        return Div("⚠️ Please enter a valid API key.", cls="alert alert-warning")

@app.post("/upload_pdf")
async def upload_pdf(pdf_file):
    """Handle PDF upload and processing."""
    global bot_instance, pdf_info, processing_status, chat_history, suggested_questions
    
    print(f"Upload request received: {pdf_file}")  # Debug print
    
    if not bot_instance:
        return Div("❌ Please set your OpenAI API key first.", cls="alert alert-danger")
    
    # Check if file was uploaded
    if not pdf_file:
        return Div("❌ No file uploaded. Please select a PDF file.", cls="alert alert-danger")
    
    # Check if it's a valid file object
    if not hasattr(pdf_file, 'filename') or not hasattr(pdf_file, 'file'):
        return Div("❌ Invalid file upload. Please try again.", cls="alert alert-danger")
    
    # Check file extension
    if not pdf_file.filename or not pdf_file.filename.lower().endswith('.pdf'):
        return Div("❌ Please upload a valid PDF file.", cls="alert alert-danger")
    
    try:
        # Reset state
        chat_history = []
        suggested_questions = []
        
        print(f"Processing file: {pdf_file.filename}")  # Debug print
        
        # Read PDF content
        pdf_content = await pdf_file.read()
        
        print(f"File size: {len(pdf_content)} bytes")  # Debug print
        
        if len(pdf_content) == 0:
            return Div("❌ The uploaded file is empty.", cls="alert alert-danger")
        
        # Process PDF
        success, message = bot_instance.process_pdf(pdf_content)
        
        print(f"Processing result: {success}, {message}")  # Debug print
        
        if success:
            # Update PDF info
            pdf_info = {
                'filename': pdf_file.filename,
                'chunks': len(bot_instance.text_chunks),
                'pages': max([chunk['page'] for chunk in bot_instance.chunk_metadata]) if bot_instance.chunk_metadata else 0,
                'index_ready': bot_instance.index is not None
            }
            
            print(f"PDF info: {pdf_info}")  # Debug print
            
            # Return success message and redirect
            return Div(
                Div("✅ PDF processed successfully! You can now ask questions.", cls="alert alert-success"),
                Script("setTimeout(() => window.location.reload(), 1000);")
            )
            
        else:
            return Div(f"❌ {message}", cls="alert alert-danger")
            
    except Exception as e:
        print(f"Exception during PDF processing: {str(e)}")  # Debug print
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return Div(f"❌ Error processing PDF: {str(e)}", cls="alert alert-danger")

@app.post("/generate_questions")
def generate_questions():
    """Generate suggested questions."""
    global bot_instance, suggested_questions
    
    if not bot_instance or not bot_instance.text_chunks:
        return Div("❌ No PDF loaded.", cls="alert alert-danger")
    
    try:
        questions = bot_instance.suggest_questions(5)
        suggested_questions = questions
        
        return Div(*[
            Div(
                f"❓ {question}",
                cls="suggested-question p-2 mb-2 border rounded",
                hx_post="/ask_question",
                hx_vals=json.dumps({"question": question}),
                hx_target="#chat-area"
            )
            for question in questions
        ])
        
    except Exception as e:
        return Div(f"❌ Error generating questions: {str(e)}", cls="alert alert-danger")

@app.post("/ask_question")
def ask_question(question: str = None):
    """Handle question asking."""
    global bot_instance, chat_history
    
    if not bot_instance or not bot_instance.text_chunks:
        return Div("❌ No PDF loaded.", cls="alert alert-danger")
    
    if not question or not question.strip():
        return Div("❌ Please enter a question.", cls="alert alert-warning")
    
    try:
        # Search for relevant chunks
        similar_chunks = bot_instance.search_similar_chunks(question.strip(), k=5)
        
        # Generate answer
        result = bot_instance.generate_answer(question.strip(), similar_chunks)
        
        # Add to chat history
        chat_entry = {
            'question': question.strip(),
            'answer': result['answer'],
            'sources': result['sources'],
            'confidence': result['confidence'],
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        chat_history.append(chat_entry)
        
        # Return updated chat area
        return Div(
            H4("💭 Chat History", cls="mb-3"),
            *[render_chat_message(chat) for chat in reversed(chat_history)],
            Button("🗑️ Clear Chat History", 
                   cls="btn btn-danger",
                   hx_post="/clear_chat",
                   hx_target="#chat-area")
        )
        
    except Exception as e:
        return Div(f"❌ Error processing question: {str(e)}", cls="alert alert-danger")

@app.post("/clear_chat")
def clear_chat():
    """Clear chat history."""
    global chat_history
    chat_history = []
    return Div()

if __name__ == "__main__":
    import uvicorn
    print("🤖 Starting PDF Knowledge Bot...")
    print("📝 Make sure you have your OpenAI API key ready!")
    print("🌐 The app will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)