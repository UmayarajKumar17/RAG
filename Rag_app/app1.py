from flask import Flask, render_template, request, Response, stream_with_context
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks import BaseCallbackHandler
import os
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone, ServerlessSpec
import json
import queue
import time

app = Flask(__name__)
load_dotenv()
api_key = "pcsk_5wAzEZ_APjUKGPT5JmWv212WhbmrurzYxsu4dgZm3ciuuXUanVVXKaagPQbu1sLpF17bc6"  # Your Pinecone API key

# Initialize Pinecone with proper error handling
try:
    pc = Pinecone(api_key=api_key)
    index = pc.Index("quickstart")
    print("Successfully connected to Pinecone index")
except Exception as e:
    print(f"Error connecting to Pinecone: {str(e)}")
    raise

# Initialize embeddings with error handling
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("Successfully initialized embedding model")
except Exception as e:
    print(f"Error initializing embeddings: {str(e)}")
    raise

class QueueCallback(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs):
        self.q.put({"type": "token", "data": token})

prompt_template = """
You are a knowledgeable assistant with extensive information about Harry Potter. Use the following information from the knowledge base to answer the user's question.
If the provided context contains the information, use it to give a detailed answer.
If you cannot find the specific information in the context, say "I apologize, but I don't have enough information in my current context to answer that specific question about Harry Potter."

Context: {context}

Question: {question}

Answer (using only the information from the context):
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Setup vector store and retriever with proper configuration
try:
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key="text"
    )
    
    retriever = VectorStoreRetriever(
        vectorstore=vector_store,
        search_type="similarity",
        search_kwargs={
            "k": 5,
            
        }
    )
    print("Successfully initialized vector store and retriever")
except Exception as e:
    print(f"Error setting up vector store and retriever: {str(e)}")
    raise

def generate_answer(question):
    q = queue.Queue()
    callback = QueueCallback(q)
    
    try:
        # Test retrieval using the new invoke method
        from langchain_core.messages import HumanMessage
        retrieved_docs = retriever.invoke(question)
        print(f"Retrieved {len(retrieved_docs)} documents")
        for i, doc in enumerate(retrieved_docs):
            print(f"Document {i + 1} content: {doc.page_content[:200]}...")
        
        llm = OllamaLLM(
            model="deepseek-r1:7b",
            callbacks=[callback],
            streaming=True
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        def process_query():
            try:
                # Get the answer and sources
                result = qa.invoke({"query": question})
                
                # Ensure streaming is complete
                time.sleep(0.5)
                
                # Process and send sources
                if "source_documents" in result and result["source_documents"]:
                    sources = []
                    for doc in result["source_documents"]:
                        if hasattr(doc, 'page_content') and doc.page_content.strip():
                            source_text = doc.page_content.strip()
                            sources.append(source_text)
                    
                    if sources:
                        print(f"Sending {len(sources)} sources")
                        q.put({"type": "sources", "data": sources})
                    else:
                        print("No valid sources found in result")
                else:
                    print("No source_documents in result")
                
            except Exception as e:
                print(f"Error in process_query: {str(e)}")
                q.put({"type": "error", "data": str(e)})
            finally:
                q.put(None)
        
        import threading
        thread = threading.Thread(target=process_query)
        thread.start()
        
        while True:
            try:
                item = q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
                time.sleep(0.05)
            except Exception as e:
                print(f"Error in generate_answer stream: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
                break
                
    except Exception as e:
        print(f"Error in generate_answer: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')
    return Response(
        stream_with_context(generate_answer(question)),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

if __name__ == '__main__':
    app.run(debug=True)