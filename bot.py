def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

import os
import pickle
import json
import numpy as np
import voyageai
from typing import List, Dict, Any
from tqdm import tqdm
import anthropic
import threading
import time
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

class ContextualVectorDB:
    def __init__(self, name: str, voyage_api_key=None, anthropic_api_key=None):
        if voyage_api_key is None:
            voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/contextual_vector_db.pkl"
        self.raw_content = "./merged_data"

        # Rate limiting config
        self.RATE_LIMIT = 60  # requests per minute
        self.DELAY = 60 / self.RATE_LIMIT  # ~2.14 seconds per request

        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()

        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()

    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """
        try:
            response = self.anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            temperature=0.0,
            messages=[
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text",
                                    "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                                    "cache_control": {"type": "ephemeral"} #we will make use of prompt caching for the full documents
                                },
                                {
                                    "type": "text",
                                    "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                                }
                            ]
                        }
                    ],
                )
            return response.content[0].text, response.usage
        except Exception as e:
            print(f"Error: {e}")
            return "", None
    

    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 1):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        
        texts_to_embed = []
        metadata = []
        total_chunks = len(dataset)

        def process_chunk(chunk, client_reference, model_name):
            file_names = os.listdir(self.raw_content)
            title_name = sanitize_filename(chunk['chapter_title'])
            file_name = next((f for f in file_names if title_name in f), None)
            if file_name is None:
                return {
                    'text_to_embed': chunk['text'],
                    'metadata': {
                        'chunk_id': chunk['chunk_id'],
                        'chapter_title': chunk['chapter_title'],
                        'original_content': chunk['text'],
                        'contextualized_content': ""
                    }
                }
            
            with open(os.path.join(self.raw_content, file_name), 'r') as f:
                doc = f.read()
            #for each chunk, produce the context
            # contextualized_text = summary_doc_chunk(doc, chunk['text'], client_reference, model_name)
            contextualized_text, usage = self.situate_context(doc, chunk['text'])

            with self.token_lock:
                try:
                    self.token_counts['input'] += usage.input_tokens
                    self.token_counts['output'] += usage.output_tokens
                    self.token_counts['cache_read'] += usage.cache_read_input_tokens
                    self.token_counts['cache_creation'] += usage.cache_creation_input_tokens
                except Exception as e:
                    pass
            
            return {
                #append the context to the original text chunk
                'text_to_embed': f"{chunk['text']}\n\n{contextualized_text}",
                'metadata': {
                    'chunk_id': chunk['chunk_id'],
                    'chapter_title': chunk['chapter_title'],
                    'original_content': chunk['text'],
                    'contextualized_content': contextualized_text
                }
            }

        count = 1
        for chunk in tqdm(dataset, desc="Processing chunks"):

            result = process_chunk(chunk, 'a', 'b')
           
            texts_to_embed.append(result['text_to_embed'])
            metadata.append(result['metadata'])
            time.sleep(self.DELAY)
            count += 1
        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()

    
    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        batch_size = 128
        with tqdm(total=len(texts), desc="Embedding chunks") as pbar:
            result = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_result = self.voyage_client.embed(batch, model="voyage-3.5-lite").embeddings
                result.extend(batch_result)
                pbar.update(len(batch))
        
        self.embeddings = result
        self.metadata = data

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed([query], model="voyage-3.5-lite").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        top_results = []
        counter = 1
        for idx in top_indices:
            result_string = "<context {counter}>\n".format(counter=counter)
            result_string += f"""**chapter** - {self.metadata[idx]['chapter_title']}\n**chunk_text** - {self.metadata[idx]['original_content']}\n**contextualized_chunk_text** - {self.metadata[idx]['contextualized_content']}\n"""
            result_string += "</context {counter}>".format(counter=counter)
            top_results.append(result_string)
            counter += 1
        
        return "\n\n##################\n\n".join(top_results)
    
    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])


    def validate_embedded_chunks(self):
        unique_contents = set()
        for meta in self.metadata:
            unique_contents.add(meta['text'])
    
        print(f"Validation results:")
        print(f"Total embedded chunks: {len(self.metadata)}")
        print(f"Unique embedded contents: {len(unique_contents)}")
    
        if len(self.metadata) != len(unique_contents):
            print("Warning: There may be duplicate chunks in the embedded data.")
        else:
            print("All embedded chunks are unique.")
