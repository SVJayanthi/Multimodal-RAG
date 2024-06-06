import os
import json
from openai import OpenAI
from fastapi import FastAPI
from typing import Dict
from pipeline.rag import LongFormQA
from load_corpus import load_corpus
from chunk_docs import chunk_and_save_docs
from pipeline import setup_dspy
from utils.citations import extract_cited_titles_from_paragraph, get_source_idxs

# Initialize FastAPI app
app = FastAPI()

# Stateful counter variable
openai_client = OpenAI(api_key=os.getenv("openaikey"))

# Create RAG Pipeline
collection_name = "gleen"
corpus_jsonl_path = "data/chunked_corpus.jsonl"

if not os.path.exists(corpus_jsonl_path):
    chunk_and_save_docs(docs_dir="docs/", save_images_dir="chatapp/assets/images", jsonl_path=corpus_jsonl_path)
corpus = load_corpus(corpus_jsonl_path)
setup_dspy.setup_llm("gpt-4o")
setup_dspy.setup_retriever(corpus, collection_name)
corpus_ids = [i['element_id'] for i in corpus]

rag_func = LongFormQA()


@app.get("/answer")
async def get_answer_with_citations(message_json: str) -> Dict[str, str]:
    incoming_question = str(json.loads(message_json)['message'])

    pred = rag_func(incoming_question)
    print(f"Question: {incoming_question}")
    
    print(f"Predicted Answer: {pred.paragraph}")
    
    extracts = extract_cited_titles_from_paragraph(pred.paragraph, pred.context)
    source_ids_idx = get_source_idxs(extracts, corpus_ids)
    source_image_paths = [corpus[idx]['metadata']['image_location'] for idx in source_ids_idx]
    
    return {"result": pred.paragraph, "source_images": ";".join(source_image_paths)}

# Define the stateful API endpoint
# @app.get("/answer")
# async def openai_process_question(message_json: str) -> Dict[str, str]:
#     incoming_question = json.loads(message_json)['message']
#     try:
#         response = openai_client.chat.completions.create(
#             model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": incoming_question},
#                     ],
#                 }
#             ],
#         )
#         response = response.choices[0].message.content
#     except Exception as e:
#         response = e
#     return {"result": response, "source_images": "/images/doc3/page0.jpg;/images/doc3/page0.jpg"}
    
    
# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)

