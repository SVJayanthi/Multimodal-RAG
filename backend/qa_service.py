import os
import json
from openai import OpenAI
from fastapi import FastAPI
from typing import Dict
from pipeline.rag import LongFormQA
from load_corpus import load_corpus
from chunk_docs import chunk_and_save_docs
from pipeline import setup_dspy
from utils.citations import extract_cited_ids_from_paragraph, filter_answer_and_get_source_imgs

# Initialize FastAPI app
app = FastAPI()

# Stateful counter variable
openai_client = OpenAI(api_key=os.getenv("openaikey"))

# Create RAG Pipeline
NUM_PASSAGES = 6
NUM_RETRIEVED_PASSAGES_TO_SHOW = 3
MAX_OUTPUT_TOKENS = 300
collection_name = "gleen"
corpus_jsonl_path = "data/chunked_corpus.jsonl"

if not os.path.exists(corpus_jsonl_path):
    chunk_and_save_docs(docs_dir="docs/", save_images_dir="chatapp/assets/images", jsonl_path=corpus_jsonl_path)
corpus = load_corpus(corpus_jsonl_path)
llm = setup_dspy.setup_llm("gpt-4o", max_tokens= MAX_OUTPUT_TOKENS)
retriever = setup_dspy.setup_retriever(corpus, collection_name)
corpus_ids = [i['element_id'] for i in corpus]

rag_func = LongFormQA(passages_per_hop=NUM_PASSAGES)
# Add optional fitting script


@app.get("/answer")
async def get_answer_with_citations(message_json: str) -> Dict[str, str]:
    incoming_question = str(json.loads(message_json)['message'])

    pred = rag_func(incoming_question)
    print(f"Question: {incoming_question}")
    
    print(f"Predicted Answer: {pred.paragraph}")
    
    cited_ids_idx = extract_cited_ids_from_paragraph(pred.paragraph, corpus_ids)
    # Use retrieved sources
    if len(cited_ids_idx) == 0:
        retrieved_ids_idx = []
        for psg in pred.context[:NUM_RETRIEVED_PASSAGES_TO_SHOW]:
            retrieved_ids_idx.extend(extract_cited_ids_from_paragraph(psg, corpus_ids))
        cited_ids_idx = retrieved_ids_idx

    answer_filtered, source_image_paths = filter_answer_and_get_source_imgs(corpus, cited_ids_idx, pred.paragraph)
    # print(f"Source Ids: {str(source_ids_idx)}")
    
    return {"result": answer_filtered, "source_images": ";".join(source_image_paths)}

    
# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)

