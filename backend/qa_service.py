import os
import json
from openai import OpenAI
from fastapi import FastAPI
from typing import Dict
from load_corpus import load_corpus, load_examples
from chunk_docs import chunk_and_save_docs
from pipeline import setup_rag
from pipeline.rag import CitationQA
from utils.citations import extract_cited_ids_from_paragraph, filter_answer_and_get_source_imgs

# Initialize FastAPI app
app = FastAPI()

# Stateful counter variable
openai_client = OpenAI(api_key=os.getenv("openaikey"))

# Create RAG Pipeline
NUM_PASSAGES = 6
NUM_RETRIEVED_PASSAGES_TO_SHOW = 3
MAX_OUTPUT_TOKENS = 300

# Name of collection & corpus
collection_name = "gleen"
corpus_jsonl_path = "data/chunked_corpus.jsonl"

# Load Corpus
if not os.path.exists(corpus_jsonl_path):
    chunk_and_save_docs(docs_dir="docs/", save_images_dir="chatapp/assets/images", jsonl_path=corpus_jsonl_path)
corpus = load_corpus(corpus_jsonl_path)
llm = setup_rag.setup_llm("gpt-4o", max_tokens= MAX_OUTPUT_TOKENS)
retriever = setup_rag.setup_retriever(corpus, collection_name)
corpus_ids = [i['element_id'] for i in corpus]

rag_func = CitationQA(retriever, collection_name, llm, passages_per_hop=NUM_PASSAGES)
# Optional, fit the pipeline
# fit_pipeline(rag_func, load_examples("data/examples.csv"))

@app.get("/answer")
async def get_answer_with_citations(message_json: str) -> Dict[str, str]:
    incoming_question = str(json.loads(message_json)['message'])

    prediction, context = rag_func(incoming_question)
    print(f"Question: {incoming_question}")
    
    print(f"Predicted Answer: {prediction}")
    
    cited_ids_idx = extract_cited_ids_from_paragraph(prediction, corpus_ids)
    # Use retrieved sources
    if len(cited_ids_idx) == 0:
        retrieved_ids_idx = []
        for psg in context:
            retrieved_ids_idx.extend(extract_cited_ids_from_paragraph(psg, corpus_ids))
        cited_ids_idx = retrieved_ids_idx

    answer_filtered, source_image_paths = filter_answer_and_get_source_imgs(corpus, cited_ids_idx, prediction)
    # print(f"Source Ids: {str(source_ids_idx)}")
    
    return {"result": answer_filtered, "source_images": ";".join(source_image_paths)}

    
# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)

