import os
import qdrant_client
from pipeline.lmm import OpenAILMM

def setup_retriever(corpus_data, collection_name, embed_ids=True):
    q_client = qdrant_client.QdrantClient(":memory:")
    if embed_ids:
        corpus_texts = ["["+i['element_id'] + "] " + i['text'] for i in corpus_data]
    else:
        corpus_texts = [i['text'] for i in corpus_data]
    corpus_ids = [c['element_id'] for c in corpus_data]
    
    q_client.add(
        collection_name,
        documents = corpus_texts,
        metadata=corpus_data,
        ids=corpus_ids, 
    )

    return q_client


def setup_llm(model_name="gpt-4o", **kwargs):
    if "gpt" in model_name:
        llm = OpenAILMM(model_name=model_name, **kwargs)
        return llm
    else:
        raise Exception("Model not yet implemented")
