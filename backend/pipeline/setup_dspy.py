import dspy
import os
from dspy.retrieve.qdrant_rm import QdrantRM
import qdrant_client
from dspy.teleprompt import BootstrapFewShot

def setup_retriever(corpus_data, collection_name, embed_ids=True):
    q_client = qdrant_client.QdrantClient(":memory:")
    if embed_ids:
        corpus_texts = ["["+i['element_id'] + "] " + i['text'] for i in corpus_data]
    else:
        corpus_texts = ["["+i['element_id'] + "] " + i['text'] for i in corpus_data]
    corpus_ids = [c['element_id'] for c in corpus_data]
    
    q_client.add(
        collection_name,
        documents = corpus_texts,
        metadata=corpus_data,
        ids=corpus_ids, 
    )

    qdrant_retriever = QdrantRM(
        qdrant_client=q_client,
        qdrant_collection_name=collection_name,
    )
    dspy.settings.configure(rm=qdrant_retriever)
    return qdrant_retriever


def setup_llm(model_name="gpt-4o", **kwargs):
    if "gpt" in model_name:
        llm = dspy.OpenAI(model="gpt-4o", api_key=os.getenv("openaikey"), **kwargs)
        dspy.settings.configure(lm=llm)
        return llm
    else:
        raise Exception("Model not yet implemented")
    
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

def fit_rag(rag_pipeline, trainset):
    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

    # Compile!
    compiled_rag = teleprompter.compile(rag_pipeline, trainset=trainset)
    return compiled_rag
