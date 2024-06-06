import dspy
import os
from dspy.retrieve.qdrant_rm import QdrantRM
import qdrant_client

def setup_retriever(corpus_data, collection_name):
    q_client = qdrant_client.QdrantClient(":memory:")
    q_client.add(
        collection_name,
        documents = [c['text'] for c in corpus_data],
        metadata=corpus_data,
        ids=[c['element_id'] for c in corpus_data], 
    )

    qdrant_retriever = QdrantRM(
        qdrant_client=q_client,
        qdrant_collection_name=collection_name,
    )
    dspy.settings.configure(rm=qdrant_retriever)


def setup_llm(model_name="gpt-4o"):
    if "gpt" in model_name:
        llm = dspy.OpenAI(model="gpt-4o", api_key=os.getenv("openaikey"))
        dspy.settings.configure(lm=llm)
    else:
        raise Exception("Model not yet implemented")