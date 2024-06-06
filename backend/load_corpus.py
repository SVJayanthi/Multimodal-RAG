import dspy
from dspy.datasets import DataLoader
import json

def annotate_corpus(loaded_data, answers, doc_names):
    for idx, doc_name in enumerate(doc_names):
        per_doc_texts = ""
        source_ids_arr = []
        for entry in loaded_data:
            if doc_name in entry['metadata']['filename']:
                source_id = entry['element_id']
                per_doc_texts += "\n" + f"[{source_id}] " + entry['text']
                source_ids_arr.append(f"[{source_id}]")
                
        answers[idx] += "\n Sources: " + ",".join(source_ids_arr)
        

def load_examples(loaded_data, examples_path, input_keys = ["question"]):
    dataset = DataLoader().from_csv(file_path=examples_path)
    questions = [d['question'] for d in dataset]
    answers = [d['answer'] for d in dataset]
    doc_names = [d['doc_name'] for d in dataset]
    
    annotate_corpus(loaded_data, answers, doc_names)
    
    trainset = [dspy.Example({"question": q, "answer": a}).with_inputs(*input_keys) for q, a in zip(questions, answers)]

    return trainset
    
    
def load_corpus(corpus_path):
    with open(corpus_path, 'r') as f:
        loaded_data = []
        for line in f:
            obj = json.loads(line.strip())
            loaded_data.append(obj)
    return loaded_data