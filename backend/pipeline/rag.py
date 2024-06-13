DEFAULT_PROMPT = """Given the below extracts & images, provide an answer to the given question.
Include citations in [] form at the end of the answer or say 'The answer is not in the given context'.
Question: {question}
Extracts: {context}
Answer: """

class CitationQA:
    def __init__(self, retriever, collection_name, lmm, prompt=None, passages_per_hop=6):
        super().__init__()
        self.retriever = retriever
        self.collection_name = collection_name
        self.lmm = lmm
        self.prompt = DEFAULT_PROMPT
        if prompt:
            self.prompt = prompt 
        self.passages_per_hop = passages_per_hop
        self.max_hops = 1
    
    def forward(self, question):
        search_result = self.retriever.query(
            collection_name=self.collection_name,
            query_text=question,
            limit=self.passages_per_hop
        )
        
        documents = [r.document for r in search_result]
        context = "\n".join(documents)
        prompt_filled = self.prompt.format(question=question, context=context)
        
        return self.lmm(prompt_filled, None), documents
    
    def __call__(self, prompt):
        return  self.forward(prompt)
