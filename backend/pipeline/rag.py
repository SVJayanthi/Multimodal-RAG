import dspy
from dsp.utils import deduplicate

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class GenerateCitedParagraph(dspy.Signature):
    """Generate a paragraph with citations."""
    content = dspy.InputField(desc="may contain relevant facts", format=list)
    question = dspy.InputField()
    paragraph = dspy.OutputField(desc="includes citations in [] form or say 'The answer is not in the given context'")

class LongFormQA(dspy.Module):
    def __init__(self, passages_per_hop=6, max_hops=1):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_cited_paragraph(content=context, question=question)
        pred = dspy.Prediction(context=context, paragraph=pred.paragraph)
        return pred
    

