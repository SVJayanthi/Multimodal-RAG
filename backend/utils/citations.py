import regex as re

REGEX_CITATION_PATTERN = r'\[(.*?)\]'

def extract_text_by_citation(paragraph):
    citation_regex = re.compile(r'(.*?)(\[\d+\]\.)', re.DOTALL)
    parts_with_citation = citation_regex.findall(paragraph)
    citation_dict = {}
    for part, citation in parts_with_citation:
        part = part.strip()
        citation_num = re.search(r'\[(\d+)\]\.', citation).group(1)
        citation_dict.setdefault(str(int(citation_num) - 1), []).append(part)
    return citation_dict


def has_citations(paragraph):
    return bool(re.search(r'\[\d+\]\.', paragraph))

def extract_cited_ids_from_paragraph(paragraph, corpus_ids):
    cited_ids = [m.group(1) for m in re.finditer(REGEX_CITATION_PATTERN, paragraph)]
    return [corpus_ids.index(item) for item in cited_ids if item in corpus_ids]

def filter_answer_and_get_source_imgs(corpus, cited_ids, answer):
    counter = iter(range(1, 100))

    filtered_answer = re.sub(REGEX_CITATION_PATTERN, lambda m: '[' + str(next(counter)) + ']', answer)
    
    source_img_paths = [corpus[idx]['metadata']['image_location'] for idx in cited_ids]
    
    return filtered_answer, source_img_paths