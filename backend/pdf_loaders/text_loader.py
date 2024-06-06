import os
from utils.utils import generate_random_id
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dict


def split_text(text, limit=1000):
    # Split text into chunks of up to 'limit' characters
    words = text.split()
    chunks = []
    current_chunk = words[0]

    for word in words[1:]:
        if len(current_chunk) + len(word) + 1 <= limit:
            current_chunk += ' ' + word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def write_entry(current_entry, chunk, orig_entry, limit=1000):
    if len(current_entry['text']) + len(chunk) + 1 <= limit:
        current_entry['text'] += '\n' + chunk
        return None
    else:
        current_entry = orig_entry.copy()
        current_entry['element_id'] = generate_random_id()
        current_entry['text'] += '\n' + chunk
        return current_entry
    

def combine_with_parents(entries, limit=1000):
    from collections import defaultdict

    combined_entries = defaultdict(list)

    for entry in entries:
        id_val = entry['element_id']
        parent_id = entry.get('metadata', {}).get('parent_id')
        if parent_id and combined_entries[parent_id]:
                combined_entries[parent_id].append(entry)
        else:
            combined_entries[id_val].append(entry)
            
            
    results = {}
    for key, entries in combined_entries.items():
        combined_list = []
        
        orig_entry = entries[0].copy()
        current_entry = entries[0].copy()
        current_entry['text'] = entries[0]['text']
    
        for entry in entries[1:]:
            if len(entry['text']) > limit:
                # Split the long text into chunks
                text_chunks = split_text(entry['text'], limit)
                for chunk in text_chunks:
                    node = write_entry(current_entry, chunk, orig_entry, limit)
                    if node is not None:
                        combined_list.append(current_entry)
                        current_entry = node
            else:
                chunk = entry['text']
                node = write_entry(current_entry, chunk, orig_entry, limit)
                if node is not None:
                    combined_list.append(current_entry)
                    current_entry = node
            
        # Append the last modified or unmodified entry to the combined list
        combined_list.append(current_entry)
        
        # Store combined entries under the same ID
        results[key] = combined_list
        
    final_list = []
    for entries in results.values():
        final_list.extend(entries)

    return final_list

def load_pdf_text_content(docs_dir):
    elements = []
    for file in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file)
        elements.extend(partition(filename=file_path, content_type="application/pdf", include_page_breaks=True))
    
    isd = convert_to_dict(elements)
    parent_combined = combine_with_parents(isd)
    
    return parent_combined