import os
import json
import fitz  
import time
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from pathlib import Path

model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB
device = "cpu"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    return pages

# Split text hierarchically by headings and heuristics
def hierarchical_split(text):
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    for line in lines:
        if line.strip() == "":
            continue
        if line.strip().endswith(":") or line.strip().istitle():
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
        current_chunk.append(line)
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

# Embed chunks and rerank by similarity to combined persona + job
def rank_chunks(chunks, persona, job):
    query = f"{persona.strip()}. Task: {job.strip()}"
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)
    results = []
    for chunk in chunks:
        chunk_embedding = model.encode(chunk, convert_to_tensor=True, device=device)
        score = util.pytorch_cos_sim(query_embedding, chunk_embedding).item()
        results.append((score, chunk))
    results.sort(reverse=True)
    return results

# Process all PDFs in folder and return structured output
def analyze_documents(pdf_dir, persona, job, top_k=5):
    output = {
        "metadata": {
            "input_documents": [],
            "persona": persona,
            "job": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for file in Path(pdf_dir).glob("*.pdf"):
        pages = extract_text_from_pdf(str(file))
        output["metadata"]["input_documents"].append(file.name)
        page_chunks = []

        for i, text in enumerate(pages):
            chunks = hierarchical_split(text)
            for chunk in chunks:
                page_chunks.append({
                    "chunk": chunk,
                    "page": i + 1,
                    "doc": file.name
                })

        # Rank all chunks
        chunk_texts = [c["chunk"] for c in page_chunks]
        ranked = rank_chunks(chunk_texts, persona, job)

        seen = set()
        for rank, (score, chunk) in enumerate(ranked[:top_k], 1):
            for entry in page_chunks:
                if entry["chunk"] == chunk and (entry["doc"], entry["page"]) not in seen:
                    output["extracted_sections"].append({
                        "document": entry["doc"],
                        "page_number": entry["page"],
                        "section_title": chunk.strip().split('\n')[0][:80],
                        "importance_rank": rank
                    })
                    output["subsection_analysis"].append({
                        "document": entry["doc"],
                        "page_number": entry["page"],
                        "refined_text": chunk.strip()
                    })
                    seen.add((entry["doc"], entry["page"]))
                    break

    return output

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="challenge1b_input.json", help="Path to input JSON")
    parser.add_argument("--pdf_dir", type=str, default="PDFs/", help="Path to folder of PDFs")
    parser.add_argument("--output", type=str, default="challenge1b_output.json", help="Path to save output JSON")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    with open(args.input) as f:
        query_data = json.load(f)

    persona = query_data["persona"]["role"]
    job = query_data["job_to_be_done"]["task"]

    start_time = time.time()
    output = analyze_documents(args.pdf_dir, persona, job)
    elapsed = time.time() - start_time

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f" Completed in {elapsed:.2f}s. Output saved to {args.output}")