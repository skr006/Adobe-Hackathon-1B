# Adobe-Hackathon-1B
This repository contains the codebase for a persona-conditioned, relevance-aware, hierarchical PDF job analyzer, built for the Adobe GenAI Hackathon. It takes a persona and job description, analyzes PDF resumes using advanced NLP methods, and returns a relevance-ranked JSON output.

## 🚀 Features

✅ Persona-conditioned query understanding  
✅ Hierarchical chunking of long documents  
✅ Refined reranking using sentence embeddings  
✅ BM25 + Embedding-based ensemble (optional)  
🧾 Outputs JSON with relevance ranking  

## 🛠 Tech Stack

- Python 3.10
- SentenceTransformers (all-MiniLM-L6-v2)
- PyMuPDF (PDF parsing)
- TQDM
- FastAPI (used internally for testing only)
- Docker

---

## 📦 Installation

### 1. Clone this repo

bash
git clone https://github.com/skr006/Adobe-hackathon-1A
cd Adobe-hackathon-1A


### 2. Add your PDF files

Put all PDF resumes inside the pdfs/ directory.

Example:

pdfs/
├── resume1.pdf
├── resume2.pdf
└── resume3.pdf


### 3. Create the input JSON file

Create a file input.json in the root directory like below:

json
{
  "persona": "A product designer who understands both design systems and UX writing",
  "job": "UX Designer - Adobe XD and Figma with content design background"
}


---

## 🐳 Docker Setup

### 1. Build Docker Image

bash
docker build -t persona-job-analyzer .


### 2. Run the Analysis

bash
docker run --rm -v $(pwd):/app persona-job-analyzer


> The output will be written to output.json inside the root directory.

---

## 📤 Output

A sample output.json will look like this:

json
{
  "ranked_resumes": [
    {
      "filename": "resume2.pdf",
      "score": 0.89,
      "matched_chunks": [
        "Experience with Adobe XD and design systems...",
        "Led UX content team..."
      ]
    },
    {
      "filename": "resume1.pdf",
      "score": 0.71,
      "matched_chunks": [ ... ]
    }
  ]
}


---

## ⚙ Internals

- *Tokenizer + Chunker*: Splits each PDF into manageable semantic chunks
- *Embedder*: Encodes both the conditioned query and resume chunks
- *Similarity Reranker*: Ranks resumes using cosine similarity
- *Hierarchical Relevance Scorer*: Combines top scores and matched content

---

## 🧪 Local Testing (Optional)

If you want to run locally (without Docker):

bash
pip install -r requirements.txt
python main.py


Ensure you’re using PyTorch >= 2.1 or TensorFlow >= 2.0.  
Install sentence-transformers manually if needed:

bash
pip install torch torchvision torchaudio --upgrade
pip install sentence-transformers


---

## 🙏 Credits

Developed by [skr006](https://github.com/skr006) for the Adobe GenAI Hackathon.

---

## 📎 License

MIT License
