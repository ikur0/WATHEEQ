## this just improves over rag.py in this directory
""" framework is from json (for NCA only)
From NCA.json


PDF Document
 ↓
Gatekeeper (Relevance Check)
 ↓
Split → Sections
 ↓
FOR EACH Section:
    ↓
    Split → small chunks
    ↓
    For each chunk:
        BM25 + FAISS
    ↓
    Merge all candidates
    ↓
    Deduplicate
    ↓
    Cross-Encoder (Section vs Candidate)
    ↓
    Top-K results
    ↓
    Context Assembly (Parent chunks)
    ↓
    LLM (Micro-Evaluation for this section)
    ↓
    Store JSON result
END LOOP
 ↓
Collect all section results
 ↓
Deduplicate findings (across sections)
 ↓
Aggregate scores
 ↓
LLM (Final Evaluation / Executive Summary)
 ↓
Final JSON Output
"""