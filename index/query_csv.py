# index/query_csv.py
import csv, json, os
from pathlib import Path
from typing import List, Dict
import numpy as np
import chromadb
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index" / "chroma"
TERMS_PATH = ROOT / "ingest" / "terms.json"
ARTIFACTS = ROOT / "artifacts" / "features.csv"
OUT = ROOT / "outputs" / "retrieval_preview.csv"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5
MIN_LEN = 12  # tokens threshold for "insufficient context"

def load_terms() -> Dict[str,str]:
    if TERMS_PATH.exists():
        return json.loads(TERMS_PATH.read_text())
    return {}

def expand_terms(text: str, terms: Dict[str,str]) -> str:
    # naive expansion: append synonyms to the query context
    # e.g., "SEA" => "SEA (Southeast Asia|Singapore|Malaysia|...)"
    parts = []
    for key, vals in terms.items():
        if key.lower() in text.lower():
            parts.append(f"{key}: {vals}")
    return text if not parts else text + "\n\n" + " ; ".join(parts)

def embedder():
    return TextEmbedding(model_name=EMBED_MODEL)

def retrieve(coll, emb, query: str, k: int):
    qvec = list(emb.embed([query]))[0]
    qvec = np.asarray(qvec, dtype=np.float32).tolist()
    res = coll.query(query_embeddings=[qvec], n_results=k,
                     include=["documents","metadatas","distances"])
    out = []
    for i, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0])):
        out.append({
            "passage_id": meta.get("chunk_id", f"chunk_{i}"),
            "jurisdiction": meta.get("jurisdiction",""),
            "law": meta.get("law",""),
            "source_url": meta.get("source_url",""),
            "snippet": (doc[:280].replace("\n"," ") + ("..." if len(doc)>280 else "")),
            "distance": dist
        })
    return out

def main():
    # setup
    OUT.parent.mkdir(parents=True, exist_ok=True)
    emb = embedder()
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    coll = client.get_collection("laws")
    terms = load_terms()

    # input CSV
    if not ARTIFACTS.exists():
        raise SystemExit(f"Missing {ARTIFACTS}. Create a CSV with columns: id,title,description")

    with ARTIFACTS.open() as f, OUT.open("w", newline="") as w:
        reader = csv.DictReader(f)
        fieldnames = ["feature_id","title","insufficient_context",
                      "top1_id","top1_jurisdiction","top1_law","top1_distance","top1_source","top1_snippet",
                      "top2_id","top2_jurisdiction","top2_law","top2_distance","top2_source","top2_snippet",
                      "top3_id","top3_jurisdiction","top3_law","top3_distance","top3_source","top3_snippet"]
        writer = csv.DictWriter(w, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            fid = row.get("id","").strip()
            title = row.get("title","").strip()
            desc = row.get("description","").strip()
            text = (title + "\n\n" + desc).strip()
            insufficient = (len(text.split()) < MIN_LEN)

            expanded = expand_terms(text, terms)
            hits = retrieve(coll, emb, expanded, TOP_K)

            out = {
                "feature_id": fid,
                "title": title,
                "insufficient_context": "yes" if insufficient else "no"
            }
            for i in range(3):
                if i < len(hits):
                    h = hits[i]
                    out.update({
                        f"top{i+1}_id": h["passage_id"],
                        f"top{i+1}_jurisdiction": h["jurisdiction"],
                        f"top{i+1}_law": h["law"],
                        f"top{i+1}_distance": f"{h['distance']:.4f}",
                        f"top{i+1}_source": h["source_url"],
                        f"top{i+1}_snippet": h["snippet"]
                    })
                else:
                    out.update({
                        f"top{i+1}_id": "",
                        f"top{i+1}_jurisdiction": "",
                        f"top{i+1}_law": "",
                        f"top{i+1}_distance": "",
                        f"top{i+1}_source": "",
                        f"top{i+1}_snippet": ""
                    })
            writer.writerow(out)

    print(f"✅ Wrote {OUT}")

if __name__ == "__main__":
    main()