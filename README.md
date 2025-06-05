# ConstBERT  
## Efficient Constant-Space Multi-Vector Retrieval

**Code coming soon!**

This repository contains the source code for the paper:  
**Efficient Constant-Space Multi-Vector Retrieval**  
by [Sean MacAvaney](https://macavaney.us/), [Antonio Mallia](https://antoniomallia.it), and [Nicola Tonellotto](https://tonellotto.github.io/), published at **ECIR 2025**.  
📄 [Read the paper (PDF)](./ConstBERT.pdf)

🏆 ConstBERT received the **Best Short Paper Honourable Mention** at **ECIR 2025**.

---

## 🔍 Overview

**ConstBERT** (Constant-Space BERT) is a multi-vector retrieval model designed for efficient and effective passage retrieval. It modifies the ColBERT architecture by encoding documents into a *fixed number of learned embeddings*, significantly reducing index size and improving storage and OS paging efficiency — all while retaining high retrieval effectiveness.

### Key Features:
- Fixed-size document representation (e.g., 32 vectors per document)
- Late interaction (MaxSim) for scoring
- End-to-end training of a pooling mechanism
- Comparable performance to ColBERT on MSMARCO and BEIR
- Efficient indexing and storage

---

## 🔗 Model Access

The pretrained model is available on Hugging Face:  
👉 [https://huggingface.co/pinecone/ConstBERT](https://huggingface.co/pinecone/ConstBERT)

```python
from transformers import AutoModel
import numpy as np

def max_sim(q: np.ndarray, d: np.ndarray) -> float:
    assert q.ndim == 2 and d.ndim == 2
    scores = np.dot(d, q.T)
    return float(np.sum(np.max(scores, axis=0)))

model = AutoModel.from_pretrained("pinecone/ConstBERT", trust_remote_code=True)

queries = ["What is the capital of France?", "latest advancements in AI"]
documents = [
    "Paris is the capital and most populous city of France.",
    "Artificial intelligence is rapidly evolving with new breakthroughs.",
    "The Eiffel Tower is a famous landmark in Paris."
]

query_embeddings = model.encode_queries(queries).numpy()
document_embeddings = model.encode_documents(documents).numpy()

print(max_sim(query_embeddings[0], document_embeddings[0]) > max_sim(query_embeddings[0], document_embeddings[1]))
# Output: True
```

## 📦 Model Details

- **Architecture**: BERT-based encoder with a learned pooling layer  
- **Embedding size**: 128  
- **Document vectors per passage**: 32  
- **Interaction**: MaxSim between document and query embeddings

### How it works

ConstBERT compresses token-level BERT embeddings into a *fixed number (C)* of document-level vectors using a learned linear projection. These vectors capture diverse semantic aspects of the document. Relevance is computed via a MaxSim operation between the query token embeddings and the fixed document vectors.

This design offers a trade-off between **storage/computation efficiency** and **retrieval effectiveness**, configurable by choosing the number of vectors `C`.

---

Please cite the following paper if you use this code, or a modified version of it:
```bibtex
@article{constbert,
  title={Efficient Constant-Space Multi-Vector Retrieval},
  author={MacAvaney, Sean and Mallia, Antonio and Tonellotto, Nicola},
  booktitle = {The 47th European Conference on Information Retrieval ({ECIR})},
  year={2025}
}
```

## 📎 Related Resources

- 🔬 [ColBERT: Original Multi-vector Retrieval Framework](https://github.com/stanford-futuredata/ColBERT)  
- 📝 [Pinecone Blog](https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/)  
- 🔗 [The Turing Post](https://www.turingpost.com/p/bert)
---
