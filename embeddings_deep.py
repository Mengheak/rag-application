from langchain_ollama.embeddings import OllamaEmbeddings
import numpy as np

embeddings = OllamaEmbeddings(
    model='nomic-embed-text'
)

def basic_embeddings(): #Single text
    text = "What is Machine Learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")

def batch_embeddings():
    texts = [ "What is Machine Learning?",
            "Explain the concept of overfitting in ML",
             "How does neural network work?",
            ]
    batch_embedding = embeddings.embed_documents(texts) # use embed documents because it's a list of text

    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")

def similarity_search():
    docs = [
        "Python is a programming language",
        "Javascript is used for web development",
        "Machine Learning enables AI applications",
        "Deep Learning uses neural networks",
        "Cats are popular pets"
    ]
    query = "What programming languages exists?"
    docs_vectors = embeddings.embed_documents(docs)
    query_vectors = embeddings.embed_query(query)

    #Compute cosine similarity
    def cosine_similarity(vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    similarities = [cosine_similarity(query_vectors, docs_vector) for docs_vector in docs_vectors]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"{score:.4f} - {doc}")

def embedding_caching():
    from langchain_classic.embeddings.cache import CacheBackedEmbeddings

    from langchain_classic.storage import LocalFileStore
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(root_path=tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings,
            document_embedding_cache=store,
            namespace="exercise",
        )

        text = "What is Reinforcement Learning?"

        # First call - hits API
        print("First call (API):")
        vectors1 = cached_embeddings.embed_documents([text])
        print(f"  Embedded {len(vectors1)} documents")

        # Second call - from cache
        print("\nSecond call (Cache):")
        vectors2 = cached_embeddings.embed_documents([text])
        print(f"  Embedded {len(vectors2)} documents")

        # Verify same results
        print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")

if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    # similarity_search()
    embedding_caching()