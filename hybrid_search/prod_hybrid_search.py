# Semantic search is bad at searching for the meaningless docs like code, identifiers
from langchain_chroma import Chroma
# So we need to add BM25 for searching the exact matched data like code and sth

from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
from langchain_core.documents import Document

documents = [
    Document(
        page_content=(
            "Product SKU-7742X is a home router with traffic prioritization. "
            "When several devices share a busy connection, it sends voice and "
            "video packets ahead of less urgent transfers. A person in an online "
            "meeting can keep speaking smoothly while another device downloads a game."
        ),
        metadata={"type": "product", "source": "catalog/SKU-7742X"},
    ),
    Document(
        page_content=(
            "Guide KB-1042 explains how to configure traffic priority on SKU-7742X. "
            "If meeting audio breaks up only while another device is downloading, "
            "open Traffic Settings, enable QoS, and set Video Conferencing to High. "
            "This gives time-sensitive calls the first share of available bandwidth."
        ),
        metadata={"type": "support", "source": "support/KB-1042"},
    ),
    Document(
        page_content=(
            "Product SKU-8810M is a two-node mesh Wi-Fi kit for homes with dead zones. "
            "A connection that works well beside the main router but fades in a distant "
            "bedroom usually points to a coverage problem. Place the second node partway "
            "between the router and that room so it can relay a strong signal."
        ),
        metadata={"type": "product", "source": "catalog/SKU-8810M"},
    ),
    Document(
        page_content=(
            "Guide KB-1043 covers error MESH-207 on the SKU-8810M kit. "
            "If the secondary node blinks amber after you move it, it has lost its "
            "wireless link to the primary node. Move it closer until the light turns "
            "solid green, then check whether devices in the distant room can reconnect."
        ),
        metadata={"type": "support", "source": "support/KB-1043"},
    ),
    Document(
        page_content=(
            "Product SKU-3308P is a battery backup for a router and fiber modem. "
            "When household electricity stops, its battery takes over automatically. "
            "Keeping both devices powered can preserve the internet connection during "
            "a short outage if the provider's network is still operating."
        ),
        metadata={"type": "product", "source": "catalog/SKU-3308P"},
    ),
    Document(
        page_content=(
            "Product SKU-7742S is an eight-port Ethernet switch. "
            "A home office can use it when the router's network sockets are all occupied. "
            "One cable links the switch to the router, leaving additional sockets for "
            "a desktop, printer, or games console to share the existing connection."
        ),
        metadata={"type": "product", "source": "catalog/SKU-7742S"},
    ),

    Document(
        page_content=(
            "For network connectivity issues, first check the "
            "ethernet cable and router status lights."
        ),
        metadata={"type": "troubleshooting", "source": "network_guide"},
    ),
    Document(
        page_content=(
            "Error code E_CONN_REFUSED indicates the server "
            "rejected the connection. Check firewall settings."
        ),
        metadata={"type": "error", "source": "connection_errors"},
    ),
    Document(
        page_content=(
            "The authentication process requires valid credentials. "
            "Use OAuth2 for secure API access."
        ),
        metadata={"type": "auth", "source": "api_auth_guide"},
    ),
]

print(f'Loaded {len(documents)} documents')

# Prepare vector retriever

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name='hybrid_search_test'
)

vector_retriever = vectorstore.as_retriever(
    search_kwargs={'k': 3}  # Return top 3
)
print("Vector retriever ready")

# Prepare BM25 retriever
bm25_retriever = BM25Retriever.from_documents(
    documents=documents,
    k=3
)
print("BM25 retriever ready")

# Ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
print("Hybrid retriever ready")

test_queries = [
    ("Which home router prioritizes video calls?", "catalog/SKU-7742X"),
    ("SKU-7742X: which QoS setting should I enable?", "support/KB-1042"),
    ("My phone gets one bar in the bedroom. What kit could help?", "catalog/SKU-8810M"),
    ("MESH-207", "support/KB-1043"),
    ("How can I stay online during a blackout?", "catalog/SKU-3308P"),
    ("SKU-7742S specifications", "catalog/SKU-7742S"),
    ("My internet stopped working. Which cable and lights should I check?", "network_guide"),
    ("E_CONN_REFUSED error", "connection_errors"),
    ("How does my application get permission to call a protected API?", "api_auth_guide"),
]


def test_query(query, name, retriever):
    results = retriever.invoke(query)
    print(f'\n{name} - Query : \"{query}\"')

    for i, doc in enumerate(results[:3], start=1):
        preview = doc.page_content[:80]
        if len(doc.page_content) > 80:
            preview += "..."
        print(f"{i}.     {preview}")

    return results


for query, expected_source in test_queries:
    print(f"\nExpected source: {expected_source}")

    for name, retriever in [
        ("VECTOR", vector_retriever),
        ("BM25", bm25_retriever),
        ("HYBRID", ensemble_retriever),
    ]:
        results = test_query(query, name, retriever)
        top_sources = [doc.metadata["source"] for doc in results[:3]]

        if expected_source in top_sources:
            rank = top_sources.index(expected_source) + 1
            print(f"  Expected document found at rank {rank}")
        else:
            print("  Expected document missing from the top 3")