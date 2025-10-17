import os
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import CouchbaseException
import couchbase.search as search
from couchbase.vector_search import VectorQuery, VectorSearch
from langchain_aws import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

question = "a nice property in a gated community"

def init_embeddings():
    # Try Bedrock Titan first
    try:
        model_id = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        region = os.getenv("AWS_REGION", "us-east-2")
        return BedrockEmbeddings(model_id=model_id, region_name=region)
    except Exception:
        pass
    # Fallback to OpenAI with dim=1024 to match Couchbase index
    oa_key = os.getenv("OPENAI_API_KEY")
    if not oa_key:
        raise RuntimeError("No embeddings available: Bedrock failed and OPENAI_API_KEY missing")
    oa_model = os.getenv("FALLBACK_EMBEDDING_MODEL")
    dims = int((os.getenv("FALLBACK_EMBEDDING_DIMENSIONS") or "").strip() or 1024)
    return OpenAIEmbeddings(model=oa_model, dimensions=dims)

embeddings_model = init_embeddings()

vector = embeddings_model.embed_query(question)

pa = PasswordAuthenticator(os.getenv("CB_USERNAME"), os.getenv("CB_PASSWORD"))
cluster = Cluster(os.getenv("CB_HOSTNAME"), ClusterOptions(pa))

bucket = cluster.bucket("properties")
scope = bucket.scope("2025-listings")
collection = scope.collection("united-states")

search_index = "properties-index"

try:
    search_req = search.SearchRequest.create(search.MatchNoneQuery()).with_vector_search(
        VectorSearch.from_vector_query(VectorQuery('embedding', vector, num_candidates=3))
    )
    
    result = scope.search(search_index,
                           search_req,
                           SearchOptions(limit=10, fields=["name", "description"])
                           )
    
    for row in result.rows():
        print("Found property:")
        print(f"  Name: {row.fields.get('name', 'N/A')}")
        print(f"  Description: {row.fields.get('description', 'N/A')}")
        #print(f"  Score: {row.score}")
        print("-" * 50)
    
    print("Reported total rows: {}".format(result.metadata().metrics().total_rows()))
        
except CouchbaseException as ex:
    print(f"Search failed: {str(ex)}")
    import traceback
    traceback.print_exc()