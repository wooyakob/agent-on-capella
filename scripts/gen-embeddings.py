import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_aws import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings

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
    oa_model = (
        os.getenv("FALLBACK_EMBEDDING_MODEL")
    )
    dims = int((os.getenv("FALLBACK_EMBEDDING_DIMENSIONS") or "").strip() or 1024)
    return OpenAIEmbeddings(model=oa_model, dimensions=dims)

embeddings_model = init_embeddings()

input_path = os.getenv("INPUT_PATH")  # Properties to embed file path
output_path = os.getenv("OUTPUT_PATH")  # Location to save Property documents with embeddings field

with open(input_path, "r", encoding="utf-8") as f:
    properties = json.load(f)

updated_properties = []

for property in properties:
    name = property.get("name", "").strip()
    description = property.get("description", "").strip()
    
    if not name:
        print("Skipping property with missing name.")
        continue

    combined_text = f"Name: {name}\nDescription: {description}"
    embedding_vector = embeddings_model.embed_documents([combined_text])[0]
    property["embedding"] = embedding_vector
    updated_properties.append(property)
    
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(updated_properties, f, indent=4)

print(f"Property embeddings generated and saved to: {output_path}")