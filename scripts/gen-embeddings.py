import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_aws import BedrockEmbeddings

embeddings_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name=os.getenv("AWS_REGION", "us-east-2")
)

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