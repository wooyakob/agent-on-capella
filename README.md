# real-estate-agent
This Real Estate Agent uses Langgraph to route between different tools based on a user's message.
The Agent can help a user with:
1. User Profile Based Property Search: Couchbase Vector Search and Bedrock Titan Embeddings for dream properties.
2. Real Time Market Research: Tavily web search for current information.
3. Expert Real Estate Advice: Bedrock LLM for real estate guidance.

Americas AI Challenge Team:
1. Kevin Farley
2. Dan James
3. Seong Cho
4. Jake Wood

Models Used:
- Text Embedding Model: Titan Text Embeddings V2 (1024 dimensions). Similarity Metric: Cosine for text similarity in a RAG application.
- Large Language Model: us.meta.llama4-maverick-17b-instruct-v1:0 

Couchbase Capella Data Structure:
- Bucket: properties
- Scope: 2025-listings 
- Collection: united-states

- Bucket: profiles
- Scope: buyers
- Collection: 2025

Google Maps API: 
Access up to 10K calls per SKU at no cost per month with Google Maps Platform APIs.

