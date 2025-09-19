# real-estate-agent
It is an Agent with Long Term Memory of a property buyer's preferences and available houses in their location. And flexibility to find properties based on specific requirements. To save properties and schedule property tours. It is a functional Agent that showcases Capella's diverse data platform capabilities: operational/KV, vector, geospatial and built in caching, to find, serve and store unique properties for a buyer at speed.

## An Agent with Long Term Memory of:
- Available Properties
- Buyer's Preferences
- Saved Properties
- Scheduled Tours

## Single Agent Architecture (for now)
### Routes:
1. Direct to LLM - us.meta.llama4-maverick-17b-instruct-v1:0, for general queries e.g. mortgage approval process.
2. CB Vector Search Tool - finding properties based on cosine similarity of text embeddings.
3. Gmaps API/Geospatial Search - using lat, lon of property addresses, to find nearby schools and restaurants.
4. Realtime web search using Tavily API - for current housing market trends, average house prices.

## Americas AI Challenge Team:
1. Kevin Farley
2. Dan James
3. Seong Cho
4. Jake Wood

## Models Used:
- Text Embedding Model: Titan Text Embeddings V2 (1024 dimensions). Similarity Metric: Cosine for text similarity in a RAG application.
- Large Language Model: us.meta.llama4-maverick-17b-instruct-v1:0 

## Couchbase Capella Data Structure:
- Bucket: properties
  - Scope: 2025-listings 
  - Collection: united-states

- Bucket: profiles
  - Scope: buyers
  - Collection: 2025

- Bucket: profiles
  - Scope: tours
  - Collection: 2025
