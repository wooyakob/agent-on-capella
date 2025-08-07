# real-estate-agent

Americas AI Challenge Team:

Kevin Farley
Dan James
Seong Cho
Jake Wood

Text Embedding Model: Titan Text Embeddings V2 (1024 dimensions)
Large Language Model: us.meta.llama4-maverick-17b-instruct-v1:0 

Bucket: properties
Scope: 2025-listings 
Collection: united-states

Similarity Metric
cosine
dot_product
l2_norm
cosine is best for semantic similarity, measuring the angle between vectors and focusing on direction versus magnitude. It is most widely used for text similarity in RAG applications. 

- Use the Llama 4 model to act as a friendly real estate agent
- Ask users to describe their dream property
- Use Titan embeddings to search for matching properties
- Present results in a conversational manner

Add basic CI/CD, add development branch with Gemini for code reviews of PRs into main.