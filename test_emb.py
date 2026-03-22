from google import genai
from google.genai import types
client = genai.Client()
response = client.models.embed_content(
    model="models/gemini-embedding-001", 
    contents="Hello world", 
    config=types.EmbedContentConfig(output_dimensionality=768)
)
print(len(response.embeddings[0].values))
