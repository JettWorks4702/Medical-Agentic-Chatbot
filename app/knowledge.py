from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from app.config import settings
from openai import OpenAI
from google import genai
from google.genai import types

@lru_cache()
def clients():
    if not settings.gemini_api_key or not settings.pinecone_api_key:
        raise ValueError("Required API keys are not set in the environment variables.")
    return genai.Client(api_key=settings.gemini_api_key), Pinecone(api_key=settings.pinecone_api_key)
    # return OpenAI(api_key=settings.gemini_api_key), Pinecone(api_key=settings.pinecone_api_key)


def ensure_index():
    _, pc = clients()
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(settings.pinecone_index)


def embed(texts):
    ai, _ = clients()
    response = ai.models.embed_content(
        model=settings.gemini_embed_model,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=1536),
    )
    return [embedding.values for embedding in response.embeddings]
    # return [x.embedding for x in ai.embeddings.create(model=settings.gemini_embed_model, input=texts).data]


def search_kb(query, top_k=5):
   index = ensure_index()
   vector = embed([query])[0]
   result = index.query(vector=vector, top_k=top_k, include_metadata=True)
   return [{'page':m.metadata.get('page'), 'text':m.metadata.get('text',''), 'score':round(m.score, 4)} for m in result.matches]