import openai
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cos_sim(text1, text2):
    text1_emb_opt = openai.Embedding.create(
        input=text1,
        model="text-embedding-ada-002"
    )

    text1_embeddings = text1_emb_opt['data'][0]['embedding']

    text2_emb_opt = openai.Embedding.create(
        input=text2,
        model="text-embedding-ada-002"
    )

    text2_embeddings = text2_emb_opt['data'][0]['embedding']

    # Calculate the cosine similarity between the LLM response and the predicate
    cos_sim = cosine_similarity([text1_embeddings], [text2_embeddings])

    return cos_sim[0][0]


# print(calculate_cos_sim("capital", "capital of"))
