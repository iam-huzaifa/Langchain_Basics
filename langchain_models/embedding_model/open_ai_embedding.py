from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
embedding = OpenAIEmbeddings(model ='text-embedding-3-small',dimensions = 64)
doc = [
    "The cat is sleeping on the warm sofa.",
    "Stock prices fell sharply after the company announced losses.",
    "How do I cook pasta with tomato sauce?",
    "The spacecraft entered orbit around Mars.",
    "She felt lonely after moving to a new city.",
    "Python is a popular programming language for data science.",
    "Heavy rainfall caused flooding across several villages.",
    "The human heart pumps blood throughout the body.",
    "Real Madrid won the match with a goal in the final minute.",
    "A blue whale can weigh more than 100 tons.",
    "Please find attached the invoice for last month's services.",
    "Solar panels convert sunlight into electrical energy.",
    "The ancient Egyptians built pyramids as royal tombs.",
    "Turn left at the next intersection to reach the airport.",
    "He invested his savings in a small neighborhood bakery."
]
doc_embedding = embedding.embed_documents(doc)

query = 'tell me about programming language'
query_embedding = embedding.embed_query(query)

score = cosine_similarity([query_embedding],doc_embedding)[0]
index,scor = sorted(list(enumerate(score)),key = lambda x:x[1])[-1]
print(doc[index])
print(scor)
