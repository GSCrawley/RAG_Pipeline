import os
import openai
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator 
# from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder

load_dotenv()

prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
retriever = InMemoryBM25Retriever(document_store = document_store)
prompt_builder = PromptBuilder(template=prompt_template)
# answer_builder = AnswerBuilder()
llm = OpenAIGenerator(api_key=os.environ.get('OPENAI_API_KEY'))


rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
# rag_pipeline.add_component("answer_builder", AnswerBuilder)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
# rag_pipeline.connect("llm", "answer_builder")


document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."), 
    Document(content="My name is Mark and I live in Berlin."), 
    Document(content="My name is Giorgio and I live in Rome.")
])

rag_pipeline.draw("rag_pipeline.png")

# Ask a question
question = input("Ask a question: ")
prediction = rag_pipeline.run(
    {
        "retriever": { "query": question}, 
        "prompt_builder": {"question": question},
    }
)

for reply in prediction["llm"]["replies"]:
    print(reply)

# question = "Who lives in Paris?"
# results = rag_pipeline.run(
#     {
#         "retriever": {"query": question},
#         "prompt_builder": {"question": question},
#         "answer_builder": {"query": question},
#     }
# )

# for answer in results["answer_builder"]["answers"]:
#     print(answer.data)