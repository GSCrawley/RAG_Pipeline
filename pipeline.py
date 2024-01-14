import os
import openai
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
# from haystack.components.summarizers import BartSummarizer
# from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
load_dotenv()

prompt_template = """
Given these documents, answer the question. It's ok to answer questions whose answers do not explicitly appear in the documents.exit
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


import PyPDF2
import glob

# Function to extract text from pdf file
def extract_text_from_pdf(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    pdf_file_obj.close()
    return text

# Iterate over all pdf files in 'docs' directory and write them to document store
for file_path in glob.glob('docs/**/*.pdf', recursive=True):
    text = extract_text_from_pdf(file_path)
    document_store.write_documents([Document(content=text)])

rag_pipeline.draw("rag_pipeline.png")

# Continuously ask questions until 'exit' command is given
while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    documents = rag_pipeline.run({"retriever": { "query": question}})
    summarized_documents = summarize_documents(documents["retriever"]["documents"])
    prediction = rag_pipeline.run(
        {
            "prompt_builder": {"question": question, "documents": summarized_documents},
            "llm": {}
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
