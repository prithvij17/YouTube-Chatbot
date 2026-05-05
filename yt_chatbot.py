import os
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

video_id = "Gfr50f6ZBvo"

#print("h1")
api = YouTubeTranscriptApi()  
transcript_list = api.fetch(video_id, languages=["en"])

transcript = " ".join(chunk.text for chunk in transcript_list)


#print("\n h3")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=880,chunk_overlap=150)
chunk= text_splitter.create_documents([transcript])

#print("h4")


model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#print("h5")

Store=FAISS.from_documents(chunk,model)
#print("h6")

ret=Store.as_retriever(search_kwargs={"k":2})


templte=PromptTemplate(
    template="""You are helpful AI assistant which answer from provided transcript context.
    If you don't know the answer say out of the scope. 
    {context}
    Question: {query}""",
    input_variables=['context','query']
)

query="summarize video"

retrieved_document= ret.invoke(query)
#since from text splitter we get output in different chunks but in template we need a single string so:
context = "\n\n".join(i.page_content for i in retrieved_document)

#final_temp=templte.invoke({"context":context,"query":query}) will be done in chain later


repo_id = "HuggingFaceH4/zephyr-7b-beta"
llm=HuggingFaceEndpoint(repo_id=repo_id,
    task="conversational",
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

chat_model=ChatHuggingFace(llm=llm)

#r=chat_model.invoke(final_temp)#will we done in chain later just done here to understand
#print(r.content)


#parser
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()


#chain formation

from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda

def f_context(retrieved_document):
    contextt = "\n\n".join(i.page_content for i in retrieved_document)
    return contextt

parallel_Chain=RunnableParallel({
    "context": ret|RunnableLambda(f_context),
    "query":RunnablePassthrough()

})
main_chain=parallel_Chain | templte | chat_model | parser
r=main_chain.invoke(query)
print(r)

