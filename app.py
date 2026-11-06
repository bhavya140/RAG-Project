'''
REAL TIME 308
1.create a flask function that takes question from postman
2.read the question from json
3.load local faiss db
4.ask the question to faiss db and get relevant chunks
5.make a prompt with question and chunk
6.pass prompt to open ai llm
7.get response from open ai and return to postman or ui in json format
'''
from flask import Flask
from flask import request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from openai import OpenAI
from flask import jsonify 

app= Flask(__name__)
@app.route('/ask',methods=["post"])
def f1():
    print("POST request Recieved")
    print("Request reached Flask route!", flush=True)
    data = request.get_json()
    question = data.get("question")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    my_vector_store = FAISS.load_local('faiss_index',embeddings,allow_dangerous_deserialization=True)#step 3
    docs = my_vector_store.similarity_search(question) #step 4
    chunks = [doc.page_content for doc in docs]

#step 5
    my_prompt = f'''i am asking a question based on a given chunk
                please give me answer from the chunk only. if chunk doesnot have answer 
                dont give me generlaised answer. just say answer not avialable
                question : {question}
                context : {chunks}
    '''

    # step 6 
    
    client = OpenAI()
    response = client.responses.create(model="gpt-5.6-luna",input=my_prompt)

    # step 7
    return jsonify({'result':response.output_text})
app.run(debug=True)