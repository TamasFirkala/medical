
import openai
import os
import streamlit as st

#Defining OpeAI API key
openai.api_key = st.secrets["api_secret"]

#Building frontend and user input dialog with Python-streamlit 

st.title("Test ChatGPT in finding contact data of doctors and institutions on this website: https://shorturl.at/ksOQW  using GPT-4-Turbo model.")

query = st.text_input("What would you like to ask?", "")

if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            #Using Python-Llama_index for LLM application 
            from llama_index import SimpleDirectoryReader

            #Loading the contect of the webpage
            documents = SimpleDirectoryReader('./data').load_data()

            from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper
            from llama_index.llms import OpenAI
            
            #Chossing an LLm model and defining relevant parameters
            llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="gpt-4-1106-preview"))
            
            max_input_size = 4096
            num_output = 256
            max_chunk_overlap = 0.5
            prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

            #building dialog mechanism
            custom_LLM_index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper = prompt_helper)

            query_engine = custom_LLM_index.as_query_engine()

            response = query_engine.query(query)

            
                      
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")




