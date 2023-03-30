import pandas as pd
import numpy as np
import openai
import pinecone
import streamlit as st
import secrets as secrets
import time
import os
import toml
from streamlit_chat import message
from PIL import Image

txtInputQuestion = "userQuestion"
pageTitle = "Bhagvad Gita GPT"

secrets = toml.load('secrets.toml')
#api_key = secrets['openai.api_key']['pinecone_api_key']
with open('secrets.toml', 'r') as f:
    secrets = toml.load(f)

# Get the OpenAI and Pinecone API keys from the secrets dictionary
openai_api_key = secrets['openai']['openai_api_key']
pinecone_api_key = secrets['pinecone']['pinecone_api_key']

pinecone.init(api_key=pinecone_api_key, environment='us-west4-gcp')

openai.api_key = openai_api_key



index_name = 'holygpt'

if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine'
    )
st.session_state_index = pinecone.Index(index_name)




df_index=pd.read_csv('only_verses.csv')

st.write("""
# GitaGPT
""")


st.write('''If you could ask Bhagavad Gita a question, what would it be?''')
st.markdown('\n')
st.markdown('\n')
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def card(context):
    return st.markdown(context)

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 300,
    "model": 'text-davinci-003',
}

header = """You are Krishna from Mahabharata, and you're here to selflessly help and answer any question or dilemma of anyone who comes to you.
    Analyze the person's question below and identify the base emotion and the root for this emotion, and then frame your answer by summarizing how your verses below
    apply to their situation and be emphatetic in your answer."""



def print_verse(q,retries=6):
    k=[]
    embed = get_embedding(q)
    for j in range(retries):
            try:
                for i in range(5):
                    k.append(int(st.session_state_index.query(embed, top_k=5)['matches'][i]['id']))
                return k    
            except Exception as e:
                if j == retries - 1:
                    raise e(st.markdown('Maximum number of retries exceeded'))
                else:
                    st.markdown("Failed to generate, trying again.")
                    time.sleep(2 ** j)
                    continue

def return_all_verses(retries=6):
    versee = []
    for j in range(retries):
        try:
            for i in verse_numbers:
                versee.append(f"{df_index['index'][i]} \n")
            return versee
        except Exception as e:
            if j == retries - 1:
                raise e(st.markdown('Maximum number of retries exceeded'))
            else:
                st.markdown("Failed to generate, trying again.")
                time.sleep(2 ** j)
                continue
               

question=st.text_input("**How are you feeling? Ask a question or describe your situation below, and then press Enter.**",'',placeholder='Type your question here')
# if st.button('Enter'):
if question != '':
    output = st.empty()
    st.write('Bhagvad Gita says: ') 
    verse_numbers = print_verse(question)
    verses = return_all_verses()
    verse_strings = "".join(return_all_verses())
    prompt = f'''{header}\nQuestion:{question}\nVerses:\n{verse_strings}\nAnswer:\n'''

    
def generate_prompt(question):
    prompt = f"I am here to help you. What is your question or problem? {question}"
    return prompt

def generate_chat_response(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        temperature=0.7,
        n=1,
        stop=None,
        timeout=20,
    )
    message = response.choices[0].text.strip()
    return message

# Get user input
question = st.text_input("Ask a question or describe your situation below, and then press Enter.")

# Generate chat response
if question:
    prompt = generate_prompt(question)
    message = generate_chat_response(prompt)
    st.write("Bhagvad Gita says: ", message)


def clear_text(textInput):

    st.session_state[textInput] = ""

def generate_response_davinci(question):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=generate_prompt(question),
        temperature=0.6,
        max_tokens=2048
    )
    return response.choices[0].text

def get_text():
    input_text = st.text_input("Hello, ask me a question about life and philosophy.",placeholder="Type Your question here.", key=txtInputQuestion)
    return input_text

def page_setup(title, icon):
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout='centered',
        initial_sidebar_state='auto',
        menu_items={
            'About': 'About your application: **This is Bhagvad Gita GPT, a simple ChatGPT use case demo to show how one can easily leverage openAI APIs to create intelligent conversational experiences related to a specific topic.**'
        }
    )
    st.sidebar.title('Creators :')
    st.sidebar.markdown('PRIYAL KHURANA')
    st.sidebar.write("DIVYANSH KUMAR")
    st.sidebar.write("MITALI CHAUDHARY")

    
if __name__ == '__main__':

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    icon = Image.open('gita.jpg')

    #setup page
    page_setup(pageTitle,icon)


    col1, col2 = st.columns(2)
    with col1:
        st.title("Bhagvad Gita GPT")
    with col2:
        st.image(icon)
    #st.write("test 1")

    user_input = get_text()

    print("get_text called.")
    if user_input:
        output = generate_response_davinci(user_input)

        # store the output
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    #if st.session_state['generated']:

    #st.markdown(response["choices"][0]["text"].strip(" \n"))
    #st.markdown('\n\n')
    #st.markdown("Relevant verses:")
    #st.markdown(verse_strings.replace('\n','\n\n'))



st.write('''\n\n Here's some examples of what you can ask:
1. I've worked very hard but I'm still not able to achieve the results I hoped for, what do I do?
2. I made a million dollars manipulating the stock market and I'm feeling great.
3. How can I attain a peace of mind?
''')

st.write('\n\n\n\n\n\n\n')

st.write('''Note: This is an AI model trained on Bhagvad Gita and it generates responses from that perspective.''')
