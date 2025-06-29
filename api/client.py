import requests
import streamlit as st

# essay endpoint
def essay(topic):
    response = requests.post("http://localhost:8000/essay/invoke", json={'input':{'topic':topic}})
    return response.json()['output']['content']

# poem endpoint
def poem(topic):
    response = requests.post("http://localhost:8000/poem/invoke", json={'input':{'topic':topic}})
    return response.json()['output']['content']


# streamlit ui
st.set_page_config(page_title="LangChain API Client", page_icon=":robot_face:")
st.title("LangChain API Client")

# essay topic input
essay_topic = st.text_input("Essay Topic")

# poem topic input
poem_topic = st.text_input("Poem Topic")

# essay button
if st.button("Generate Essay"):
    with st.spinner("Generating..."):
        essay_response = essay(essay_topic)
        st.write(essay_response)

# poem button
if st.button("Generate Poem"):
    with st.spinner("Generating..."):
        poem_response = poem(poem_topic)
        st.write(poem_response)


# if __name__ == "__main__":
#     st.run()
    