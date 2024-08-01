import streamlit as st 
from streamlit_chat import message 
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# about the chat bot 
def Email_spam():
    url='https://miro.medium.com/v2/resize:fit:1200/1*_igArwmR7Pj_Mu_KUGD1SQ.png'
    st.image(url,width=400,)
    st.markdown(f"""
    <style>
        /* Set the background image for the entire app */
        .stApp {{
            background-color: #708090;
            background-size: 100px;
            background-repeat:no;
            background-attachment: auto;
            background-position:center;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Load your model
    model = load_model(r'Pre-Built-Models/email_spam_model.h5')

    # Initialize tokenizer (Assuming you have it saved or recreate it if needed)
    # Load the tokenizer
    with open(r'Pre-Built-Models/email_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Streamlit interface
    st.write("Enter a message to check if it's spam or not:")

    # Text input
    user_input = st.text_input("Message:", "")

    # Button to classify
    if st.button("Classify"):
        # Preprocess the input
        input_sequences = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_sequences, padding='post', maxlen=25)
        
        # Get the model prediction
        prediction = model.predict(input_padded)[0][0]
        result = "Spam" if prediction > 0.5 else "Not Spam"
        
        # Display the result
        st.write(f"Prediction: {result}")
        st.write(f"Spam Probability: {prediction:.2f}")


def pre_process(text):
    import re
    import nltk
    from nltk.stem import PorterStemmer
    ps=PorterStemmer()
    text=text.strip()
    text=re.sub("<[^>]*>", " ",text)
    text=re.sub("[^a-zA-Z]"," ",text)
    text=text.lower()
    text=text.split()
    text = [ ps.stem(word) for word in text]
    return " ".join(text)

def Movie_review():
    url='https://tse4.mm.bing.net/th?id=OIP.sTY48YntmZNaHPDAWFGxlAHaDe&pid=Api&P=0&h=180'
    st.image(url,width=300)
    st.markdown(f"""
    <style>
        /* Set the background image for the entire app */
        .stApp {{
            background-color: #d3d3d3;
            background-size: 1300px;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
      
        </style>
    """, unsafe_allow_html=True)
    
    # Load your model
    model = load_model(r'Pre-Built-Models/movie_review_model.h5')

    # Initialize tokenizer (Assuming you have it saved or recreate it if needed)
    # Load the tokenizer
    with open(r'Pre-Built-Models/movie_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Streamlit interface
    st.write("Enter a Review to check if it's Positive or Negative:")

    # Text input
    user_input = st.text_input("Review:", "")
    processed_input=pre_process(user_input)

    # Button to classify
    if st.button("Review"):
        # Preprocess the input
        input_sequences = tokenizer.texts_to_sequences([processed_input])
        input_padded = pad_sequences(input_sequences, padding='post', maxlen=175)
        
        # Get the model prediction
        prediction = model.predict(input_padded)[0][0]
        result = "Positive " if prediction > 0.5 else "Negative"
        
        # Display the result
        st.write(f"Prediction: {result}")
        st.write(f"Review Probability: {prediction:.2f}")
    
                
# main code 
st.sidebar.title("Select your Choice ")
file_type = st.sidebar.radio("Choose your BOT", ("Email Spam", "Movie Review"))

if file_type == "Email Spam":
    st.title("                  Email Spam 🤖")
    Email_spam()
elif file_type == "Movie Review":
    st.title("    Movie Review  📸 ")
    Movie_review()  
