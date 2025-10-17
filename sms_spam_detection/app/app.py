import streamlit as st 
import pickle

# Load Model and Vectorizer
model = pickle.load(open('C:\\Users\\mohan\\Downloads\\sms_spam_detection\\model\\spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('C:\\Users\\mohan\\Downloads\\sms_spam_detection\\model\\vectorizer.pkl', 'rb'))

# UI
st.title("📩 SMS Spam Detection Using Machine Learning")
st.write("Enter an SMS message below to check if it's spam or not.")

user_input = st.text_area("✉️ Enter Message:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        data = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][prediction]
        
        if prediction == 1:
            st.error(f"🚨 Spam Message Detected! (Confidence: {probability*100:.2f}%)")
        else:
            st.success(f"✅ This message is not spam. (Confidence: {probability*100:.2f}%)")
