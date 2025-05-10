import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(page_title="AI Book Recommender Chatbot")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = PromptTemplate(
    input_variables=["genre", "author", "published_year", "language", "rating"],
    template="""
    You are an intelligent AI book recommendation system that provides book suggestions based on user preferences:

    - Genre: {genre}
    - Preferred Language: {language}
    - Author (Optional): {author}
    - Minimum Average Rating (Optional): {rating}
    - Year of Publication (Optional): {published_year}

    Recommend 10 books in the specified language and format the output as follows:

    **Title:** <Book Title>  \n
    **Genre:** <Book Genre>  \n
    **Year of Publication:** <Year> \n 
    **Author:** <Book Author>  \n
    **Average Rating:** <Rating> \n 
    **Description:** <Brief Description>  

    Sort the books by their ratings in descending order (highest to lowest).
    Ensure that all recommendations are relevant and strictly in {language}.
    Separate each recommendation clearly using "---".
    And in each recommendation use a separate line for every detail.
    """
)

# Streamlit UI
st.title("AI Book Recommender Chatbot")

# User Input Fields
genre = st.text_input("Enter a book genre (e.g., Fiction, Science, Fantasy):")
author = st.text_input("Enter an author (Optional):")
published_year = st.slider("Select the minimum published year (Optional):", 1900, 2025, 2000)
language = st.text_input("Enter your preferred language:", value="English")  # Default to English
rating = st.slider("Select minimum average rating (Optional):", 0.0, 5.0, 0.0)

if st.button("Recommend Books"):
    if genre.strip():
        response = (prompt | llm).invoke({
            "genre": genre,
            "author": author,
            "published_year": published_year,
            "language": language,
            "rating": rating
        })

        recommendations = response.content if hasattr(response, "content") else str(response)

        if recommendations.strip():
            st.subheader(f"Recommended Books in {language}")

            books_list = recommendations.split("---") 

            for book in books_list:
                st.markdown("**Book Recommendation:**")
                st.markdown(book.strip())
                st.write("───────────────────────────────────────") 
        else:
            st.warning("No recommendations found. Try different filters.")
    else:
        st.error("Please enter a genre!")
