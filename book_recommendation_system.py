import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


prompt = PromptTemplate(
    input_variables=["genre", "language", "year_range", "authors", "ratings", "mood"],
    template="""
    Suggest 5 books based on the following preferences:
    - Genre: {genre}
    - Language: {language}
    - Year of Publication: {year_range}
    - Preferred Authors: {authors}
    - Minimum Book Rating: {ratings}
    - Current Mood: {mood}
    
    Provide the following details for each book:
    - Book Title
    - Author
    - Publishing Date
    - Price (approximate, in Indian Rupees)
    - Genre
    - Language
    - Book Rating (out of 5)
    - A short but exciting description that hooks the reader.
    
    Format the response so each book's details appear clearly on a new line.
    """
)

def get_book_recommendations(genre, language, year_range, authors, ratings, mood):

    chain = prompt | llm
    return chain.invoke({
        "genre": genre,
        "language": language if language else "Any",
        "year_range": year_range if year_range else "Any",
        "authors": authors if authors else "Any",
        "ratings": ratings if ratings else "Any",
        "mood": mood if mood else "Any"
    }).content


st.set_page_config(page_title="AI Book Recommendations")
st.title("Intelligent Book Recommendation System")
st.write("Discover personalized book recommendations powered by AI!")


genre = st.text_input("Favourite Genre:", placeholder="e.g., Sci-Fi, Mystery, Romance")
language = st.text_input("Preferred Language:", placeholder="e.g., English, Hindi, Spanish")
authors = st.text_input("Favourite Authors (Optional):", placeholder="e.g., J.K. Rowling, Dan Brown")
mood = st.text_input("Current Mood (Optional):", placeholder="e.g., Happy, Thoughtful, Adventurous")
ratings = st.slider("Minimum Book Rating (Optional):", 1.0, 5.0, 3.0, 0.1)
year_range = st.slider("Year of Publication Range (Optional):", 1900, 2025, (2000, 2025))

if st.button("Get Recommendations"):
    if not genre:
        st.error("Please enter a genre to get recommendations.")
    else:
        st.subheader("Here are some book recommendations tailored to your preferences.")
        recommendations = get_book_recommendations(genre, language, f"{year_range[0]}-{year_range[1]}", authors, ratings, mood)
        book_list = recommendations.split("\n\n")  

        for book in book_list:
            if book.strip():
                st.markdown(f"{book}")
                lines = book.split("\n")
                st.write("---")

