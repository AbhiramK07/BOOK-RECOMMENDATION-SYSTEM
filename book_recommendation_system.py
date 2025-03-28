import os
import streamlit as st
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import langdetect  # For language detection
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure Streamlit page
st.set_page_config(page_title="AI Book Recommender Chatbot")

# Initialize LangChain LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Load Google Books API Key from Environment Variable
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

if not GOOGLE_BOOKS_API_KEY:
    raise ValueError("Please set the GOOGLE_BOOKS_API_KEY environment variable.")

GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes"

# Define chatbot prompt template
prompt = PromptTemplate(
    input_variables=["genre", "author", "published_year"],
    template="""
    You are an AI Book Recommender Chatbot. The user is looking for books with the following preferences:
    
    Genre: {genre}
    Author: {author}
    Published Year: {published_year}
    
    Correct any spelling mistakes or unclear text to provide the best recommendations.
    
    First, interpret the user input correctly. Then, generate a search query for finding books.
    
    Provide a structured query that can be used to search book databases effectively.
    """
)

def is_english(text):
    """Check if a given text is in English using language detection."""
    try:
        return langdetect.detect(text) == "en"
    except:
        return False  # If detection fails, assume it's not English

def fetch_books_from_google(genre, author, published_year):
    """Fetch books from Google Books API in English and filter non-English results."""
    query = f"{genre}"
    if author:
        query += f"+inauthor:{author}"
    
    params = {
        "q": query,
        "printType": "books",
        "orderBy": "relevance",
        "maxResults": 10,
        "langRestrict": "en",  # Fetch only English books
        "key": GOOGLE_BOOKS_API_KEY
    }

    response = requests.get(GOOGLE_BOOKS_API_URL, params=params)
    
    if response.status_code == 200:
        books_data = response.json().get("items", [])
        recommendations = []
        
        for book in books_data:
            volume_info = book.get("volumeInfo", {})
            title = volume_info.get("title", "Unknown Title")
            authors = ", ".join(volume_info.get("authors", ["Unknown Author"]))
            categories = ", ".join(volume_info.get("categories", ["Unknown Genre"]))
            pub_year = volume_info.get("publishedDate", "Unknown Year")[:4]
            rating = volume_info.get("averageRating", None)
            description = volume_info.get("description", "No description available.")

            # **Strictly ensure the book is in English**
            if is_english(title) and is_english(description):
                recommendations.append((rating if rating else 0, f"""
                Title: {title}  
                Author(s): {authors}  
                Genre: {categories}  
                Published Year: {pub_year}  
                Average Rating: {rating if rating else "Not Rated"}  
                Description: {description}
                """))

        # Sort recommendations by rating (higher first)
        recommendations.sort(reverse=True, key=lambda x: x[0])

        return [book[1] for book in recommendations] if recommendations else ["No English books found matching the criteria."]
    else:
        return ["Failed to fetch book data. Please try again later."]

# Streamlit UI
st.title("AI Book Recommender Chatbot")

# User preferences input
genre = st.text_input("Enter a Genre:")
author = st.text_input("Enter an Author (Optional):")
published_year = st.slider("Select Minimum Published Year (Optional):", 1900, 2025, 2000)

if st.button("Suggest"):
    if genre.strip():
        # AI refines the query
        refined_query = (prompt | llm).invoke({
            "genre": genre,
            "author": author,
            "published_year": published_year
        })

        # Extract AI-generated query
        refined_text = refined_query.content if hasattr(refined_query, "content") else str(refined_query)

        # Fetch books using Google Books API
        books = fetch_books_from_google(genre, author, published_year)

        # Display AI response
        st.subheader("Recommended Books")
        for book in books:
            st.markdown(book)
            st.write("---")  # Separate each suggestion with a line
    else:
        st.error("Please enter a genre!")
