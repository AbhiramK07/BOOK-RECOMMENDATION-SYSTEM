import streamlit as st
import requests
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API key is missing! Please check your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# Prompt template
prompt = PromptTemplate(
    input_variables=["author", "genre", "language"],
    template="""
You are a helpful assistant. Based on the following inputs, suggest relevant books:
- Author: {author}
- Genre: {genre}
- Language: {language}

Ensure the following:
1. Correct any typing errors or misspellings in the inputs.
2. Verify that each input is valid based on available data sources i.e Open Library.
3. Return a concise list of book titles only, including the author and language if available. 
4. Do not include extra information, labels, or formatting.

If any input is invalid or not found, do not include it in the response. Only valid suggestions should be returned.
"""
)

chain = prompt | llm

# Streamlit UI
st.set_page_config(page_title="Book Recommender")
st.title("AI Book Recommender")

# Inputs
author = st.text_input("Author Name").strip()
genre = st.text_input("Genre").strip()
language = st.text_input("Language").strip()

if st.button("Get Recommendations"):
    if not author and not genre and not language:
        st.warning("Please enter at least one field to continue.")
    else:
        st.spinner("Fetching books...")

        result = chain.invoke({
            "author": author,
            "genre": genre,
            "language": language
        })

        # Build Open Library query
        query_parts = []
        if author:
            query_parts.append(f"author:{author}")
        if genre:
            query_parts.append(f"subject:{genre}")
        if language:
            query_parts.append(f"language:{language}")

        search_query = " ".join(query_parts)

        @st.cache_resource
        def fetch_books(query):
            url = f"https://openlibrary.org/search.json?q={query}"
            return requests.get(url)

        response = fetch_books(search_query)

        if response.status_code == 200:
            books = response.json().get("docs", [])
            if books:
                st.subheader("Top Results")
                for book in books[:10]:
                    title = book.get("title")
                    book_author = ", ".join(book.get("author_name"))
                    year = book.get("first_publish_year")
                    book_lang = ", ".join(book.get("language", [])).upper() if "language" in book else "Unknown"
                    book_genre = ", ".join(book.get("subject", [])[:2]) if "subject" in book else None
                    cover_id = book.get("cover_i")
                    key = book.get("key")

                    st.markdown(f"**Title:** {title} ({year})")
                    st.markdown(f"**Author:** {book_author}")
                    st.markdown(f"**Language:** {book_lang}")
                    if book_genre:
                        st.markdown(f"**Genre:** {book_genre}")

                    if cover_id:
                        st.image(f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg", width=100)

                    if key:
                        st.markdown(f"[🔗 View](https://openlibrary.org{key})")

                    # Description
                    desc_response = requests.get(f"https://openlibrary.org{key}.json")
                    if desc_response.status_code == 200:
                        desc = desc_response.json().get("description", "")
                        if isinstance(desc, dict):
                            desc = desc.get("value", "")
                        if desc:
                            st.markdown(f"**Description:** {desc}")

                    st.markdown("---")
            else:
                st.info("No books found.")
        else:
            st.error("Could not retrieve data.")
