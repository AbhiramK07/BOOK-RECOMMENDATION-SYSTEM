**1. INTRODUCTION**

  **1.1 Project Overview**
  The AI Book Recommender Chatbot is a web-based application built using Streamlit and LangChain, 
  integrated with Google Gemini AI (Gemini-2.0-Flash) to provide personalized book recommendations 
  based on user preferences. Users can specify parameters such as genre, author, language, year of 
  publication, and rating to receive AI-generated book suggestions.

  **1.2 Objectives**

   - To create an AI-powered chatbot that recommends books based on user inputs.

   - To ensure recommendations are sorted based on average rating.

   - To provide detailed book information, including title, genre, publication year, author, rating, and description.

   - To implement a user-friendly interface using Streamlit.


**2. TECHNOLOGIES USED**

**Technology**                                **Purpose**

Python                                  :  Backend logic & development

Streamlit                               :  Web UI framework for chatbot deployment

LangChain                               :  AI-driven conversational framework

Google Gemini AI (Gemini-2.0-Flash)     :  LLM for book recommendations

dotenv                                  :  Environment variable management


**3. SYSTEM ARCHITECTURE**

  **3.1 Workflow**

   - User Input: The user provides details such as genre, author, language, publication year, and rating.
   
   - Prompt Creation: A structured prompt is generated with these parameters using LangChain PromptTemplate.
   
   - AI Processing: The prompt is sent to Google Gemini AI via LangChain.
    
   - Response Handling: The AI model returns book recommendations in a structured format.
    
   - Display Results: The recommendations are displayed using Streamlit with proper formatting.

  **3.2 Components**

   - Frontend: Streamlit UI for user input and result display.
    
   - Backend: Python-based logic using LangChain to process user requests.
    
   - AI Model: Google Gemini AI for generating book recommendations.


**4. FEATURES AND FUNCTIONALITIES**

 **4.1 User Input Fields**

   - Genre: Required input (e.g., Fiction, Science, Fantasy)
   
   - Author: Optional input for filtering books by a specific author
    
   - Published Year: Adjustable slider for selecting a minimum publication year (default: 2000)
    
   - Language: Preferred book language (default: English)
    
   - Minimum Rating: Adjustable slider for selecting a minimum average rating (0.0 to 5.0)

 **4.2 AI Book Recommendations**

   - AI generates 10 book recommendations based on user input.
    
   - Books are sorted in descending order of rating.
    
   - Each recommendation includes:
    
   - Title
    
   - Genre
    
   - Year of Publication
    
   - Author
    
   - Average Rating
    
   - Brief Description

  **4.3 UI Display Enhancements**
  
   - Markdown formatting for structured display.
    
   - Separator ("---") to clearly differentiate book recommendations.
    
   - Error Handling: Displays appropriate warnings if no recommendations are found.

