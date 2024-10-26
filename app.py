# Import necessary libraries
import json
import random
import uuid
from functools import wraps
import jsonlines
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import requests
import os

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

# Load environment variables
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI LLM
llm = ChatOpenAI(
    temperature=0.7,
    api_key=openai_api_key,
    model_name="gpt-4o-mini"
)
embedding = OpenAIEmbeddings(api_key=openai_api_key)

# User authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# User-related functions
def save_user(username, password):
    """Save user information to db.jsonl"""
    user_id = str(uuid.uuid4())
    user_data = {
        'user_id': user_id,
        'username': username,
        'password': generate_password_hash(password),
        'saved_itineraries': []
    }
    with jsonlines.open('db.jsonl', mode='a') as writer:
        writer.write(user_data)
    return user_id


def get_user(username):
    """Retrieve user information from db.jsonl"""
    try:
        with jsonlines.open('db.jsonl', mode='r') as reader:
            for user in reader:
                if user['username'] == username:
                    return user
    except FileNotFoundError:
        pass
    return None


def update_user_itineraries(user_id, itinerary_id, itinerary_data):
    """Update user's itinerary information"""
    users = []
    try:
        with jsonlines.open('db.jsonl', mode='r') as reader:
            users = list(reader)
    except FileNotFoundError:
        users = []

    for user in users:
        if user['user_id'] == user_id:
            if 'saved_itineraries' not in user:
                user['saved_itineraries'] = []
            user['saved_itineraries'].append({
                'id': itinerary_id,
                'data': itinerary_data
            })

    with jsonlines.open('db.jsonl', mode='w') as writer:
        for user in users:
            writer.write(user)


# Load attraction data
def load_attractions():
    try:
        with open('attractions.txt', 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        return []

# Create vector store
vectorstore = FAISS.from_texts(load_attractions(), embedding)

# Helper function: Extract JSON content from AI response
def extract_message_content(ai_response):
    """
    Extract JSON content from AI response and validate
    """
    # Get response text
    if hasattr(ai_response, 'content'):
        content = ai_response.content
    elif isinstance(ai_response, str):
        content = ai_response
    elif isinstance(ai_response, dict) and 'content' in ai_response:
        content = ai_response['content']
    else:
        content = str(ai_response)

    try:
        # Try to parse JSON directly
        return json.loads(content)
    except json.JSONDecodeError:
        # If direct parsing fails, attempt to clean content and re-parse
        try:
            # Find content between the first { and last }
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            # If parsing still fails, return a default error JSON
            return {
                "title": "Itinerary generation error",
                "summary": "Sorry, an error occurred while generating the itinerary",
                "itinerary": []
            }


# Helper function: Extract plain text content from AI response
def extract_text_content(ai_response):
    """
    Extract plain text content from AI response
    """
    if hasattr(ai_response, 'content'):
        return ai_response.content
    elif isinstance(ai_response, str):
        return ai_response
    elif isinstance(ai_response, dict) and 'content' in ai_response:
        return ai_response['content']
    return str(ai_response)


# Route definitions
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if get_user(username):
            return render_template('register.html', error="Username already exists")

        user_id = save_user(username, password)
        session['user_id'] = user_id
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user(username)
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['user_id']
            return redirect(url_for('index'))

        return render_template('login.html', error="Incorrect username or password")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))


@app.route('/trip-planner')
def trip_planner():
    current_itinerary = session.get('current_itinerary', {}).get('final_itinerary')
    if current_itinerary:
        return render_template('itinerary.html', itinerary=current_itinerary)
    return render_template('trip_planner.html')


@app.route('/translator')
def translator():
    return render_template('translator.html')


@app.route('/translate', methods=['POST'])
def translate():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'translated': ''})

    prompt = f"""
    You are a professional travel translator. Translate the following text into English.
    Please pay special attention to:
    1. Accurately translating travel-related terms
    2. Using natural Chinese expressions
    3. Maintaining professionalism and readability in the travel context
    4. Return the translation directly, without additional information or explanations

    Text to translate:
    {text}
    """

    try:
        response = llm.invoke(prompt)
        translated_text = extract_text_content(response)
        return jsonify({'translated': translated_text})
    except Exception as e:
        return jsonify({'translated': f'Translation error: {str(e)}'})


@app.route('/attractions')
def attractions():
    all_attractions = load_attractions()
    selected_attractions = random.sample(all_attractions, min(9, len(all_attractions)))
    return render_template('attractions.html', attractions=selected_attractions)


@app.route('/save_itinerary', methods=['POST'])
@login_required
def save_itinerary():
    user_id = session['user_id']
    current_itinerary = session.get('current_itinerary', {}).get('final_itinerary')

    if not current_itinerary:
        return jsonify({'error': 'No itinerary to save'}), 400

    itinerary_id = str(uuid.uuid4())
    update_user_itineraries(user_id, itinerary_id, current_itinerary)

    share_url = url_for('view_itinerary', itinerary_id=itinerary_id, _external=True)
    return jsonify({'share_url': share_url})


@app.route('/itinerary/<itinerary_id>')
def view_itinerary(itinerary_id):
    try:
        with jsonlines.open('db.jsonl', mode='r') as reader:
            for user in reader:
                if 'saved_itineraries' in user:
                    for itinerary in user['saved_itineraries']:
                        if itinerary['id'] == itinerary_id:
                            return render_template('itinerary.html',
                                                   itinerary=itinerary['data'])
    except FileNotFoundError:
        pass

    return render_template('error.html', message="Itinerary not found")


# Get weather information
def get_weather_info(destination):
    """
    Get weather information for the destination
    """
    return {
        "time": "2024-10-26T02:45",
        "interval": 900,
        "temperature": 19.5,
        "windspeed": 4,
        "winddirection": 350,
        "is_day": 1,
        "weathercode": 3
    }


# Generate RAG query
def generate_rag_query(destination, preferences):
    """
    Generate query based on destination and preferences
    """
    return f"Recommend relevant attractions and activities for destination {destination} and preferences {preferences}."


# Get RAG recommendations
def get_rag_recommendations(query):
    """
    Get travel recommendations using the RAG system
    """
    # Sample document database
    retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Use invoke instead of run
    response = retrieval_chain.invoke({"query": query})
    return response['result']


def generate_initial_itinerary_prompt(destination, start_date, preferences):
    """
    Generate prompt for initial itinerary planning
    """
    prompt = f"""
    As a tour guide for {destination}, please generate a 5-day travel itinerary based on the following information and return it in JSON format:

    - Destination: {destination}
    - Departure Date: {start_date}
    - Preferences: {preferences}

    Please return the result following the JSON format below:
    {{
        "title": "A travel theme name generated based on the destination and preferences",
        "summary": "A brief overview of the trip, highlighting the destination's features, the current season and possible weather, recommended clothing, and special considerations for traveling to this destination (detailed cultural introductions)",
        "itinerary": [
            {{
                "day": "Day number",
                "date": "Specific date",
                "activities": [
                    {{
                        "time": "Activity time",
                        "description": "Detailed activity content",
                        "location": "Location",
                        "notes": "Notes or suggestions"
                    }}
                ]
            }}
        ]
    }}
    """
    return prompt


def generate_react_adjustment_prompt(initial_itinerary, current_weather, rag_recommendations):
    """
    Generate ReAct prompt for itinerary adjustments
    """
    prompt = f"""
    This is the initial itinerary JSON:
    {initial_itinerary}

    Please infer and adjust the itinerary based on the following information:
    1. Weather information: {current_weather}
    2. Recommended attractions and activities: {rag_recommendations}

    Please return the adjusted itinerary in the same JSON format.
    #response with English
    """
    return prompt

@app.route('/generate_itinerary', methods=['POST'])
def generate_itinerary():
    """
    Process itinerary generation request
    """

    try:
        # Get form data
        destination = request.form['destination']
        start_date = request.form['start_date']
        preferences = request.form['preferences']

        # Get supplementary information
        current_weather = get_weather_info(destination)
        rag_query = generate_rag_query(destination, preferences)
        rag_recommendations = get_rag_recommendations(rag_query)
        print(f'rag_recommendations: {len(rag_recommendations)}')
        # Generate initial itinerary
        initial_prompt = generate_initial_itinerary_prompt(destination, start_date, preferences)
        initial_response = llm.invoke(initial_prompt)
        initial_itinerary = extract_message_content(initial_response)
        print(f'initial_itinerary: {initial_itinerary}')
        # Generate final itinerary using ReAct
        react_prompt = generate_react_adjustment_prompt(initial_itinerary, current_weather, rag_recommendations)
        final_response = llm.invoke(react_prompt)
        final_itinerary = extract_message_content(final_response)
        print(f'final_itinerary: {final_itinerary}')
        # Store serializable data in session
        session['current_itinerary'] = {
            'destination': destination,
            'start_date': start_date,
            'preferences': preferences,
            'final_itinerary': final_itinerary,
            'rag_recommendations': rag_recommendations
        }

        return render_template('itinerary.html', itinerary=final_itinerary)
    except Exception as e:
        print(f'final_itinerary Error: {str(e)}')
        error_itinerary = {
            "title": "Itinerary generation error",
            "summary": f"Sorry, an error occurred while generating the itinerary: {str(e)}",
            "itinerary": []
        }
        return render_template('itinerary.html', itinerary=error_itinerary)


@app.route('/adjust_itinerary', methods=['POST'])
def adjust_itinerary():
    """
    Process itinerary adjustment request
    """
    try:
        adjustment_suggestion = request.form['adjustment']
        current_itinerary = session.get('current_itinerary', {})

        # Generate adjustment prompt
        adjustment_prompt = f"""
        原始行程JSON：
        {current_itinerary.get('final_itinerary')}
        
        用户的调整建议：
        {adjustment_suggestion}

        请根据用户的调整建议修改行程，保持相同的JSON格式返回调整后的结果。
        """

        # Generate adjusted itinerary with LLM
        adjusted_response = llm.invoke(adjustment_prompt)
        adjusted_itinerary = extract_message_content(adjusted_response)

        # Update itinerary in session
        current_itinerary['final_itinerary'] = adjusted_itinerary
        session['current_itinerary'] = current_itinerary

        return render_template('itinerary.html', itinerary=adjusted_itinerary)
    except Exception as e:
        error_itinerary = {
            "title": "Itinerary adjustment error",
            "summary": f"Sorry, an error occurred while adjusting the itinerary: {str(e)}",
            "itinerary": []
        }
        return render_template('itinerary.html', itinerary=error_itinerary)


# Start application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
