"""
Flask web application for the Real Estate Agent.
Provides a simple chat interface for users to interact with the AI agent.
"""
from flask import Flask, render_template, request, jsonify, session
import os
import sys
import logging
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("real_estate_agent", "real-estate-agent.py")
real_estate_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_estate_agent)
RealEstateAgent = real_estate_agent.RealEstateAgent

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agents = {}

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Initialize a new chat session."""
    try:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        agent = RealEstateAgent()
        agents[session_id] = agent
        
        system_prompt = """You are a friendly, professional real estate agent. 
        You help clients find their dream properties by understanding their needs and preferences.
        Be conversational, warm, and helpful. Ask follow-up questions to better understand what they're looking for.
        Keep responses concise but engaging. Don't mention technical details about embeddings or vector search.
        Simply introduce yourself as a real estate agent, do not give yourself a specific name."""
        
        greeting = agent.get_llm_response(
            "Introduce yourself as a real estate agent and ask the user to describe their dream property.",
            system_prompt
        )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'greeting': greeting
        })
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to initialize chat session'
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        buyer_name = data.get('buyer_name', '').strip() 
        session_id = session.get('session_id')
        
        if not session_id or session_id not in agents:
            return jsonify({
                'success': False,
                'error': 'Session not found. Please refresh the page.'
            }), 400
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400
        
        agent = agents[session_id]
        
        buyer_profile = {}
        if buyer_name:
            buyer_profile = agent.get_buyer_profile(buyer_name)
            if buyer_profile:
                logger.info(f"Using profile for {buyer_name}: {buyer_profile}")
        
        system_prompt = """You are a friendly, professional real estate agent. 
        You help clients find their dream properties by understanding their needs and preferences.
        Be conversational, warm, and helpful. Ask follow-up questions to better understand what they're looking for.
        Keep responses concise but engaging. Don't mention technical details about embeddings or vector search.
        Simply introduce yourself as a real estate agent, do not give yourself a specific name."""
        
        if agent.is_property_search_query(user_message):
            properties = agent.search_properties(user_message, buyer_profile=buyer_profile)
            property_results = agent.format_property_results(properties, buyer_profile)
            
            profile_context = ""
            if buyer_profile:
                budget = buyer_profile.get('budget', {})
                profile_context = f"""
                
                Client Profile:
                - Name: {buyer_profile.get('buyer', 'Unknown')}
                - Budget: ${budget.get('min', 0):,} - ${budget.get('max', 0):,}
                - Bedrooms: {buyer_profile.get('bedrooms', 'any')}
                - Bathrooms: {buyer_profile.get('bathrooms', 'any')}
                - Preferred Location: {buyer_profile.get('location', 'any')}
                """
            
            llm_prompt = f"""The client described their dream property as: "{user_message}"
            {profile_context}
            
            Here are the search results:
            {property_results}
            
            Provide a brief, enthusiastic response about these properties considering their profile and preferences. 
            Ask if they'd like more details about any specific property or if they'd like to refine their search."""
            
            agent_response = agent.get_llm_response(llm_prompt, system_prompt)
            
            return jsonify({
                'success': True,
                'response': agent_response,
                'properties': properties,
                'has_properties': len(properties) > 0,
                'profile_used': bool(buyer_profile)
            })
        else:
            conversation_context = ""
            if buyer_profile:
                budget = buyer_profile.get('budget', {})
                conversation_context = f" (Client: {buyer_profile.get('buyer', '')}, Budget: ${budget.get('min', 0):,}-${budget.get('max', 0):,}, {buyer_profile.get('location', 'any location')})"
            
            enhanced_message = user_message + conversation_context
            agent_response = agent.get_llm_response(enhanced_message, system_prompt)
            
            return jsonify({
                'success': True,
                'response': agent_response,
                'properties': [],
                'has_properties': False,
                'profile_used': bool(buyer_profile)
            })
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': 'Sorry, I encountered an issue. Please try again.'
        }), 500

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
