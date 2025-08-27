from flask import Flask, render_template, request, jsonify, session
import os
import sys
import logging
import traceback
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import LangGraphRealEstateAgent

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    logger.warning("SECRET_KEY is not set. Sessions may be insecure in production.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agents = {}
# In-memory per-session saved/hidden properties
saved_properties = {}
hidden_properties = {}

@app.route('/')
def index():
    """Render the LangGraph chat interface."""
    return render_template('index.html')

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Initialize a new chat session."""
    try:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        logger.info(f"Starting new session: {session_id}")
        
        agent = LangGraphRealEstateAgent()
        agents[session_id] = agent
        
        logger.info(f"Agent created and stored for session: {session_id}")
        logger.info(f"Total active sessions: {len(agents)}")
        
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
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Failed to initialize chat session'
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages using the LangGraph workflow."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        buyer_name = data.get('buyer_name', '').strip() 
        session_id = session.get('session_id')
        
        logger.info(f"Chat request - Session ID: {session_id}")
        logger.info(f"Available agents: {list(agents.keys())}")
        
        if not session_id or session_id not in agents:
            logger.warning(f"Session not found: {session_id}")
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
        
        # Snapshot previous properties to detect if this turn produced new results
        prev_props = getattr(agent, 'last_properties', []) or []
        prev_keyset = {(p.get('id') or p.get('address') or p.get('name') or str(idx)) for idx, p in enumerate(prev_props)}

        response = agent.chat(user_message, buyer_profile)

        # Only surface properties if they changed this turn
        current_props = getattr(agent, 'last_properties', []) or []
        curr_keyset = {(p.get('id') or p.get('address') or p.get('name') or str(idx)) for idx, p in enumerate(current_props)}
        changed = (curr_keyset and curr_keyset != prev_keyset)
        properties = current_props[:3] if changed else []
        
        return jsonify({
            'success': True,
            'response': response,
            'buyer_profile': buyer_profile,
            'properties': properties
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your message'
        }), 500

@app.route('/api/buyers', methods=['GET'])
def list_buyers():
    """Return available buyer profiles for autocomplete."""
    try:
        from pathlib import Path
        profiles_path = Path(os.path.dirname(os.path.dirname(__file__))) / 'data-models' / 'profiles' / 'buyers.json'
        buyers = []
        if profiles_path.exists():
            import json
            with open(profiles_path, 'r') as f:
                data = json.load(f)
                # Normalize minimal fields for UI
                for p in data:
                    buyers.append({
                        'buyer': p.get('buyer'),
                        'budget': p.get('budget'),
                        'bedrooms': p.get('bedrooms'),
                        'bathrooms': p.get('bathrooms'),
                        'location': p.get('location')
                    })
        return jsonify({'success': True, 'buyers': buyers})
    except Exception as e:
        logger.error(f"Failed to load buyers: {e}")
        return jsonify({'success': False, 'buyers': []}), 500

@app.route('/api/save_property', methods=['POST'])
def save_property():
    try:
        data = request.get_json() or {}
        prop = data.get('property') or {}
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        key = prop.get('id') or f"{prop.get('address','')}|{prop.get('name','')}"
        if not key.strip():
            return jsonify({'success': False, 'error': 'Invalid property payload'}), 400
        saved_properties.setdefault(session_id, [])
        # Prevent duplicates by key
        if not any((p.get('id') or f"{p.get('address','')}|{p.get('name','')}") == key for p in saved_properties[session_id]):
            saved_properties[session_id].append(prop)
        return jsonify({'success': True, 'saved_count': len(saved_properties[session_id])})
    except Exception as e:
        logger.error(f"Save property error: {e}")
        return jsonify({'success': False, 'error': 'Failed to save property'}), 500

@app.route('/api/hide_property', methods=['POST'])
def hide_property():
    try:
        data = request.get_json() or {}
        prop = data.get('property') or {}
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        key = prop.get('id') or f"{prop.get('address','')}|{prop.get('name','')}"
        if not key.strip():
            return jsonify({'success': False, 'error': 'Invalid property payload'}), 400
        hidden_properties.setdefault(session_id, set())
        hidden_properties[session_id].add(key)
        return jsonify({'success': True, 'hidden_count': len(hidden_properties[session_id])})
    except Exception as e:
        logger.error(f"Hide property error: {e}")
        return jsonify({'success': False, 'error': 'Failed to hide property'}), 500

@app.route('/api/saved_properties', methods=['GET'])
def get_saved_properties():
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        props = saved_properties.get(session_id, [])
        return jsonify({'success': True, 'properties': props})
    except Exception as e:
        logger.error(f"Get saved properties error: {e}")
        return jsonify({'success': False, 'error': 'Failed to load saved properties'}), 500

@app.route('/api/search_properties', methods=['POST'])
def search_properties():
    """Dedicated endpoint for property search using LangGraph."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        buyer_name = data.get('buyer_name', '').strip()
        num_results = data.get('num_results', 5)
        session_id = session.get('session_id')
        
        if not session_id or session_id not in agents:
            return jsonify({
                'success': False,
                'error': 'Session not found. Please refresh the page.'
            }), 400
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        agent = agents[session_id]

        buyer_profile = {}
        if buyer_name:
            buyer_profile = agent.get_buyer_profile(buyer_name)
        
        result = agent.search_properties(query, num_results, buyer_profile)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'properties': result['properties'],
            'buyer_profile': buyer_profile
        })
        
    except Exception as e:
        logger.error(f"Property search error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while searching for properties'
        }), 500

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history for the current session."""
    try:
        session_id = session.get('session_id')
        logger.info(f"Clear history request - Session ID: {session_id}")
        logger.info(f"Available agents: {list(agents.keys())}")
        
        if not session_id:
            logger.warning("No session_id found in session")
            return jsonify({
                'success': False,
                'error': 'No active session found. Please refresh the page.'
            }), 400
            
        if session_id not in agents:
            logger.warning(f"Session {session_id} not found in agents")
            return jsonify({
                'success': False,
                'error': 'Session expired. Please refresh the page.'
            }), 400
        
        agent = agents[session_id]
        agent.clear_conversation_history()
        
        logger.info(f"Conversation history cleared for session {session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Conversation history cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while clearing history'
        }), 500

@app.route('/api/conversation_history', methods=['GET'])
def get_conversation_history():
    """Get conversation history for the current session."""
    try:
        session_id = session.get('session_id')
        
        if not session_id or session_id not in agents:
            return jsonify({
                'success': False,
                'error': 'Session not found. Please refresh the page.'
            }), 400
        
        agent = agents[session_id]
        history = agent.get_conversation_history()
        
        return jsonify({
            'success': True,
            'conversation_history': history
        })
        
    except Exception as e:
        logger.error(f"Get conversation history error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while retrieving history'
        }), 500

@app.route('/api/graph_info')
def graph_info():
    """Endpoint to get information about the LangGraph workflow."""
    try:
        return jsonify({
            'success': True,
            'graph_structure': {
                'nodes': [
                    'analyze_intent',
                    'property_search', 
                    'market_search',
                    'location_context',
                    'general_chat',
                    'format_response'
                ],
                'flow': 'START → analyze_intent → [property_search|market_search|location_context|general_chat] → format_response → END',
                'tools': {
                    'couchbase_vector_search': 'Semantic property search using Titan embeddings',
                    'tavily_search': 'Real-time web search for market information',
                    'google_maps': 'Reverse geocoding and nearby places (schools, restaurants) via Google Maps Platform',
                    'bedrock_llm': 'Llama 4 model for general conversation'
                }
            }
        })
    except Exception as e:
        logger.error(f"Graph info error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get graph information'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)