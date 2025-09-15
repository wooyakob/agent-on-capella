from flask import Flask, render_template, request, jsonify, session
import os
import sys
import logging
import traceback
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import json
from dotenv import load_dotenv
# Couchbase SDK
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, QueryOptions

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import LangGraphRealEstateAgent

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")
# Make sessions persistent and increase lifetime to reduce timeouts
app.permanent_session_lifetime = timedelta(days=7)

@app.before_request
def _keep_session_alive():
    # Ensure the session remains permanent across requests
    try:
        session.permanent = True
    except Exception as e:
        logger.warning(f"Could not set session to permanent: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if not app.secret_key:
    logger.warning("SECRET_KEY is not set. Sessions may be insecure in production.")

agents = {}
# In-memory per-session saved/hidden properties
saved_properties = {}
hidden_properties = {}
tour_requests = {}

# --- Couchbase Capella (profiles/buyers/2025) ---
_cb_cluster = None

def get_cb_cluster():
    """Initialize and cache the Couchbase Cluster connection."""
    global _cb_cluster
    if _cb_cluster is not None:
        return _cb_cluster
    host = os.getenv("CB_HOSTNAME")
    user = os.getenv("CB_USERNAME")
    pwd = os.getenv("CB_PASSWORD")
    if not host or not user or not pwd:
        logger.error("Couchbase credentials not configured (CB_HOSTNAME, CB_USERNAME, CB_PASSWORD)")
        raise RuntimeError("Couchbase credentials not configured")
    auth = PasswordAuthenticator(user, pwd)
    _cb_cluster = Cluster(host, ClusterOptions(auth))
    _cb_cluster.wait_until_ready(timedelta(seconds=10))
    return _cb_cluster

def get_profiles_path() -> str:
    """Return `bucket`.`scope`.`collection` path for buyer saved properties.
    Defaults: profiles.buyers.2025, overridable by env vars CB_PROFILES_BUCKET, CB_PROFILES_SCOPE, CB_PROFILES_COLLECTION.
    """
    bucket = os.getenv("CB_PROFILES_BUCKET", "profiles")
    scope = os.getenv("CB_PROFILES_SCOPE", "buyers")
    collection = os.getenv("CB_PROFILES_COLLECTION", "2025")
    return f"`{bucket}`.`{scope}`.`{collection}`"

@app.route('/')
def index():
    """Render the LangGraph chat interface."""
    return render_template('index.html')

@app.route('/saved')
def saved_page():
    """Render a dedicated Saved Properties page."""
    return render_template('saved.html')

@app.route('/tours')
@app.route('/tours/<tour_id>')
def tours_page(tour_id=None):
    """Render the Tours page. If tour_id provided, the page will load that tour."""
    return render_template('tours.html', tour_id=tour_id or '')

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Initialize a new chat session."""
    try:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session.permanent = True
        
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
        
        if not session_id:
            logger.warning("No session_id in session cookie")
            return jsonify({'success': False, 'error': 'No active session. Please refresh the page.'}), 400
        if session_id not in agents:
            # Recreate agent for existing session to avoid timeouts after server restarts
            logger.info(f"Recreating agent for session: {session_id}")
            agents[session_id] = LangGraphRealEstateAgent()
        
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

        # Get saved properties for this session and buyer
        session_saved = saved_properties.get(session_id, [])
        buyer_saved = []
        if buyer_name:
            try:
                cluster = get_cb_cluster()
                collection_path = get_profiles_path()
                buyer_key = f"buyer::{buyer_name.lower()}"
                res = cluster.query(
                    f"SELECT b.saved_properties AS properties FROM {collection_path} AS b USE KEYS $buyer_key",
                    QueryOptions(named_parameters={'buyer_key': buyer_key})
                ).execute()
                rows = list(res)
                buyer_saved = rows[0].get('properties', []) if rows else []
                if buyer_saved is None:
                    buyer_saved = []
            except Exception as e:
                logger.error(f"Failed to retrieve buyer saved properties: {e}")
                buyer_saved = []
        
        # Combine session and buyer saved properties
        all_saved = session_saved + buyer_saved

        response = agent.chat(user_message, buyer_profile, all_saved)

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
        buyer_name = (data.get('buyer_name') or '').strip()
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        # Build a robust deduplication key
        addr = (prop.get('address') or (prop.get('location') or {}).get('address') or '').strip().lower()
        name = (prop.get('name') or prop.get('title') or '').strip().lower()
        price = prop.get('price')
        # Key priority: id → address|name → address → name|price
        if prop.get('id'):
            key = prop.get('id')
        elif addr and name:
            key = f"{addr}|{name}"
        elif addr:
            key = addr
        elif name and price is not None:
            key = f"{name}|{price}"
        else:
            key = ''
        if not str(key).strip():
            return jsonify({'success': False, 'error': 'Invalid property payload'}), 400
        logger.info(f"SaveProperty: session={session_id} buyer='{buyer_name}' key='{key}' addr='{addr}' name='{name}' price='{price}'")
        saved_properties.setdefault(session_id, [])
        # Prevent duplicates by key
        if not any((
            p.get('id')
            or (
                ((p.get('address') or (p.get('location') or {}).get('address') or '').strip().lower())
                and (p.get('name') or p.get('title'))
                and f"{(p.get('address') or (p.get('location') or {}).get('address') or '').strip().lower()}|{(p.get('name') or p.get('title')).strip().lower()}"
            )
            or ((p.get('address') or (p.get('location') or {}).get('address') or '').strip().lower())
            or (
                (p.get('name') or p.get('title')) and (p.get('price') is not None)
                and f"{(p.get('name') or p.get('title')).strip().lower()}|{p.get('price')}"
            )
        ) == key for p in saved_properties[session_id]):
            saved_properties[session_id].append(prop)

        # Also persist to Couchbase buyer document if buyer_name provided
        persisted = False
        saved_count_profile = None
        if buyer_name:
            try:
                cluster = get_cb_cluster()
                collection_path = get_profiles_path()
                buyer_key = f"buyer::{buyer_name.lower()}"
                # Build minimal snapshot
                snapshot = {
                    'id': prop.get('id'),
                    'name': prop.get('name') or prop.get('title'),
                    'address': prop.get('address') or (prop.get('location') or {}).get('address'),
                    'price': prop.get('price'),
                    'bedrooms': prop.get('bedrooms'),
                    'bathrooms': prop.get('bathrooms'),
                    'house_sqft': prop.get('house_sqft') or prop.get('sqft'),
                    'dedupe_key': key,
                    'saved_at': datetime.now().isoformat(),
                    # Linkage/provenance
                    'saved_by_key': buyer_key,
                    'saved_by_name': buyer_name,
                    'session_id': session_id
                }
                # Ensure buyer document exists WITHOUT overwriting existing content
                # 1) Try a plain INSERT; if the doc exists, ignore the error
                try:
                    cluster.query(
                        f"""
                        INSERT INTO {collection_path} (KEY, VALUE)
                        VALUES ($buyer_key, {{ "saved_properties": [] }})
                        """,
                        QueryOptions(named_parameters={
                            'buyer_key': buyer_key
                        })
                    ).execute()
                except Exception:
                    # Document likely exists — ignore
                    pass
                # 2) If doc exists but array missing, initialize it
                cluster.query(
                    f"""
                    UPDATE {collection_path} AS b
                    SET b.saved_properties = IFMISSINGORNULL(b.saved_properties, [])
                    WHERE META(b).id = $buyer_key
                    """,
                    QueryOptions(named_parameters={
                        'buyer_key': buyer_key
                    })
                ).execute()
                # Append snapshot if not duplicate
                qr = cluster.query(
                    f"""
                    UPDATE {collection_path} AS b
                    SET b.saved_properties = ARRAY_APPEND(b.saved_properties, $prop_snapshot)
                    WHERE META(b).id = $buyer_key
                      AND NOT ANY p IN b.saved_properties SATISFIES
                        COALESCE(p.dedupe_key, IFMISSINGORNULL(p.id, p.address || '|' || p.name)) = $dedupe_key
                      END
                    RETURNING ARRAY_LENGTH(b.saved_properties) AS saved_count
                    """,
                    QueryOptions(named_parameters={
                        'buyer_key': buyer_key,
                        'prop_snapshot': snapshot,
                        'dedupe_key': key
                    })
                ).execute()
                try:
                    upd_rows = list(qr)
                    logger.info(f"SaveProperty: update rows={upd_rows}")
                except Exception:
                    logger.info("SaveProperty: unable to log update rows")
                # Get current count
                res = cluster.query(
                    f"SELECT ARRAY_LENGTH(b.saved_properties) AS cnt FROM {collection_path} AS b USE KEYS $buyer_key",
                    QueryOptions(named_parameters={'buyer_key': buyer_key})
                ).execute()
                rows = list(res)
                if rows:
                    saved_count_profile = rows[0].get('cnt')
                persisted = True
            except Exception as e:
                logger.error(f"Failed to persist saved property to Couchbase: {e}")

        return jsonify({
            'success': True,
            'saved_count': len(saved_properties[session_id]),
            'profile_persisted': persisted,
            'profile_saved_count': saved_count_profile,
            'dedupe_key': key
        })
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

@app.route('/api/buyer_saved', methods=['GET'])
def get_buyer_saved():
    """Return saved properties for a given buyer profile from Couchbase"""
    try:
        buyer_name = (request.args.get('buyer_name') or '').strip()
        if not buyer_name:
            return jsonify({'success': False, 'error': 'buyer_name is required'}), 400
        cluster = get_cb_cluster()
        collection_path = get_profiles_path()
        buyer_key = f"buyer::{buyer_name.lower()}"
        res = cluster.query(
            f"SELECT b.saved_properties AS properties FROM {collection_path} AS b USE KEYS $buyer_key",
            QueryOptions(named_parameters={'buyer_key': buyer_key})
        ).execute()
        rows = list(res)
        props = rows[0].get('properties', []) if rows else []
        if props is None:
            props = []
        return jsonify({'success': True, 'properties': props})
    except Exception as e:
        logger.error(f"Get buyer saved error: {e}")
        return jsonify({'success': False, 'error': 'Failed to load buyer saved properties'}), 500

@app.route('/api/buyer_saved', methods=['DELETE'])
def delete_buyer_saved():
    """Delete a saved property from a buyer profile by key (Couchbase)"""
    try:
        data = request.get_json() or {}
        buyer_name = (data.get('buyer_name') or '').strip()
        key = (data.get('key') or '').strip()
        if not buyer_name or not key:
            return jsonify({'success': False, 'error': 'buyer_name and key are required'}), 400
        cluster = get_cb_cluster()
        collection_path = get_profiles_path()
        buyer_key = f"buyer::{buyer_name.lower()}"
        # Remove the matching property by dedupe key
        cluster.query(
            f"""
            UPDATE {collection_path} AS b
            SET b.saved_properties = ARRAY p FOR p IN b.saved_properties WHEN
                COALESCE(p.dedupe_key, IFMISSINGORNULL(p.id, p.address || '|' || p.name)) != $prop_key
            END
            WHERE META(b).id = $buyer_key
            RETURNING ARRAY_LENGTH(b.saved_properties) AS remaining
            """,
            QueryOptions(named_parameters={
                'buyer_key': buyer_key,
                'prop_key': key
            })
        ).execute()
        # Fetch remaining count for response
        res = cluster.query(
            f"SELECT ARRAY_LENGTH(b.saved_properties) AS cnt FROM {collection_path} AS b USE KEYS $buyer_key",
            QueryOptions(named_parameters={'buyer_key': buyer_key})
        ).execute()
        rows = list(res)
        remaining = rows[0].get('cnt') if rows else 0
        return jsonify({'success': True, 'deleted': True, 'remaining': remaining})
    except Exception as e:
        logger.error(f"Delete buyer saved error: {e}")
        return jsonify({'success': False, 'error': 'Failed to delete saved property'}), 500

@app.route('/api/search_properties', methods=['POST'])
def search_properties():
    """Dedicated endpoint for property search using LangGraph."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        buyer_name = data.get('buyer_name', '').strip()
        num_results = data.get('num_results', 5)
        session_id = session.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found. Please refresh the page.'}), 400
        if session_id not in agents:
            logger.info(f"Recreating agent for session: {session_id}")
            agents[session_id] = LangGraphRealEstateAgent()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        agent = agents[session_id]

        buyer_profile = {}
        if buyer_name:
            buyer_profile = agent.get_buyer_profile(buyer_name)
        
        # Get saved properties for this session and buyer
        session_saved = saved_properties.get(session_id, [])
        buyer_saved = []
        if buyer_name:
            try:
                cluster = get_cb_cluster()
                collection_path = get_profiles_path()
                buyer_key = f"buyer::{buyer_name.lower()}"
                res = cluster.query(
                    f"SELECT b.saved_properties AS properties FROM {collection_path} AS b USE KEYS $buyer_key",
                    QueryOptions(named_parameters={'buyer_key': buyer_key})
                ).execute()
                rows = list(res)
                buyer_saved = rows[0].get('properties', []) if rows else []
                if buyer_saved is None:
                    buyer_saved = []
            except Exception as e:
                logger.error(f"Failed to retrieve buyer saved properties: {e}")
                buyer_saved = []
        
        # Combine session and buyer saved properties
        all_saved = session_saved + buyer_saved
        
        result = agent.search_properties(query, num_results, buyer_profile, all_saved)
        
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
            return jsonify({'success': False, 'error': 'No active session found. Please refresh the page.'}), 400
        
        if session_id not in agents:
            logger.info(f"Recreating agent for session: {session_id}")
            agents[session_id] = LangGraphRealEstateAgent()
        
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

 # ------------------- Tours APIs -------------------

@app.route('/api/tours', methods=['POST'])
def create_tour():
    """Create a tour request for the current session.
    Payload: { property: {...}, preferred_time: optional ISO/date string, buyer_name: optional }
    Returns: { success, tour_id }
    """
    try:
        data = request.get_json() or {}
        prop = data.get('property') or {}
        preferred_time = (data.get('preferred_time') or '').strip()
        buyer_name = (data.get('buyer_name') or '').strip()
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        tour_id = str(uuid.uuid4())
        tour = {
            'id': tour_id,
            'session_id': session_id,
            'buyer_name': buyer_name or None,
            'property': prop,
            'status': 'pending',  # pending | accepted | confirmed | declined | cancelled
            'preferred_time': preferred_time or None,
            'confirmed_time': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
        }
        tour_requests.setdefault(session_id, {})[tour_id] = tour

        # Persist to Couchbase under buyer document if buyer_name provided
        if buyer_name:
            try:
                cluster = get_cb_cluster()
                collection_path = get_profiles_path()
                buyer_key = f"buyer::{buyer_name.lower()}"
                # Ensure buyer doc exists
                try:
                    cluster.query(
                        f"""
                        INSERT INTO {collection_path} (KEY, VALUE)
                        VALUES ($buyer_key, {{"saved_properties": [], "tours": []}})
                        """,
                        QueryOptions(named_parameters={'buyer_key': buyer_key})
                    ).execute()
                except Exception:
                    pass
                # Ensure tours array exists
                cluster.query(
                    f"""
                    UPDATE {collection_path} AS b
                    SET b.tours = IFMISSINGORNULL(b.tours, [])
                    WHERE META(b).id = $buyer_key
                    """,
                    QueryOptions(named_parameters={'buyer_key': buyer_key})
                ).execute()
                # Append tour if not already present by id
                cluster.query(
                    f"""
                    UPDATE {collection_path} AS b
                    SET b.tours = ARRAY_APPEND(b.tours, $tour)
                    WHERE META(b).id = $buyer_key
                      AND NOT ANY t IN b.tours SATISFIES t.id = $tour_id END
                    RETURNING 1
                    """,
                    QueryOptions(named_parameters={'buyer_key': buyer_key, 'tour_id': tour_id, 'tour': tour})
                ).execute()
            except Exception as e:
                logger.error(f"Failed to persist tour to Couchbase: {e}")

        return jsonify({'success': True, 'tour_id': tour_id})
    except Exception as e:
        logger.error(f"Create tour error: {e}")
        return jsonify({'success': False, 'error': 'Failed to create tour'}), 500

@app.route('/api/tours', methods=['GET'])
def list_tours():
    """List tour requests for the current session or for a buyer if buyer_name is provided."""
    try:
        buyer_name = (request.args.get('buyer_name') or '').strip()
        if buyer_name:
            cluster = get_cb_cluster()
            collection_path = get_profiles_path()
            buyer_key = f"buyer::{buyer_name.lower()}"
            res = cluster.query(
                f"SELECT b.tours AS tours FROM {collection_path} AS b USE KEYS $buyer_key",
                QueryOptions(named_parameters={'buyer_key': buyer_key})
            ).execute()
            rows = list(res)
            tours = rows[0].get('tours', []) if rows else []
            if tours is None:
                tours = []
            return jsonify({'success': True, 'tours': tours})
        # Default: session-based
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        tours = list(tour_requests.get(session_id, {}).values())
        return jsonify({'success': True, 'tours': tours})
    except Exception as e:
        logger.error(f"List tours error: {e}")
        return jsonify({'success': False, 'tours': []}), 500

@app.route('/api/tours/<tour_id>', methods=['GET'])
def get_tour(tour_id):
    """Get a specific tour by ID for the current session, or for a buyer if buyer_name is provided."""
    try:
        buyer_name = (request.args.get('buyer_name') or '').strip()
        if buyer_name:
            cluster = get_cb_cluster()
            collection_path = get_profiles_path()
            buyer_key = f"buyer::{buyer_name.lower()}"
            res = cluster.query(
                f"""
                SELECT FIRST t FOR t IN b.tours WHEN t.id = $tour_id END AS tour
                FROM {collection_path} AS b USE KEYS $buyer_key
                """,
                QueryOptions(named_parameters={'buyer_key': buyer_key, 'tour_id': tour_id})
            ).execute()
            rows = list(res)
            tour = rows[0].get('tour') if rows else None
            if not tour:
                return jsonify({'success': False, 'error': 'Tour not found'}), 404
            return jsonify({'success': True, 'tour': tour})
        # Default: session-based
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        tour = tour_requests.get(session_id, {}).get(tour_id)
        if not tour:
            return jsonify({'success': False, 'error': 'Tour not found'}), 404
        return jsonify({'success': True, 'tour': tour})
    except Exception as e:
        logger.error(f"Get tour error: {e}")
        return jsonify({'success': False, 'error': 'Failed to load tour'}), 500

@app.route('/api/tours/<tour_id>', methods=['PATCH'])
def update_tour(tour_id):
    """Update tour status or times. Payload can include status, confirmed_time, and/or preferred_time.
    If the tour belongs to a buyer, also update in Couchbase.
    """
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        tour = tour_requests.get(session_id, {}).get(tour_id)
        # It may be only in Couchbase if created on a different session; allow updates by buyer_name
        buyer_name = (request.args.get('buyer_name') or (tour or {}).get('buyer_name') or '').strip()
        data = request.get_json() or {}
        status = (data.get('status') or '').strip().lower()
        confirmed_time = (data.get('confirmed_time') or '').strip()
        preferred_time = (data.get('preferred_time') or '').strip()
        allowed = {'pending', 'accepted', 'confirmed', 'declined', 'cancelled'}
        # Update in-memory if present
        if tour:
            if status and status in allowed:
                tour['status'] = status
            if confirmed_time:
                tour['confirmed_time'] = confirmed_time
            if preferred_time:
                tour['preferred_time'] = preferred_time
            tour['updated_at'] = datetime.now().isoformat()

        # Update in Couchbase if buyer_name provided
        updated_from_cb = None
        if buyer_name:
            try:
                cluster = get_cb_cluster()
                collection_path = get_profiles_path()
                buyer_key = f"buyer::{buyer_name.lower()}"
                cluster.query(
                    f"""
                    UPDATE {collection_path} AS b
                    SET b.tours = ARRAY CASE WHEN t.id = $tour_id THEN OBJECT_PUT(OBJECT_PUT(OBJECT_PUT(t, 'status', CASE WHEN $status_ok THEN $status ELSE t.status END), 'confirmed_time', CASE WHEN $has_confirmed THEN $confirmed ELSE t.confirmed_time END), 'preferred_time', CASE WHEN $has_preferred THEN $preferred ELSE t.preferred_time END) ELSE t END FOR t IN b.tours END
                    WHERE META(b).id = $buyer_key
                    RETURNING FIRST t FOR t IN b.tours WHEN t.id = $tour_id END AS tour
                    """,
                    QueryOptions(named_parameters={
                        'buyer_key': buyer_key,
                        'tour_id': tour_id,
                        'status': status,
                        'status_ok': status in allowed if status else False,
                        'confirmed': confirmed_time,
                        'has_confirmed': bool(confirmed_time),
                        'preferred': preferred_time,
                        'has_preferred': bool(preferred_time),
                    })
                ).execute()
                # Fetch updated tour to return
                res = cluster.query(
                    f"SELECT FIRST t FOR t IN b.tours WHEN t.id = $tour_id END AS tour FROM {collection_path} AS b USE KEYS $buyer_key",
                    QueryOptions(named_parameters={'buyer_key': buyer_key, 'tour_id': tour_id})
                ).execute()
                rows = list(res)
                updated_from_cb = rows[0].get('tour') if rows else None
            except Exception as e:
                logger.error(f"Failed to update tour in Couchbase: {e}")
        final_tour = updated_from_cb or tour
        if not final_tour:
            return jsonify({'success': False, 'error': 'Tour not found'}), 404
        return jsonify({'success': True, 'tour': final_tour})
    except Exception as e:
        logger.error(f"Update tour error: {e}")
        return jsonify({'success': False, 'error': 'Failed to update tour'}), 500

# ------------------- Nearby Places API -------------------

def _get_or_create_agent(session_id: str) -> LangGraphRealEstateAgent:
    if session_id not in agents:
        agents[session_id] = LangGraphRealEstateAgent()
    return agents[session_id]

@app.route('/api/nearby', methods=['GET'])
def nearby():
    """Return nearby schools and restaurants for a given lat/lon or address.
    Query params: lat, lon, address
    """
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        address = (request.args.get('address') or '').strip()
        agent = _get_or_create_agent(session_id)
        if (lat is None or lon is None) and address:
            coords = agent._geocode_address(address)
            lat, lon = coords.get('lat'), coords.get('lon')
        schools = agent._nearby_places(lat, lon, 'school', radius=4000, min_rating=4.0, max_results=5) if lat is not None and lon is not None else []
        restaurants = agent._nearby_places(lat, lon, 'restaurant', radius=3000, min_rating=4.2, max_results=5) if lat is not None and lon is not None else []
        return jsonify({'success': True, 'schools': schools, 'restaurants': restaurants, 'lat': lat, 'lon': lon})
    except Exception as e:
        logger.error(f"Nearby places error: {e}")
        return jsonify({'success': False, 'schools': [], 'restaurants': []}), 500

@app.route('/api/tours/<tour_id>', methods=['DELETE'])
def delete_tour(tour_id):
    """Delete a tour by ID from session storage and, if buyer_name provided, from Couchbase buyer doc."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session not found'}), 400
        # Remove from session in-memory store
        if session_id in tour_requests and tour_id in tour_requests[session_id]:
            del tour_requests[session_id][tour_id]
        # Optionally remove from Couchbase if buyer_name provided
        buyer_name = (request.args.get('buyer_name') or '').strip()
        if buyer_name:
            try:
                cluster = get_cb_cluster()
                collection_path = get_profiles_path()
                buyer_key = f"buyer::{buyer_name.lower()}"
                cluster.query(
                    f"""
                    UPDATE {collection_path} AS b
                    SET b.tours = ARRAY t FOR t IN b.tours WHEN t.id != $tour_id END
                    WHERE META(b).id = $buyer_key
                    RETURNING 1
                    """,
                    QueryOptions(named_parameters={'buyer_key': buyer_key, 'tour_id': tour_id})
                ).execute()
            except Exception as e:
                logger.error(f"Failed to delete tour from Couchbase: {e}")
        return jsonify({'success': True, 'deleted': True})
    except Exception as e:
        logger.error(f"Delete tour error: {e}")
        return jsonify({'success': False, 'error': 'Failed to delete tour'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)