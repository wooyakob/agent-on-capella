import os
import sys
import types
import importlib.util
import pytest

# Reuse stub strategy from fallback test (minimal subset)
def ensure_stub(name, attrs: dict):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod

class _StubStateGraph:
    def __init__(self, *_a, **_k): pass
    def add_node(self, *a, **k): pass
    def add_edge(self, *a, **k): pass
    def add_conditional_edges(self, *a, **k): pass
    def compile(self):
        class G: 
            def invoke(self, s): return s
        return G()

ensure_stub('langgraph', {})
ensure_stub('langgraph.graph', {'StateGraph': _StubStateGraph, 'START': 'START', 'END': 'END'})

class _StubChatBedrock:
    def __init__(self, *a, **k): pass
    def invoke(self, payload):
        # Intent emulation: attempt to extract the "Current User Query" line if present
        if isinstance(payload, list):
            raw = payload[-1]['content']
            text_lower = raw.lower()
            user_query = text_lower
            marker = 'current user query:'
            if marker in text_lower:
                # Extract substring after marker up to newline
                after = text_lower.split(marker, 1)[1].strip()
                user_query = after.split('\n', 1)[0].strip('"')
            class R: pass
            r = R()
            # Heuristics tuned to reduce false MARKET classification
            if any(k in user_query for k in ['market', 'prices', 'trends']) and not any(p in user_query for p in ['find', 'looking for', 'bedroom']):
                r.content = 'MARKET_SEARCH'
            elif any(k in user_query for k in ['school', 'schools', 'restaurant', 'restaurants']):
                r.content = 'LOCATION_CONTEXT'
            elif any(k in user_query for k in ['find', 'looking for', 'bedroom', 'home', 'house']):
                r.content = 'PROPERTY_SEARCH'
            else:
                r.content = 'GENERAL_CHAT'
            return r
        class R: pass
        r = R(); r.content = "GENERAL_CHAT"; return r

class _StubEmbeddings: ...
class _StubVectorStore:
    def __init__(self, *a, **k): pass
    def similarity_search_with_score(self, query, k=5):
        # Return synthetic docs (tuples of doc-like and score)
        class Doc:
            def __init__(self, meta, content):
                self.metadata = meta
                self.page_content = content
        docs = []
        for i in range(k):
            docs.append((Doc({
                'name': f'Property {i+1}',
                'price': 700000 + i*10000,
                'address': f'123{i} San Diego, CA 9212{i}',
                'bedrooms': 3,
                'bathrooms': 2,
                'house_sqft': 1600 + i*50,
                'geo': {'lat': 32.8 + i*0.01, 'lon': -117.1 - i*0.01}
            }, f"Nice home with features {i}"), 0.85 - i*0.01))
        return docs

class _StubTavily:
    def __init__(self, *a, **k): pass
    def invoke(self, q):
        return {'results': [
            {'title': 'Market Update', 'url': 'https://example.com/market', 'content': 'Median prices up 5% inventory tightening.'}
        ]}

ensure_stub('langchain_aws', {'ChatBedrock': _StubChatBedrock, 'BedrockEmbeddings': _StubEmbeddings})
ensure_stub('langchain_openai', {'ChatOpenAI': _StubChatBedrock})
ensure_stub('langchain_couchbase.vectorstores', {'CouchbaseSearchVectorStore': _StubVectorStore})
ensure_stub('langchain_tavily', {'TavilySearch': _StubTavily})
ensure_stub('couchbase.auth', {'PasswordAuthenticator': object})
class _StubCluster: 
    def __init__(self,*a,**k): pass
    def wait_until_ready(self,*a,**k): pass
ensure_stub('couchbase.cluster', {'Cluster': _StubCluster})
ensure_stub('couchbase.options', {'ClusterOptions': lambda x: x, 'QueryOptions': object})
ensure_stub('dotenv', {'load_dotenv': lambda *a, **k: True})

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AGENT_PATH = os.path.join(PROJECT_ROOT, 'real-estate-agent', 'agent.py')
spec = importlib.util.spec_from_file_location('agent_module2', AGENT_PATH)
agent_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = agent_module
spec.loader.exec_module(agent_module)  # type: ignore
AgentClass = agent_module.LangGraphRealEstateAgent

class TestHarnessAgent(AgentClass):
    __test__ = False
    def setup_tools(self):
        # Override to inject stubs directly
        self.llm = _StubChatBedrock()
        self.embeddings = _StubEmbeddings()
        self.vector_store = _StubVectorStore(None, None, None, None, None)
        self.tavily_search = _StubTavily()
        self.gmaps_key = None
    def setup_graph(self):
        # Skip full graph since we will call nodes directly
        self.graph = None

@pytest.fixture
def agent():
    return TestHarnessAgent()

def make_state(query: str, buyer=None):
    return {
        'messages': [{'role': 'user', 'content': query}],
        'user_query': query,
        'search_results': [],
        'buyer_profile': buyer or {},
        'next_action': '',
        'formatted_response': '',
        'saved_properties': []
    }

def test_route_property_search(agent):
    state = make_state("I'm looking for a 3 bedroom home in San Diego under 800k")
    state = agent.analyze_user_intent(state)
    assert state['next_action'] == 'property_search'
    state = agent.property_search_node(state)
    assert state['search_results'], 'Expected synthetic properties returned'

def test_route_market_search_deterministic(agent):
    state = make_state("What are current market trends and prices in San Diego?")
    state = agent.analyze_user_intent(state)
    assert state['next_action'] == 'market_search'
    state = agent.market_search_node(state)
    assert state['search_results'], 'Market results should be attached'

def test_route_location_context(agent):
    # Query emphasizing amenities
    state = make_state("Are there good schools and restaurants near these properties?", buyer={'location': 'San Diego CA'})
    # Need prior properties for location context node; simulate property search first
    ps_state = make_state("Find a family home", buyer={'location': 'San Diego CA'})
    ps_state = agent.analyze_user_intent(ps_state)
    ps_state = agent.property_search_node(ps_state)
    # Transfer last_properties into new state scenario
    state['search_results'] = ps_state['search_results']
    state = agent.location_context_node(state)
    assert state['search_results'], 'Should enrich with nearby info (even if empty lists)'

def test_general_chat_fallback(agent):
    state = make_state("Tell me about the buying process")
    state = agent.analyze_user_intent(state)
    assert state['next_action'] == 'general_chat'
    state = agent.general_chat_node(state)
    assert state['search_results'] and state['search_results'][0]['type'] == 'chat_response'

def test_property_strict_location_filter(agent):
    buyer = {'location': 'San Diego CA', 'budget': {'max': 900000}}
    state = make_state("Find me a 3 bedroom house", buyer=buyer)
    state = agent.analyze_user_intent(state)
    state = agent.property_search_node(state)
    # All synthetic addresses include San Diego, CA; ensure not flagged as no local matches
    assert not state.get('_no_local_matches'), 'Should have local matches'
    assert state['search_results'], 'Expected filtered properties present'


def test_formatting_node_fallback(agent, monkeypatch):
    """Force formatting path to raise and ensure fallback formatting kicks in with clean sentence end."""
    # Prepare a property_search scenario with results
    buyer = {'location': 'San Diego CA'}
    state = make_state("Find a coastal home", buyer=buyer)
    state = agent.analyze_user_intent(state)
    state = agent.property_search_node(state)
    # Monkeypatch safe invoke to raise during formatting only
    original_safe = agent._safe_llm_invoke
    calls = {'count': 0}
    def failing_safe(payload):
        # First call inside format_response_node will raise
        calls['count'] += 1
        raise RuntimeError("Simulated formatting failure")
    agent._safe_llm_invoke = failing_safe
    # Run formatting
    formatted = agent.format_response_node({**state})
    # Restore
    agent._safe_llm_invoke = original_safe
    # Assert fallback text produced (should start with house icon or 'Found these')
    resp = formatted['formatted_response']
    assert resp, 'Should produce a fallback formatted response'
    # Ensure sentence completion helper applied (ends with punctuation or newline after punctuation)
    assert resp.rstrip()[-1] in '.!?', 'Fallback response should end cleanly'
