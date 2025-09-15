import os
import sys
import importlib.util
import types
import pytest

# ---------------------------------------------------------------------------
# Lightweight stubs for external dependencies so we can load agent.py without
# installing heavy or unavailable packages in the unit test environment.
# ---------------------------------------------------------------------------

def ensure_stub(name, attrs: dict):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod

# langgraph.graph stubs
class _StubStateGraph:
    def __init__(self, *_a, **_k): pass
    def add_node(self, *a, **k): pass
    def add_edge(self, *a, **k): pass
    def add_conditional_edges(self, *a, **k): pass
    def compile(self):
        class _G:
            def invoke(self, state): return state
        return _G()

ensure_stub('langgraph', {})
ensure_stub('langgraph.graph', {
    'StateGraph': _StubStateGraph,
    'START': 'START',
    'END': 'END'
})

# langchain_aws stubs
class _StubBedrockEmbeddings:
    def __init__(self, *a, **k): pass
class _StubChatBedrock:
    def __init__(self, *a, **k): pass
    def invoke(self, payload):
        class R: pass
        r = R(); r.content = "PRIMARY RESPONSE"
        return r
ensure_stub('langchain_aws', {
    'BedrockEmbeddings': _StubBedrockEmbeddings,
    'ChatBedrock': _StubChatBedrock
})

# langchain_openai stub (only used for fallback instantiation; we'll override anyway)
class _StubChatOpenAI:
    def __init__(self, *a, **k): pass
    def invoke(self, payload):
        class R: pass
        r = R(); r.content = "OPENAI RESPONSE"
        return r
ensure_stub('langchain_openai', {'ChatOpenAI': _StubChatOpenAI})

# langchain_couchbase, langchain_tavily stubs
class _StubVectorStore: pass
class _StubTavily:
    def __init__(self, *a, **k): pass
    def invoke(self, q): return []
ensure_stub('langchain_couchbase.vectorstores', {'CouchbaseSearchVectorStore': _StubVectorStore})
ensure_stub('langchain_tavily', {'TavilySearch': _StubTavily})

# couchbase stubs
class _StubAuth: pass
class _StubCluster:
    def __init__(self, *a, **k): pass
    def wait_until_ready(self, *a, **k): pass
ensure_stub('couchbase.auth', {'PasswordAuthenticator': _StubAuth})
ensure_stub('couchbase.cluster', {'Cluster': _StubCluster})
ensure_stub('couchbase.options', {'ClusterOptions': lambda x: x, 'QueryOptions': object})

# dotenv stub
def _stub_load_dotenv(*a, **k): return True
ensure_stub('dotenv', {'load_dotenv': _stub_load_dotenv})

# ---------------------------------------------------------------------------
# Dynamically load agent module
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AGENT_PATH = os.path.join(PROJECT_ROOT, 'real-estate-agent', 'agent.py')
spec = importlib.util.spec_from_file_location('agent_module', AGENT_PATH)
agent_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = agent_module
assert spec.loader is not None
spec.loader.exec_module(agent_module)  # type: ignore

LangGraphRealEstateAgent = agent_module.LangGraphRealEstateAgent


class DummyPrimaryLLM:
    """Primary LLM stub that can be toggled to fail or succeed."""
    def __init__(self, should_fail=True):
        self.should_fail = should_fail
        self.invocations = 0

    def invoke(self, payload):
        self.invocations += 1
        if self.should_fail:
            raise RuntimeError("Simulated Bedrock failure")
        # Return minimal object with .content
        class R: pass
        r = R()
        r.content = "PRIMARY RESPONSE"
        return r


class DummyFallbackLLM:
    def __init__(self):
        self.invocations = 0

    def invoke(self, payload):
        self.invocations += 1
        class R: pass
        r = R()
        r.content = "FALLBACK RESPONSE"
        return r


class FallbackTestAgent(LangGraphRealEstateAgent):
    """Override heavy setup to keep tests fast and isolated."""
    __test__ = False  # prevent pytest from treating this as a test container
    def setup_tools(self):
        # Provide a placeholder primary llm; tests will overwrite with dummy
        class _TempLLM:
            def invoke(self, payload):
                class R: pass
                r = R(); r.content = "TEMP"
                return r
        self.llm = _TempLLM()
        self._fallback_llm = None
        self._fallback_llm_name = 'gpt-4.1'
    def setup_graph(self):
        self.graph = None

@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return FallbackTestAgent()


def test_fallback_triggered(agent, monkeypatch):
    """If primary LLM raises, fallback should be initialized and used."""
    primary = DummyPrimaryLLM(should_fail=True)
    fallback = DummyFallbackLLM()

    # Inject primary & stub initializer to return our dummy fallback
    agent.llm = primary
    def fake_init_fallback():
        agent._fallback_llm = fallback
        return fallback
    agent._init_fallback_llm = fake_init_fallback

    resp = agent._safe_llm_invoke("Hello")
    assert resp.content == "FALLBACK RESPONSE"
    assert primary.invocations == 1, "Primary should be tried once"
    assert fallback.invocations == 1, "Fallback should be invoked once"


def test_primary_success_no_fallback(agent, monkeypatch):
    """If primary succeeds, fallback should not be touched."""
    primary = DummyPrimaryLLM(should_fail=False)
    agent.llm = primary
    # Ensure fallback init would fail if called (should not be)
    def bad_init():
        raise AssertionError("Fallback should not initialize on success path")
    agent._init_fallback_llm = bad_init

    resp = agent._safe_llm_invoke("Hello")
    assert resp.content == "PRIMARY RESPONSE"
    assert primary.invocations == 1


def test_double_failure(agent, monkeypatch):
    """If both primary and fallback fail, graceful message returned."""
    primary = DummyPrimaryLLM(should_fail=True)
    agent.llm = primary

    class BadFallback:
        def invoke(self, payload):
            raise RuntimeError("Fallback also failed")

    def fake_init_fallback():
        agent._fallback_llm = BadFallback()
        return agent._fallback_llm

    agent._init_fallback_llm = fake_init_fallback
    resp = agent._safe_llm_invoke("Hello")
    assert "both primary and backup" in resp.content.lower() or "unavailable" in resp.content.lower()
