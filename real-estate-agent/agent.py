import os
import json
import logging
import traceback
import re
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from datetime import timedelta
import math
from urllib.parse import quote_plus, urlparse
import requests

from langgraph.graph import StateGraph, START, END
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_openai import ChatOpenAI  # Fallback LLM
from langchain_openai import OpenAIEmbeddings  # Fallback Embeddings
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_tavily import TavilySearch

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.options import QueryOptions

from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealEstateAgentState(TypedDict):
    messages: List[Dict[str, Any]]
    user_query: str                 
    search_results: List[Dict[str, Any]]  
    buyer_profile: Dict[str, Any]  
    next_action: str                
    formatted_response: str
    saved_properties: List[Dict[str, Any]]          

class LangGraphRealEstateAgent:
    """
    A conversational real estate agent that uses LangGraph to orchestrate 
    multiple tools: Couchbase vector search, Tavily web search, and Bedrock LLM.
    """
    
    def __init__(self):
        """Initialize the LangGraph Real Estate Agent."""
        self.conversation_history = []
        self.last_properties: List[Dict[str, Any]] = []
        self.setup_tools()
        self.setup_graph()
        
    def setup_tools(self):
        """Set up all tools: LLM, embeddings, Couchbase, and Tavily."""
        try:
            # Primary (Bedrock) LLM
            self.llm = ChatBedrock(
                model_id=("us.meta.llama4-maverick-17b-instruct-v1:0"),
                region_name=("us-east-2"),
                temperature=0.7
            )
            self._fallback_llm = None  # Lazy init
            self._fallback_llm_name = os.getenv("FALLBACK_LLM")
            
            # Prefer Bedrock Titan embeddings with OpenAI fallback (dim 1024 to match index)
            self.embeddings = self._init_embeddings()

            auth = PasswordAuthenticator(os.getenv("CB_USERNAME"), os.getenv("CB_PASSWORD"))
            options = ClusterOptions(auth)
            connstr = os.getenv("CB_HOSTNAME")
            cluster = Cluster(connstr, options)
            cluster.wait_until_ready(timedelta(seconds=5))
            self.cluster = cluster
            
            BUCKET = os.getenv("CB_BUCKET", "properties")
            SCOPE = os.getenv("CB_SCOPE", "2025-listings")
            COLLECTION = os.getenv("CB_COLLECTION", "united-states")
            SEARCH_INDEX = os.getenv("CB_SEARCH_INDEX", "properties-index")
            
            self.vector_store = CouchbaseSearchVectorStore(
                cluster=cluster,
                bucket_name=BUCKET,
                scope_name=SCOPE,
                collection_name=COLLECTION,
                embedding=self.embeddings,
                index_name=SEARCH_INDEX,
                text_key="description",
                embedding_key="embedding",
            )
            
            self.tavily_search = TavilySearch(max_results=3)
            # Google Maps Platform API key
            self.gmaps_key = os.getenv("GMAPS_API_KEY")
            
            logger.info("All tools initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise

    # ---------------- Fallback / Resilience Helpers ----------------
    class _SimpleLLMResponse:
        def __init__(self, content: str):
            self.content = content

    def _init_fallback_llm(self):
        """Lazily initialize the OpenAI fallback LLM if available."""
        if self._fallback_llm is not None:
            return self._fallback_llm
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            logger.warning("OPENAI_API_KEY not set; cannot initialize fallback LLM")
            return None
        try:
            model_name = self._fallback_llm_name or "gpt-4o-mini"
            self._fallback_llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
            )
            logger.info(f"Initialized fallback OpenAI LLM: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize fallback LLM: {e}")
            self._fallback_llm = None
        return self._fallback_llm

    def _init_embeddings(self):
        """Initialize embeddings with Bedrock Titan primary and OpenAI fallback.
        Returns an embeddings object compatible with LangChain's embed_query/documents APIs.
        OpenAI fallback uses text-embedding-3-small with dimensions matching the Couchbase index (1024).
        """
        # Small wrapper to transparently fall back on runtime errors
        class ResilientEmbeddings:
            def __init__(self, primary, fallback, log):
                self._primary = primary
                self._fallback = fallback
                self._using_fallback = False
                self._log = log

            def _ensure_provider(self):
                # If primary failed previously, keep using fallback
                if self._using_fallback:
                    return self._fallback
                return self._primary or self._fallback

            def _switch_to_fallback(self, err: Exception):
                if not self._fallback:
                    raise err
                self._log.warning(f"Primary embeddings failed; switching to OpenAI fallback. Error: {err}")
                self._using_fallback = True
                return self._fallback

            def embed_query(self, text: str):
                provider = self._ensure_provider()
                try:
                    return provider.embed_query(text)
                except Exception as e:
                    provider = self._switch_to_fallback(e)
                    return provider.embed_query(text)

            def embed_documents(self, texts):
                provider = self._ensure_provider()
                try:
                    return provider.embed_documents(texts)
                except Exception as e:
                    provider = self._switch_to_fallback(e)
                    return provider.embed_documents(texts)

        # Try to construct providers
        primary = None
        try:
            region = os.getenv("AWS_REGION", "us-east-2")
            titan_model = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
            logger.info(f"Initializing Bedrock embeddings: model={titan_model} region={region}")
            primary = BedrockEmbeddings(model_id=titan_model, region_name=region)
        except Exception as e:
            logger.warning(f"Bedrock embeddings init failed: {e}")

        fallback = None
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                oa_model = (
                    os.getenv("FALLBACK_EMBEDDING_MODEL")
                    or os.getenv("OPENAI_EMBEDDING_MODEL")
                    or "text-embedding-3-small"
                )
                # Prefer explicit fallback dims env var, else OPENAI_EMBEDDING_DIMENSIONS, else 1024
                oa_dims_env = (
                    os.getenv("FALLBACK_EMBEDDING_DIMENSIONS")
                    or os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
                    or ""
                )
                oa_dims = int(oa_dims_env) if oa_dims_env.strip() else 1024
                logger.info(f"Initializing OpenAI embeddings: model={oa_model} dims={oa_dims}")
                fallback = OpenAIEmbeddings(model=oa_model, dimensions=oa_dims)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")

        if primary and fallback:
            return ResilientEmbeddings(primary, fallback, logger)
        if primary and not fallback:
            # No fallback available; return primary (may error later)
            return primary
        if fallback:
            # Bedrock unavailable at init; use OpenAI directly
            return fallback
        # Neither available
        raise RuntimeError("No embeddings available: neither Bedrock nor OpenAI could be initialized")

    def _safe_llm_invoke(self, payload):
        """Invoke primary LLM; on failure, fall back to OpenAI if configured.
        payload can be either a raw string (treated as user message) or a list of role/content dicts.
        Returns an object with a .content attribute.
        """
        try:
            return self.llm.invoke(payload)
        except Exception as primary_err:
            logger.warning(f"Primary LLM failed, attempting fallback. Error: {primary_err}")
            fb = self._init_fallback_llm()
            if fb is None:
                logger.error("Fallback LLM unavailable; returning graceful error message")
                return self._SimpleLLMResponse("I'm sorry, I'm having temporary trouble reaching the model. Could you try again in a moment?")
            try:
                # If payload is a plain string, wrap it
                if isinstance(payload, str):
                    payload = [{"role": "user", "content": payload}]
                return fb.invoke(payload)
            except Exception as fallback_err:
                logger.error(f"Fallback LLM also failed: {fallback_err}")
                return self._SimpleLLMResponse("I'm sorry, both primary and backup models are unavailable right now. Please try again shortly.")
    
    def couchbase_property_search(self, query: str, k: int = 5, saved_properties: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search properties in Couchbase using vector similarity, excluding saved properties"""
        try:
            # Get more results initially to account for filtering out saved properties
            search_k = max(k * 2, 10) if saved_properties else k
            results = self.vector_store.similarity_search_with_score(query, k=search_k)
            
            # Build set of saved property keys for fast lookup
            saved_keys = set()
            if saved_properties:
                for saved_prop in saved_properties:
                    # Generate same deduplication key as in API
                    addr = (saved_prop.get('address') or (saved_prop.get('location') or {}).get('address') or '').strip().lower()
                    name = (saved_prop.get('name') or saved_prop.get('title') or '').strip().lower()
                    price = saved_prop.get('price')
                    
                    if saved_prop.get('id'):
                        saved_keys.add(saved_prop.get('id'))
                    elif addr and name:
                        saved_keys.add(f"{addr}|{name}")
                    elif addr:
                        saved_keys.add(addr)
                    elif name and price is not None:
                        saved_keys.add(f"{name}|{price}")
            
            properties = []
            for doc, score in results:
                property_data = {
                    "name": doc.metadata.get('name', 'Unknown Property'),
                    "price": doc.metadata.get('price', 'Price not available'),
                    "address": doc.metadata.get('address', 'Address not available'),
                    "bedrooms": doc.metadata.get('bedrooms', 'N/A'),
                    "bathrooms": doc.metadata.get('bathrooms', 'N/A'),
                    "house_sqft": doc.metadata.get('house_sqft', 'N/A'),
                    "geo": {
                        "lat": (doc.metadata.get('geo', {}) or {}).get('lat'),
                        "lon": (doc.metadata.get('geo', {}) or {}).get('lon')
                    },
                    "description": doc.page_content,
                    "similarity_score": round(score, 3)
                }
                
                # Check if this property is already saved
                if saved_keys:
                    prop_addr = (property_data.get('address') or '').strip().lower()
                    prop_name = (property_data.get('name') or '').strip().lower()
                    prop_price = property_data.get('price')
                    prop_id = doc.metadata.get('id')
                    
                    # Generate same deduplication key for comparison
                    if prop_id and prop_id in saved_keys:
                        continue  # Skip this saved property
                    elif prop_addr and prop_name and f"{prop_addr}|{prop_name}" in saved_keys:
                        continue  # Skip this saved property
                    elif prop_addr and prop_addr in saved_keys:
                        continue  # Skip this saved property
                    elif prop_name and prop_price is not None and f"{prop_name}|{prop_price}" in saved_keys:
                        continue  # Skip this saved property
                
                properties.append(property_data)
                
                # Stop when we have enough non-saved properties
                if len(properties) >= k:
                    break
            
            return properties
        except Exception as e:
            logger.error(f"Couchbase search error: {e}")
            return []

    # --- Response post-processing helpers ---
    def _ensure_complete_sentence(self, text: str) -> str:
        """Ensure the response does not end mid-sentence.
        If the last line is cut off (no terminal punctuation), trim to previous sentence.
        """
        if not text:
            return text
        trimmed = text.rstrip()
        if trimmed.endswith(('.', '!', '?')):
            return trimmed
        # Attempt to find last terminal punctuation before potential truncation
        matches = list(re.finditer(r'[\.?!]', trimmed))
        if matches:
            last = matches[-1].end()
            # Keep everything up to last punctuation and ensure single newline
            return trimmed[:last].rstrip() + "\n"
        # Fallback: return original trimmed text (avoid overly aggressive deletion)
        return trimmed + ("." if not trimmed.endswith('.') else "")

    def _parse_price(self, price_val: Any) -> float:
        """Attempt to parse price into a float (USD). Handles numbers and strings like "$1,234,567".
        Returns float('inf') if unparseable so it's filtered out by max budget.
        """
        try:
            if isinstance(price_val, (int, float)):
                return float(price_val)
            if isinstance(price_val, str):
                cleaned = price_val.replace('$', '').replace(',', '').strip()
                return float(cleaned)
        except (ValueError, TypeError):
            pass
        return float('inf')

    # --- Google Maps helper functions ---
    def _reverse_geocode(self, lat: float, lon: float) -> Dict[str, Any]:
        """Reverse geocode coordinates to human-readable address and embeddable map URL."""
        if lat is None or lon is None:
            return {"address": None, "mapsLink": None}
        # Always provide a safe maps link that doesn't expose API keys
        safe_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        # If no server-side key, we can't reverse geocode the address but can still return a safe link
        if not self.gmaps_key:
            return {"address": None, "mapsLink": safe_link}
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={self.gmaps_key}"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    address = data["results"][0].get("formatted_address")
                    # Provide safe maps search link, not an embed URL with key
                    return {"address": address, "mapsLink": safe_link}
        except Exception as e:
            logger.debug(f"Reverse geocode error: {e}")
        return {"address": None, "mapsLink": safe_link}

    def _geocode_address(self, address: str) -> Dict[str, Any]:
        """Forward geocode an address to lat/lon when geo is missing."""
        if not self.gmaps_key or not address:
            return {"lat": None, "lon": None}
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={quote_plus(address)}&key={self.gmaps_key}"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    loc = results[0].get("geometry", {}).get("location", {})
                    return {"lat": loc.get("lat"), "lon": loc.get("lng")}
        except Exception as e:
            logger.debug(f"Forward geocode error: {e}")
        return {"lat": None, "lon": None}

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        if None in [lat1, lon1, lat2, lon2]:
            return None
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _nearby_places(self, lat: float, lon: float, place_type: str, radius: int = 3000, min_rating: float = 4.0, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find nearby places of a given type using Google Places Nearby Search API."""
        if not self.gmaps_key or lat is None or lon is None:
            return []
        try:
            url = (
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                f"?location={lat},{lon}&radius={radius}&type={place_type}&key={self.gmaps_key}"
            )
            resp = requests.get(url, timeout=8)
            results = []
            if resp.status_code == 200:
                data = resp.json()
                for r in data.get("results", []):
                    rating = r.get("rating")
                    loc = (r.get("geometry", {}).get("location") or {})
                    types = r.get("types", []) or []
                    dist_km = self._haversine_km(lat, lon, loc.get("lat"), loc.get("lng"))
                    # Enforce strict type filtering to avoid gyms/centers appearing as schools, etc.
                    if place_type == "school" and "school" not in types:
                        continue
                    if place_type == "restaurant" and "restaurant" not in types:
                        continue
                    if rating is None or (min_rating is not None and rating < min_rating):
                        continue
                    results.append({
                        "name": r.get("name"),
                        "rating": rating,
                        "user_ratings_total": r.get("user_ratings_total"),
                        "address": r.get("vicinity"),
                        "distance_km": round(dist_km, 2) if dist_km is not None else None,
                        "types": types,
                        "place_id": r.get("place_id")
                    })
            # sort by distance then rating desc
            results.sort(key=lambda x: (x["distance_km"] if x["distance_km"] is not None else 1e9, -float(x["rating"] or 0)))
            return results[:max_results]
        except Exception as e:
            logger.debug(f"Nearby places error: {e}")
            return []
    
    def tavily_market_search(self, query: str, location: str | None = None) -> List[Dict[str, Any]]:
        """Search market information using Tavily.
        If a location is provided (e.g., from buyer profile), prefer it in search variations and fallbacks.
        """
        try:
            # Include location context in the optimization prompt if available
            location_context = f"Location context: {location}" if location else "No specific location provided"
            
            optimization_prompt = f"""
            Create optimized search queries for finding real estate market information.
            Original user query: "{query}"
            {location_context}
            
            Generate 3 different focused search queries that would find current market data, price trends, 
            and housing market information. Each should be concise (under 15 words) and target different aspects:
            1. Current market prices/trends
            2. Average home prices  
            3. Real estate market analysis
            
            If a location is provided, ALWAYS include it in each search query to get location-specific results.
            Return each query on a separate line, nothing else.
            """
            optimized_queries = self._safe_llm_invoke(optimization_prompt)
            if hasattr(optimized_queries, 'content'):
                optimized_queries = optimized_queries.content
            optimized_queries = str(optimized_queries).strip().split('\n')
            logger.info(f"Generated {len(optimized_queries)} search queries")
            
            all_results = []

            # Determine if the user asked about a specific property type (e.g., condos)
            ql = (query or "").lower()
            wants_condo = any(k in ql for k in ["condo", "condos", "condominium", "condominiums"]) 
            wants_townhome = any(k in ql for k in ["townhome", "town house", "townhouse"]) 
            wants_single_family = any(k in ql for k in ["single family", "sfh", "detached"]) 

            def add_type_variants(base: str) -> List[str]:
                variants = [base]
                if wants_condo:
                    variants.append(f"condo {base}")
                    variants.append(f"condos {base}")
                if wants_townhome:
                    variants.append(f"townhome {base}")
                    variants.append(f"townhouse {base}")
                if wants_single_family:
                    variants.append(f"single family {base}")
                return variants
            
            for i, opt_query in enumerate(optimized_queries[:3]): 
                try:
                    clean_query = opt_query.strip().lstrip('123456789.-').strip()
                    logger.info(f"Searching with query: {clean_query}")
                    
                    # Prefer profile location if provided; otherwise fall back to generic variations
                    loc = (location or '').strip()
                    if loc:
                        base_vars = [
                            f"{loc} {clean_query}",
                            f"{loc} {clean_query} 2025",
                            f"current {loc} {clean_query}",
                            clean_query  # fallback without location
                        ]
                    else:
                        base_vars = [
                            clean_query,
                            f"{clean_query} 2025",
                            f"current {clean_query}"
                        ]
                    # Expand with property-type variants if applicable
                    search_variations = []
                    for b in base_vars:
                        search_variations.extend(add_type_variants(b))
                    
                    for search_query in search_variations:
                        try:
                            results = self.tavily_search.invoke(search_query)
                            
                            if isinstance(results, dict) and 'results' in results:
                                actual_results = results['results']
                            elif isinstance(results, list):
                                actual_results = results
                            else:
                                actual_results = [results] if results else []
                            
                            if actual_results and len(actual_results) > 0:
                                logger.info(f"Found {len(actual_results)} results with query: {search_query}")
                                all_results.extend(actual_results)
                                break 
                        except Exception as e:
                            logger.debug(f"Search variation '{search_query}' failed: {e}")
                            continue
                    
                    if len(all_results) >= 3:  
                        break
                        
                except Exception as e:
                    logger.warning(f"Query {i+1} failed: {e}")
                    continue
            
            seen_urls = set()
            unique_results = []
            for result in all_results:
                if isinstance(result, dict) and 'url' in result:
                    if result['url'] not in seen_urls:
                        seen_urls.add(result['url'])
                        unique_results.append(result)
                else:
                    unique_results.append(result)
            
            # Filter out irrelevant domains and non-housing/market items
            blacklist_domains = [
                "att.com", "forums.att.com", "communityforums.at&t.com", "facebook.com", "twitter.com",
                "pinterest.com", "reddit.com", "youtube.com"
            ]
            require_keywords = ["real estate", "housing", "market", "price", "condo", "condos", "homes"]

            def domain_of(url: str) -> str:
                try:
                    return (urlparse(url).netloc or "").lower()
                except Exception as e:
                    logger.debug(f"Failed to parse domain from URL '{url}': {e}")
                    return ""

            def relevant(item: Dict[str, Any]) -> bool:
                title = (item.get('title') or '').lower()
                content = (item.get('content') or '').lower()
                url = (item.get('url') or '')
                dom = domain_of(url)
                if any(bd in dom for bd in blacklist_domains):
                    return False
                text = f"{title} {content}"
                return any(k in text for k in require_keywords)

            filtered = [r for r in unique_results if relevant(r)]
            logger.info(f"Filtered market info: {len(filtered)} relevant out of {len(unique_results)} unique")
            if not filtered and unique_results:
                # Relax filtering: allow items mentioning housing or price without domain blacklist
                logger.info("No strictly relevant results, attempting relaxed filtering...")
                relaxed_keywords = ["housing", "median", "price", "condo", "inventory", "sales", "market"]
                relaxed = []
                for r in unique_results:
                    title = (r.get('title') or '').lower()
                    content = (r.get('content') or '').lower()
                    text = f"{title} {content}"
                    if any(k in text for k in relaxed_keywords):
                        relaxed.append(r)
                if relaxed:
                    logger.info(f"Relaxed filtering recovered {len(relaxed)} items")
                    filtered = relaxed
            # If still empty, try a broader secondary wave of queries
            if not filtered:
                try:
                    secondary_terms = [
                        "housing statistics", "market report", "median home price", "mls report", "housing inventory"
                    ]
                    loc = (location or '').strip()
                    secondary_queries = []
                    for term in secondary_terms:
                        if loc:
                            secondary_queries.append(f"{loc} {term} 2025")
                        secondary_queries.append(f"{term} {loc}".strip())
                    added = []
                    for sq in secondary_queries:
                        try:
                            results = self.tavily_search.invoke(sq)
                            if isinstance(results, dict) and 'results' in results:
                                results = results['results']
                            if not results:
                                continue
                            for r in results:
                                url = (r.get('url') if isinstance(r, dict) else None)
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    added.append(r)
                            if len(added) >= 5:
                                break
                        except Exception as e:
                            logger.debug(f"Secondary query '{sq}' failed: {e}")
                            continue
                    if added:
                        logger.info(f"Secondary search wave added {len(added)} items")
                        filtered = added
                except Exception as e:
                    logger.debug(f"Secondary search wave failed: {e}")
            return filtered[:5]
            
        except Exception as e:
            logger.error(f"Market search error: {e}")
            try:
                logger.info("Trying fallback search...")
                # Fallback prefers provided location if any
                loc = (location or '').strip()
                if loc:
                    fb_query = f"{loc} real estate market trends prices 2025"
                    logger.info(f"Fallback search with location: {fb_query}")
                else:
                    fb_query = "real estate market trends prices"
                    logger.info(f"Fallback search without location: {fb_query}")
                fallback_results = self.tavily_search.invoke(fb_query)
                
                if isinstance(fallback_results, dict) and 'results' in fallback_results:
                    return fallback_results['results'] or []
                elif isinstance(fallback_results, list):
                    return fallback_results
                else:
                    return [fallback_results] if fallback_results else []
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []
    
    def load_buyer_profile(self, buyer_name: str) -> Dict[str, Any]:
        """Load buyer profile from JSON file"""
        try:
            profile_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "data-models", "profiles", "buyers.json"
            )
            with open(profile_path, "r") as f:
                profiles = json.load(f)
                for profile in profiles:
                    if profile.get("buyer", "").lower() == buyer_name.lower():
                        return profile
            return {}
        except Exception as e:
            logger.error(f"Error loading buyer profile: {e}")
            return {}
    
    def analyze_user_intent(self, state: RealEstateAgentState) -> RealEstateAgentState:
        """Use LLM to analyze user intent and determine appropriate tool/action"""
        logger.info("Analyzing user intent with LLM...")
        
        user_query = state.get("user_query", "")
        messages = state.get("messages", [])

        # Deterministic override for MARKET_SEARCH: if query has strong market signals and lacks property search verbs
        ql = user_query.lower()
        market_keywords = ["market", "price", "prices", "trends", "inventory", "median", "dom", "days on market", "appreciation"]
        property_verbs = ["find", "show me", "looking for", "search for", "list", "listings", "want a", "need a"]
        if any(k in ql for k in market_keywords) and not any(v in ql for v in property_verbs):
            logger.info("Deterministic override: MARKET_SEARCH triggered by keywords")
            state["next_action"] = "market_search"
            return state
        
        conversation_context = ""
        if len(messages) > 1:  
            recent_messages = messages[-4:] 
            conversation_context = "\n\nConversation Context:\n"
            for msg in recent_messages[:-1]:  
                role = "User" if msg["role"] == "user" else "Agent"
                conversation_context += f"{role}: {msg['content']}\n"
        
        analysis_prompt = """You are an intelligent real estate agent assistant. Analyze the user's query and determine which tool would be most appropriate:

1. PROPERTY_SEARCH - Use when user describes their dream property, specific requirements, or wants to find properties
   Examples: "I want a 3-bedroom house", "looking for a beachside property", "show me condos under $800k", "find me properties with a pool"

2. MARKET_SEARCH - Use when user asks for current market information, pricing data, trends, or statistics they need to research online
   Examples: "what are San Diego market trends?", "average price for 2-bedroom homes", "how much do houses cost in...", "what's the real estate market like?", "are prices going up?", "market analysis for..."

3. LOCATION_CONTEXT - Use when user asks about neighborhood/location amenities or proximity such as schools, restaurants, safety, commute, or "is this house near a good school/restaurant?"
   Examples: "is this near good schools?", "what restaurants are nearby?", "how's the neighborhood?"

4. GENERAL_CHAT - Use for general real estate advice, process questions, or conversational responses
   Examples: "how do I get pre-approved?", "what's the buying process?", "tell me about real estate", "should I buy or rent?"

Key indicators:
- Market questions often ask about: prices, costs, averages, trends, market conditions, statistics
- Property searches ask about: finding, looking for, showing, specific property features
- Location context asks about: nearby schools, restaurants, commute time, amenities, neighborhood quality
- General chat asks about: processes, advice, how-to questions

{context}
Current User Query: "{query}"

Based on the conversation context and current query, respond with ONLY one word: PROPERTY_SEARCH, MARKET_SEARCH, LOCATION_CONTEXT, or GENERAL_CHAT"""

        try:
            response = self._safe_llm_invoke([
                {"role": "user", "content": analysis_prompt.format(
                    context=conversation_context,
                    query=user_query
                )}
            ])
            
            intent = response.content.strip().upper()
            
            if "PROPERTY_SEARCH" in intent:
                next_action = "property_search"
            elif "MARKET_SEARCH" in intent:
                next_action = "market_search"
            elif "LOCATION_CONTEXT" in intent:
                next_action = "location_context"
            else:
                next_action = "general_chat"
                
            logger.info(f"LLM determined intent: {intent} → {next_action}")
            
        except Exception as e:
            logger.error(f"Intent analysis error: {e}")
            next_action = "general_chat"
        
        state["next_action"] = next_action
        return state
    
    def property_search_node(self, state: RealEstateAgentState) -> RealEstateAgentState:
        """Search for properties using Couchbase vector search based on user's dream property description"""
        logger.info("Searching properties using Couchbase vector search...")
        
        user_query = state.get("user_query", "")
        buyer_profile = state.get("buyer_profile", {})
        saved_properties = state.get("saved_properties", [])
        
        enhancement_prompt = """You are helping enhance a property search query for vector search. 
        
Original Query: "{query}"
Buyer Profile: {profile}

Create an enhanced search query that combines the user's requirements with their profile.
Focus on descriptive terms that would match property descriptions.
Include specific details like location preferences, property types, amenities, and lifestyle needs.

Enhanced Query:"""

        try:
            enhanced_response = self._safe_llm_invoke([
                {"role": "user", "content": enhancement_prompt.format(
                    query=user_query, 
                    profile=json.dumps(buyer_profile, indent=2) if buyer_profile else "No profile available"
                )}
            ])
            
            enhanced_query = enhanced_response.content.strip()
            logger.info(f"Enhanced search query: {enhanced_query}")
            
        except Exception as e:
            logger.error(f"Query enhancement error: {e}")
            enhanced_query = user_query
        
        properties = self.couchbase_property_search(enhanced_query, k=12, saved_properties=saved_properties)
        
        if not properties and enhanced_query != user_query:
            logger.info("No results with enhanced query, trying original...")
            properties = self.couchbase_property_search(user_query, k=12, saved_properties=saved_properties)
        
        # Apply strict budget filtering if present in buyer profile
        try:
            budget_max = (buyer_profile.get('budget') or {}).get('max')
            if budget_max:
                budget_max = float(budget_max)
                filtered = [p for p in properties if self._parse_price(p.get('price')) <= budget_max]
                properties = filtered
        except Exception as e:
            logger.debug(f"Budget filtering skipped due to error: {e}")

        # Strict location filtering: keep only properties within buyer city/state if provided.
        try:
            loc_pref = (buyer_profile or {}).get('location')
            if loc_pref and properties:
                loc_lower = str(loc_pref).lower()
                state_code = None
                tokens = [t.strip(',') for t in str(loc_pref).split()]
                for t in tokens:
                    if len(t) == 2 and t.isalpha():
                        state_code = t.upper()
                        break
                if not state_code:
                    # This hardcoded city-to-state mapping is brittle and has been removed.
                    # For example, "San Antonio" would be incorrectly mapped to "CA".
                    # A more robust geocoding or location parsing solution should be used.
                    pass
                def addr_state(addr: str):
                    m = re.search(r",\s*([A-Z]{2})\s*\d{5}?", str(addr))
                    return m.group(1) if m else None
                city_tokens = [t for t in tokens if len(t) > 2]
                filtered_loc = []
                for p in properties:
                    addr = str(p.get('address',''))
                    addr_low = addr.lower()
                    state_ok = True if not state_code else addr_state(addr) == state_code
                    city_ok = True if not city_tokens else any(ct in addr_low for ct in city_tokens)
                    if state_ok and city_ok:
                        filtered_loc.append(p)
                if filtered_loc:
                    properties = filtered_loc
                else:
                    logger.info("Strict location filter removed all properties; will not substitute remote matches.")
        except Exception as e:
            logger.debug(f"Strict location filtering skipped due to error: {e}")

        # Fallback: filter by state if buyer location mentions a known state (e.g., CA) and geocoding is unavailable
        try:
            loc_pref = (buyer_profile or {}).get('location', '')
            state_hint = None
            # naive parse for two-letter state code present in location
            for token in str(loc_pref).split():
                if len(token) == 2 and token.isalpha():
                    state_hint = token.upper()
                    break
            if not state_hint:
                # quick mapping for common cities
                city_to_state = {"san": "CA", "diego": "CA", "seattle": "WA", "bellevue": "WA"}
                for t in city_to_state:
                    if t in str(loc_pref).lower():
                        state_hint = city_to_state[t]
                        break
            if state_hint:
                def addr_state(addr: str):
                    import re
                    m = re.search(r",\s*([A-Z]{2})\s*\d{5}?", str(addr))
                    return m.group(1) if m else None
                filtered = [p for p in properties if addr_state(p.get('address')) == state_hint]
                if filtered:
                    properties = filtered
        except Exception:
            pass

        # If after strict filtering nothing remains, do NOT show remote properties; prompt user instead.
        if not properties:
            state["search_results"] = []
            state["_no_local_matches"] = True  # ephemeral flag for formatter
        else:
            state["search_results"] = properties
        # Persist for follow-up location questions like "Is this house near good schools?"
        self.last_properties = properties
        logger.info(f"Found {len(properties)} properties")
        
        return state

    def location_context_node(self, state: RealEstateAgentState) -> RealEstateAgentState:
        """Answer questions about nearby amenities using Google Maps around top matched properties."""
        logger.info("Fetching location context (schools, restaurants) using Google Maps...")
        user_query = state.get("user_query", "")
        properties = state.get("search_results", [])
        buyer_profile = state.get("buyer_profile", {})

        # Detect what the user cares about; if none explicitly mentioned, fetch both
        uq = user_query.lower()
        wants_schools = any(w in uq for w in ["school", "schools", "education", "district"]) 
        wants_restaurants = any(w in uq for w in ["restaurant", "food", "dining", "eat", "coffee", "cafe"]) 
        if not (wants_schools or wants_restaurants):
            wants_schools = wants_restaurants = True

        # If a buyer profile location is available, answer relative to that location directly (no properties)
        try:
            loc_pref = (buyer_profile or {}).get('location')
            if loc_pref:
                geo_pref = self._geocode_address(loc_pref)
                blat, blon = geo_pref.get('lat'), geo_pref.get('lon')
                if blat is not None and blon is not None:
                    addr_info = self._reverse_geocode(blat, blon) or {}
                    schools = self._nearby_places(blat, blon, "school", radius=4000, min_rating=4.0, max_results=5) if wants_schools else []
                    restaurants = self._nearby_places(blat, blon, "restaurant", radius=3000, min_rating=4.2, max_results=5) if wants_restaurants else []
                    state["search_results"] = [{
                        "location": {
                            "lat": blat,
                            "lon": blon,
                            "address": addr_info.get("address") or loc_pref,
                            "mapsLink": addr_info.get("mapsLink")
                        },
                        "nearby": {
                            "top_schools": schools,
                            "top_restaurants": restaurants
                        }
                    }]
                    return state
        except Exception as e:
            logger.debug(f"Failed to get location context from profile, falling back. Error: {e}")
            # Fall back to property-based approach if profile-centric fails
            pass

        # If we don't have properties yet, try to find some broadly
        if not properties:
            # Prefer last known properties from a previous result set
            if getattr(self, "last_properties", None):
                properties = self.last_properties
            else:
                fallback_props = self.couchbase_property_search(user_query, k=3)
                properties = fallback_props

        # Prefer properties in buyer's preferred location, if provided
        try:
            loc_pref = (buyer_profile or {}).get('location')
            if loc_pref and properties:
                loc_lower = str(loc_pref).lower()
                addr_matches = [
                    p for p in properties
                    if loc_lower in str(p.get('address', '')).lower()
                    or loc_lower in str((p.get('location') or {}).get('address', '')).lower()
                ]
                if addr_matches:
                    properties = addr_matches
            elif loc_pref and not properties and getattr(self, "last_properties", None):
                loc_lower = str(loc_pref).lower()
                lp = getattr(self, "last_properties", []) or []
                addr_matches = [
                    p for p in lp
                    if loc_lower in str(p.get('address', '')).lower()
                    or loc_lower in str((p.get('location') or {}).get('address', '')).lower()
                ]
                if addr_matches:
                    properties = addr_matches
        except Exception:
            pass

        # If buyer location is provided, attempt a radius filter (<= 50 km)
        try:
            loc_pref = (buyer_profile or {}).get('location')
            if loc_pref:
                # Geocode buyer preferred location
                geo_pref = self._geocode_address(loc_pref)
                blat, blon = geo_pref.get('lat'), geo_pref.get('lon')
                if blat is not None and blon is not None:
                    by_distance = []
                    for p in properties:
                        plat = (p.get('geo') or {}).get('lat')
                        plon = (p.get('geo') or {}).get('lon')
                        if (plat is None or plon is None) and p.get('address'):
                            g = self._geocode_address(p.get('address'))
                            plat, plon = g.get('lat'), g.get('lon')
                        if plat is None or plon is None:
                            # Keep if we cannot determine distance (be conservative)
                            continue
                        d = self._haversine_km(blat, blon, plat, plon)
                        if d is not None and d <= 50:
                            by_distance.append(p)
                    if by_distance:
                        properties = by_distance
        except Exception:
            pass

        # Fallback: filter by state if city/geo filters didn't narrow and location contains a state hint
        try:
            loc_pref = (buyer_profile or {}).get('location', '')
            state_hint = None
            for token in str(loc_pref).split():
                if len(token) == 2 and token.isalpha():
                    state_hint = token.upper()
                    break
            if not state_hint:
                city_to_state = {"san": "CA", "diego": "CA", "seattle": "WA", "bellevue": "WA"}
                for t in city_to_state:
                    if t in str(loc_pref).lower():
                        state_hint = city_to_state[t]
                        break
            if state_hint:
                def addr_state(addr: str):
                    import re
                    m = re.search(r",\s*([A-Z]{2})\s*\d{5}?", str(addr))
                    return m.group(1) if m else None
                filtered = [p for p in properties if addr_state(p.get('address')) == state_hint]
                if filtered:
                    properties = filtered
        except Exception:
            pass

        enriched = []
        for prop in properties[:3]:
            lat = (prop.get("geo") or {}).get("lat")
            lon = (prop.get("geo") or {}).get("lon")

            # If no lat/lon, try forward geocoding from address
            if (lat is None or lon is None) and prop.get("address"):
                ge = self._geocode_address(prop.get("address"))
                lat, lon = ge.get("lat"), ge.get("lon")

            if (lat is not None and lon is not None):
                addr_info = self._reverse_geocode(lat, lon)
            else:
                # Build a safe link from address if available
                addr = prop.get("address")
                maps_link = f"https://www.google.com/maps/search/?api=1&query={quote_plus(addr)}" if addr else None
                addr_info = {"address": addr, "mapsLink": maps_link}

            schools = self._nearby_places(lat, lon, "school", radius=4000, min_rating=4.0, max_results=5) if wants_schools else []
            restaurants = self._nearby_places(lat, lon, "restaurant", radius=3000, min_rating=4.2, max_results=5) if wants_restaurants else []

            enriched.append({
                **prop,
                "location": {
                    "lat": lat,
                    "lon": lon,
                    "address": addr_info.get("address") or prop.get("address"),
                    "mapsLink": addr_info.get("mapsLink")
                },
                "nearby": {
                    "top_schools": schools,
                    "top_restaurants": restaurants
                }
            })

        state["search_results"] = enriched
        # Persist enriched properties so API can surface updated list this turn
        self.last_properties = enriched
        return state
    
    def market_search_node(self, state: RealEstateAgentState) -> RealEstateAgentState:
        """Search for current market information using Tavily web search"""
        logger.info("Searching market information using Tavily...")
        
        user_query = state.get("user_query", "")
        buyer_profile = state.get("buyer_profile", {}) or {}
        location = (buyer_profile.get("location") or "").strip() or None

        # Attempt to extract a ZIP code from saved properties if available (higher specificity)
        zip_code = None
        try:
            saved_props = (buyer_profile or {}).get('saved_properties') or []
            import re
            for sp in saved_props:
                addr = sp.get('address') or ''
                m = re.search(r'\b(\d{5})(?:-\d{4})?\b', addr)
                if m:
                    zip_code = m.group(1)
                    break
        except Exception:
            pass
        if zip_code:
            logger.info(f"Using extracted ZIP for market search context: {zip_code}")
            # Prepend ZIP to user query to bias optimization stage
            if zip_code not in user_query:
                user_query = f"{user_query} {zip_code}".strip()

        logger.info(f"Market search - User query: '{user_query}'")
        logger.info(f"Market search - Buyer profile: {buyer_profile}")
        logger.info(f"Market search - Extracted location: '{location}' (zip={zip_code})")

        # Delegate to the helper that already handles location-aware variations and fallbacks
        market_items = self.tavily_market_search(user_query, location)
        # Attach metadata about zip to state for formatting
        if zip_code:
            state['__zip_used'] = zip_code
        state["search_results"] = market_items
        logger.info(f"Market search returned {len(market_items)} items (location={location})")
        return state
    
    def general_chat_node(self, state: RealEstateAgentState) -> RealEstateAgentState:
        """Handle general real estate conversation using LLM with context"""
        logger.info("Engaging in general real estate conversation...")
        
        user_query = state.get("user_query", "")
        messages = state.get("messages", [])
        buyer_profile = state.get("buyer_profile", {})
        
        system_prompt = """You are an expert real estate agent with deep knowledge of the real estate industry. 
        
Your expertise includes:
- Home buying and selling process
- Mortgage and financing options
- Real estate market analysis
- Property valuation
- Investment strategies
- Legal aspects of real estate
- Neighborhood insights
- First-time buyer guidance

Provide helpful, accurate, and professional advice. Be conversational but knowledgeable.
Reference previous conversation when relevant to provide context-aware responses.
If you're not certain about specific legal or financial advice, recommend consulting with relevant professionals.
"""
        
        if buyer_profile:
            profile_context = f"""

BUYER PROFILE CONTEXT:
- Name: {buyer_profile.get('buyer', 'Unknown')}
- Budget: ${buyer_profile.get('budget', {}).get('min', 0):,} - ${buyer_profile.get('budget', {}).get('max', 0):,}
- Looking for: {buyer_profile.get('bedrooms', 'N/A')} bedrooms, {buyer_profile.get('bathrooms', 'N/A')} bathrooms
- Location preference: {buyer_profile.get('location', 'N/A')}

Tailor your advice to this buyer's profile when relevant."""
            system_prompt += profile_context
        
        try:
            conversation_messages = [{"role": "system", "content": system_prompt}]
            
            recent_messages = messages[-6:] if len(messages) > 6 else messages
            for msg in recent_messages:
                conversation_messages.append({
                    "role": msg["role"] if msg["role"] in ["user", "assistant"] else "user",
                    "content": msg["content"]
                })
            
            response = self._safe_llm_invoke(conversation_messages)
            
            llm_content = response.content.lower()
            suggestion = ""
            
            if any(phrase in llm_content for phrase in ["search for properties", "look for homes", "find properties"]):
                suggestion = "\n\nIf you'd like me to search for specific properties, just describe your dream home!"
            elif any(phrase in llm_content for phrase in ["market data", "current trends", "market information"]):
                suggestion = "\n\nIf you'd like current market information, just ask about market trends or specific areas"
            
            final_response = response.content + suggestion
            
        except Exception as e:
            logger.error(f"General chat error: {e}")
            final_response = "I'm here to help with your real estate questions! Could you please rephrase your question?"
        
        state["search_results"] = [{"type": "chat_response", "content": final_response}]
        return state
    
    def format_response_node(self, state: RealEstateAgentState) -> RealEstateAgentState:
        """Format the final response using LLM to create natural, helpful responses"""
        logger.info("Formatting response with LLM...")
        
        search_results = state.get("search_results", [])
        next_action = state.get("next_action", "")
        user_query = state.get("user_query", "")
        buyer_profile = state.get("buyer_profile", {})
        
        if next_action == "property_search" and search_results:
            
            properties_summary = []
            for prop in search_results[:3]:
                prop_summary = f"""
Property: {prop['name']}
Price: {prop['price']}
Address: {prop['address']}
Specs: {prop['bedrooms']} bedrooms, {prop['bathrooms']} bathrooms, {prop['house_sqft']} sqft
Description: {prop['description'][:300]}...
Match Score: {prop['similarity_score']}
"""
                properties_summary.append(prop_summary)
            
            formatting_prompt = f"""You are a friendly real estate agent presenting property search results to a client.

Client's Request: "{user_query}"
Client Profile: {json.dumps(buyer_profile, indent=2) if buyer_profile else "No specific profile"}

Properties Found:
{chr(10).join(properties_summary)}

Create a natural, enthusiastic response that:
1. Acknowledges their request
2. Highlights the best matches
3. Points out key features that match their needs
4. Suggests next steps or asks follow-up questions

Be conversational and helpful, like a real estate agent would be."""

            try:
                response = self._safe_llm_invoke([
                    {"role": "user", "content": formatting_prompt}
                ])
                formatted_response = response.content
            except Exception as e:
                logger.error(f"Response formatting error: {e}")
                formatted_response = "🏠 Found these properties for you:\n\n"
                for i, prop in enumerate(search_results[:3], 1):
                    formatted_response += f"{i}. {prop['name']} - {prop['price']}\n"
                    formatted_response += f"📍 {prop['address']}\n"
                    formatted_response += f"🛏️ {prop['bedrooms']}bd/{prop['bathrooms']}ba • {prop['house_sqft']} sqft\n"
                    formatted_response += f"📝 {prop['description'][:200]}...\n"
                    formatted_response += f"🎯 Similarity: {prop['similarity_score']}\n\n"
        
        elif next_action == "property_search" and not search_results:
            buyer_profile = state.get("buyer_profile", {})
            max_budget = (buyer_profile.get('budget') or {}).get('max')
            budget_text = f" under ${max_budget:,.0f}" if max_budget else ""
            if state.get("_no_local_matches"):
                loc = buyer_profile.get('location') or 'your preferred area'
                formatted_response = (
                    f"I focused on {loc} and didn't find properties{budget_text} that match yet. "
                    "Would you like me to: (a) widen the radius, (b) relax budget/features, or (c) include nearby coastal cities?"
                )
            else:
                formatted_response = (
                    f"I didn't find any properties{budget_text} that match right now. "
                    "Want me to expand the search area or adjust other criteria?"
                )
        elif next_action == "market_search" and search_results:
            market_data = []
            for item in search_results[:5]:
                source_info = f"""
Source: {item.get('title', 'Market Info')}
URL: {item.get('url', 'N/A')}
Content: {item.get('content', 'No content available')[:500]}...
"""
                market_data.append(source_info)
            
            market_prompt = f"""You are a real estate expert analyzing current market information for a client.

Client's Question: "{user_query}"

Context:
Location Provided: {(buyer_profile or {}).get('location','N/A')}
ZIP Focus: {state.get('__zip_used','None')}

Market Information Found:
{chr(10).join(market_data)}

Create a comprehensive, informative response that:
1. Directly answers their specific question with data/numbers if available
2. Provides relevant market context and trends
3. Explains what this means for buyers/sellers
4. Cites sources when possible
5. If specific numbers aren't available, explain what factors affect pricing

For price questions specifically:
- Include specific price ranges or averages if mentioned in sources
- Explain factors that influence pricing (location, condition, amenities)
- Provide context about market trends (rising/falling, hot/slow market)
- Suggest next steps for getting precise pricing

Be professional but approachable, like a knowledgeable real estate expert who has done research."""

            try:
                response = self._safe_llm_invoke([
                    {"role": "user", "content": market_prompt}
                ])
                formatted_response = response.content
            except Exception as e:
                logger.error(f"Market response formatting error: {e}")
                
                formatted_response = f"Market Information for: {user_query}\n\n"
                
                for i, item in enumerate(search_results[:3], 1):
                    title = item.get('title', f'Market Info {i}')
                    content = item.get('content', 'No content available')
                    url = item.get('url', '')
                    
                    formatted_response += f"**{i}. {title}**\n"
                    formatted_response += f"{content[:300]}...\n"
                    if url:
                        formatted_response += f"Source: {url}\n"
                    formatted_response += "\n"
                
                if not search_results:
                    formatted_response += "I couldn't find current market data for your specific question. For accurate pricing information, I recommend:\n\n"
                    formatted_response += "• Checking recent sales on Zillow or Redfin\n"
                    formatted_response += "• Consulting with a local real estate agent\n"
                    formatted_response += "• Looking at MLS data for comparable properties\n"
        elif next_action == "market_search" and not search_results:
            # Provide a specific helpful fallback instead of generic assistant blurb
            loc = (buyer_profile or {}).get('location') or 'your area'
            zip_note = ''
            if state.get('__zip_used'):
                zip_note = f" (ZIP {state.get('__zip_used')})"
            formatted_response = (
                f"I tried searching for up-to-date market reports in {loc}{zip_note}, "
                "but the sources I queried didn't return recent, clearly relevant data right now.\n\n"
                "Here's how you can still get insight quickly:\n"
                "1. Check median list and sold prices on Zillow / Redfin (filter to condos, last 30-90 days).\n"
                "2. Look for local association or MLS monthly market reports (often published as a PDF).\n"
                "3. Compare year-over-year changes in: median price, days on market, inventory, and price per sqft.\n"
                "4. If you share a ZIP code, I can refine and try again with more targeted queries.\n\n"
                "Want me to broaden the search timeframe (include 2024 reports) or switch to overall residential instead of just condos?"
            )
        
        elif next_action == "location_context" and search_results:
            # If the result is profile-centric (single dict with nearby lists), present only schools/restaurants
            if isinstance(search_results, list) and len(search_results) == 1 and isinstance(search_results[0], dict) and "nearby" in search_results[0]:
                loc = search_results[0].get("location", {})
                nearby = search_results[0].get("nearby", {})
                schools = nearby.get("top_schools", [])
                restaurants = nearby.get("top_restaurants", [])
                lines = ["Here are top nearby options:"]
                if schools:
                    lines.append("\nTop Schools:")
                    for s in schools[:5]:
                        lines.append(f"- {s.get('name')} ({s.get('rating','?')}⭐, {s.get('distance_km','?')} km)")
                if restaurants:
                    lines.append("\nTop Restaurants:")
                    for r in restaurants[:5]:
                        lines.append(f"- {r.get('name')} ({r.get('rating','?')}⭐, {r.get('distance_km','?')} km)")
                if loc.get('mapsLink'):
                    lines.append(f"\nMap: {loc.get('mapsLink')}")
                formatted_response = "\n".join(lines)
            else:
                # Legacy property-enriched results: omit property names/addresses per requirement and only list nearby places
                lines = ["Here are nearby options:"]
                agg_schools = []
                agg_restaurants = []
                for prop in search_results[:3]:
                    nb = (prop.get('nearby') or {})
                    agg_schools.extend(nb.get('top_schools') or [])
                    agg_restaurants.extend(nb.get('top_restaurants') or [])
                # Deduplicate by name
                def dedupe(items):
                    seen = set()
                    out = []
                    for it in items:
                        n = it.get('name')
                        if n and n not in seen:
                            seen.add(n)
                            out.append(it)
                    return out
                agg_schools = dedupe(agg_schools)[:5]
                agg_restaurants = dedupe(agg_restaurants)[:5]
                if agg_schools:
                    lines.append("\nTop Schools:")
                    for s in agg_schools:
                        lines.append(f"- {s.get('name')} ({s.get('rating','?')}⭐, {s.get('distance_km','?')} km)")
                if agg_restaurants:
                    lines.append("\nTop Restaurants:")
                    for r in agg_restaurants:
                        lines.append(f"- {r.get('name')} ({r.get('rating','?')}⭐, {r.get('distance_km','?')} km)")
                formatted_response = "\n".join(lines)

        elif search_results and search_results[0].get("type") == "chat_response":
            formatted_response = search_results[0]["content"]
        
        else:
            formatted_response = "I'd be happy to help you! I can search for properties based on your dream home description, find current market information, or answer general real estate questions. What would you like to know?"
        
        messages = state.get("messages", [])
        # Ensure response ends cleanly
        formatted_response = self._ensure_complete_sentence(formatted_response)
        messages.append({"role": "assistant", "content": formatted_response})
        state["messages"] = messages
        state["formatted_response"] = formatted_response
        return state
    
    def route_decision(self, state: RealEstateAgentState) -> Literal["property_search", "market_search", "location_context", "general_chat"]:
        """Route to appropriate tool based on user's message"""
        next_action = state.get("next_action", "general_chat")
        logger.info(f"Routing to: {next_action}")
        return next_action
    
    def setup_graph(self):
        """Build the LangGraph workflow"""
        try:
            builder = StateGraph(RealEstateAgentState)
            
            builder.add_node("analyze_intent", self.analyze_user_intent)
            builder.add_node("property_search", self.property_search_node)
            builder.add_node("market_search", self.market_search_node)
            builder.add_node("location_context", self.location_context_node)
            builder.add_node("general_chat", self.general_chat_node)
            builder.add_node("format_response", self.format_response_node)
            
            builder.add_edge(START, "analyze_intent")
            builder.add_conditional_edges(
                "analyze_intent", 
                self.route_decision,
                {
                    "property_search": "property_search",
                    "market_search": "market_search", 
                    "location_context": "location_context",
                    "general_chat": "general_chat"
                }
            )
            builder.add_edge("property_search", "format_response")
            builder.add_edge("market_search", "format_response")
            builder.add_edge("location_context", "format_response")
            builder.add_edge("general_chat", "format_response")
            builder.add_edge("format_response", END)
            
            self.graph = builder.compile()
            logger.info("LangGraph workflow created successfully!")
            
        except Exception as e:
            logger.error(f"Failed to setup graph: {e}")
            raise
    
    def get_llm_response(self, user_message: str, system_prompt: str = None) -> str:
        """Get LLM response for simple interactions (like greeting)"""
        try:
            if system_prompt:
                response = self._safe_llm_invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ])
            else:
                response = self._safe_llm_invoke([{"role": "user", "content": user_message}])
            return response.content
        except Exception as e:
            logger.error(f"LLM response error: {e}")
            return "I'm sorry, I'm having trouble processing your request right now."
    
    def get_buyer_profile(self, buyer_name: str) -> Dict[str, Any]:
        """Get buyer profile (wrapper for compatibility)"""
        return self.load_buyer_profile(buyer_name)
    
    def search_properties(self, query_text: str, num_results: int = 3, buyer_profile: Dict[str, Any] = None, saved_properties: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main method to search properties using the LangGraph workflow"""
        try:
            initial_state = RealEstateAgentState(
                messages=[{"role": "user", "content": query_text}],
                user_query=query_text,
                search_results=[],
                buyer_profile=buyer_profile or {},
                next_action="",
                formatted_response="",
                saved_properties=saved_properties or []
            )
            
            result = self.graph.invoke(initial_state)
            
            if isinstance(result, dict):
                return {
                    "response": result.get("formatted_response", ""),
                    "properties": result.get("search_results", [])[:num_results]
                }
            else:
                logger.warning(f"Unexpected result type in search_properties: {type(result)}")
                return {
                    "response": str(result) if result else "No results found.",
                    "properties": []
                }
            
        except Exception as e:
            logger.error(f"Search properties error: {e}")
            logger.error(traceback.format_exc())
            return {
                "response": "I'm sorry, I encountered an error while searching for properties.",
                "properties": []
            }
    
    def chat(self, user_message: str, buyer_profile: Dict[str, Any] = None, saved_properties: List[Dict[str, Any]] = None) -> str:
        """Main chat method for Flask integration with conversation memory"""
        try:
            self.conversation_history.append({"role": "user", "content": user_message})
            
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            initial_state = RealEstateAgentState(
                messages=self.conversation_history.copy(), 
                user_query=user_message,
                search_results=[],
                buyer_profile=buyer_profile or {},
                next_action="",
                formatted_response="",
                saved_properties=saved_properties or []
            )
            
            result = self.graph.invoke(initial_state)
            
            if isinstance(result, dict):
                response = result.get("formatted_response", "I'm sorry, I couldn't process your request.")
                
                self.conversation_history.append({"role": "assistant", "content": response})
                
                return response
            else:
                logger.warning(f"Unexpected result type: {type(result)}")
                response = str(result) if result else "I'm sorry, I couldn't process your request."
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            logger.error(traceback.format_exc())
            error_response = "I'm sorry, I'm having trouble processing your request right now."
            self.conversation_history.append({"role": "assistant", "content": error_response})
            return error_response
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.conversation_history.copy()

class RealEstateAgent(LangGraphRealEstateAgent):
    """Alias for backward compatibility"""
    pass