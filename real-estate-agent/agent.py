import os
import json
import logging
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from datetime import timedelta

from langgraph.graph import StateGraph, START, END
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_tavily import TavilySearch

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions

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

class LangGraphRealEstateAgent:
    """
    A conversational real estate agent that uses LangGraph to orchestrate 
    multiple tools: Couchbase vector search, Tavily web search, and Bedrock LLM.
    """
    
    def __init__(self):
        """Initialize the LangGraph Real Estate Agent."""
        self.conversation_history = []
        self.setup_tools()
        self.setup_graph()
        
    def setup_tools(self):
        """Set up all tools: LLM, embeddings, Couchbase, and Tavily."""
        try:
            self.llm = ChatBedrock(
                model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
                region_name=os.getenv("AWS_REGION", "us-east-2"),
                temperature=0.7
            )
            
            self.embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

            auth = PasswordAuthenticator(os.getenv("CB_USERNAME"), os.getenv("CB_PASSWORD"))
            options = ClusterOptions(auth)
            cluster = Cluster(os.getenv("CB_HOSTNAME"), options)
            cluster.wait_until_ready(timedelta(seconds=5))
            
            BUCKET = "properties"
            SCOPE = "2025-listings"
            COLLECTION = "united-states"
            SEARCH_INDEX = "properties-index"
            
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
            
            logger.info("All tools initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise
    
    def couchbase_property_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search properties in Couchbase using vector similarity"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            properties = []
            for doc, score in results:
                property_data = {
                    "name": doc.metadata.get('name', 'Unknown Property'),
                    "price": doc.metadata.get('price', 'Price not available'),
                    "address": doc.metadata.get('address', 'Address not available'),
                    "bedrooms": doc.metadata.get('bedrooms', 'N/A'),
                    "bathrooms": doc.metadata.get('bathrooms', 'N/A'),
                    "house_sqft": doc.metadata.get('house_sqft', 'N/A'),
                    "description": doc.page_content,
                    "similarity_score": round(score, 3)
                }
                properties.append(property_data)
            return properties
        except Exception as e:
            logger.error(f"Couchbase search error: {e}")
            return []
    
    def tavily_market_search(self, query: str) -> List[Dict[str, Any]]:
        """Search market information using Tavily"""
        try:
            optimization_prompt = f"""
            Create optimized search queries for finding real estate market information.
            Original user query: "{query}"
            
            Generate 3 different focused search queries that would find current market data, price trends, 
            and housing market information. Each should be concise (under 15 words) and target different aspects:
            1. Current market prices/trends
            2. Average home prices
            3. Real estate market analysis
            
            Return each query on a separate line, nothing else.
            """
            optimized_queries = self.llm.invoke(optimization_prompt)
            if hasattr(optimized_queries, 'content'):
                optimized_queries = optimized_queries.content
            optimized_queries = str(optimized_queries).strip().split('\n')
            logger.info(f"Generated {len(optimized_queries)} search queries")
            
            all_results = []
            
            for i, opt_query in enumerate(optimized_queries[:3]): 
                try:
                    clean_query = opt_query.strip().lstrip('123456789.-').strip()
                    logger.info(f"Searching with query: {clean_query}")
                    
                    search_variations = [
                        clean_query,
                        f"San Diego {clean_query}",
                        f"{clean_query} 2025",
                        f"current {clean_query}"
                    ]
                    
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
            
            logger.info(f"Found {len(unique_results)} unique market information items")
            return unique_results[:3]
            
        except Exception as e:
            logger.error(f"Market search error: {e}")
            try:
                logger.info("Trying fallback search...")
                fallback_results = self.tavily_search.invoke("San Diego real estate market trends prices")
                
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

3. GENERAL_CHAT - Use for general real estate advice, process questions, or conversational responses
   Examples: "how do I get pre-approved?", "what's the buying process?", "tell me about real estate", "should I buy or rent?"

Key indicators:
- Market questions often ask about: prices, costs, averages, trends, market conditions, statistics
- Property searches ask about: finding, looking for, showing, specific property features
- General chat asks about: processes, advice, how-to questions

{context}
Current User Query: "{query}"

Based on the conversation context and current query, respond with ONLY one word: PROPERTY_SEARCH, MARKET_SEARCH, or GENERAL_CHAT"""

        try:
            response = self.llm.invoke([
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
        
        enhancement_prompt = """You are helping enhance a property search query for vector search. 
        
Original Query: "{query}"
Buyer Profile: {profile}

Create an enhanced search query that combines the user's requirements with their profile.
Focus on descriptive terms that would match property descriptions.
Include specific details like location preferences, property types, amenities, and lifestyle needs.

Enhanced Query:"""

        try:
            enhanced_response = self.llm.invoke([
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
        
        properties = self.couchbase_property_search(enhanced_query, k=5)
        
        if not properties and enhanced_query != user_query:
            logger.info("No results with enhanced query, trying original...")
            properties = self.couchbase_property_search(user_query, k=5)
        
        state["search_results"] = properties
        logger.info(f"Found {len(properties)} properties")
        
        return state
    
    def market_search_node(self, state: RealEstateAgentState) -> RealEstateAgentState:
        """Search for current market information using Tavily web search"""
        logger.info("Searching market information using Tavily...")
        
        user_query = state.get("user_query", "")
        
        search_optimization_prompt = """You are helping create optimal web search queries for real estate market information.

User's Question: "{query}"

Create 3 different focused search queries that would find current, relevant real estate market information:
1. A specific query targeting the exact question
2. A broader market trends query 
3. A pricing/statistics focused query

Include terms like "2025", "current", "average price", "market data", and specific locations.

Query 1:
Query 2:  
Query 3:"""

        search_queries = []
        
        try:
            optimized_response = self.llm.invoke([
                {"role": "user", "content": search_optimization_prompt.format(query=user_query)}
            ])
            
            response_lines = optimized_response.content.strip().split('\n')
            for line in response_lines:
                if line.strip() and not line.startswith('Query'):
                    search_queries.append(line.strip())
            
            logger.info(f"Generated {len(search_queries)} search queries")
            
        except Exception as e:
            logger.error(f"Search optimization error: {e}")
           
            search_queries = [
                f"{user_query} 2025",
                f"San Diego real estate average prices 2025",
                f"current real estate market trends San Diego"
            ]
        
        all_market_info = []
        
        for query in search_queries[:3]: 
            try:
                logger.info(f"Searching with query: {query}")
                results = self.tavily_market_search(query)
                if results:
                    all_market_info.extend(results)
                    logger.info(f"Found {len(results)} results for query: {query}")
            except Exception as e:
                logger.error(f"Error searching with query '{query}': {e}")
                continue
        
        unique_results = []
        seen_urls = set()
        seen_titles = set()
        
        logger.info(f"Processing {len(all_market_info)} total market info items")
        for i, item in enumerate(all_market_info):
            if not isinstance(item, dict):
                logger.debug(f"Item {i}: Skipping non-dict item: {type(item)} - {str(item)[:100]}")
                continue
            
            logger.debug(f"Item {i}: {item.keys()}")
            url = item.get('url', '')
            title = item.get('title', '')
            
            if not url and not title:
                logger.debug(f"Item {i}: No URL or title, adding anyway")
                unique_results.append(item)
            elif url and url not in seen_urls:
                unique_results.append(item)
                seen_urls.add(url)
                if title:
                    seen_titles.add(title)
                logger.debug(f"Item {i}: Added with URL: {url[:50]}...")
            elif title and title not in seen_titles:
                unique_results.append(item)
                seen_titles.add(title)
                logger.debug(f"Item {i}: Added with title: {title[:50]}...")
            else:
                logger.debug(f"Item {i}: Duplicate - URL: {url[:30]}... Title: {title[:30]}...")
        
        state["search_results"] = unique_results[:3]
        logger.info(f"Found {len(unique_results)} unique market information items")
        
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
            
            response = self.llm.invoke(conversation_messages)
            
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
                response = self.llm.invoke([
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
                response = self.llm.invoke([
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
        
        elif search_results and search_results[0].get("type") == "chat_response":
            formatted_response = search_results[0]["content"]
        
        else:
            formatted_response = "I'd be happy to help you! I can search for properties based on your dream home description, find current market information, or answer general real estate questions. What would you like to know?"
        
        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": formatted_response})
        state["messages"] = messages
        state["formatted_response"] = formatted_response
        
        return state
    
    def route_decision(self, state: RealEstateAgentState) -> Literal["property_search", "market_search", "general_chat"]:
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
            builder.add_node("general_chat", self.general_chat_node)
            builder.add_node("format_response", self.format_response_node)
            
            builder.add_edge(START, "analyze_intent")
            builder.add_conditional_edges(
                "analyze_intent", 
                self.route_decision,
                {
                    "property_search": "property_search",
                    "market_search": "market_search", 
                    "general_chat": "general_chat"
                }
            )
            builder.add_edge("property_search", "format_response")
            builder.add_edge("market_search", "format_response")
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
                response = self.llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ])
            else:
                response = self.llm.invoke([{"role": "user", "content": user_message}])
            return response.content
        except Exception as e:
            logger.error(f"LLM response error: {e}")
            return "I'm sorry, I'm having trouble processing your request right now."
    
    def get_buyer_profile(self, buyer_name: str) -> Dict[str, Any]:
        """Get buyer profile (wrapper for compatibility)"""
        return self.load_buyer_profile(buyer_name)
    
    def search_properties(self, query_text: str, num_results: int = 3, buyer_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main method to search properties using the LangGraph workflow"""
        try:
            initial_state = RealEstateAgentState(
                messages=[{"role": "user", "content": query_text}],
                user_query=query_text,
                search_results=[],
                buyer_profile=buyer_profile or {},
                next_action="",
                formatted_response=""
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
            import traceback
            logger.error(traceback.format_exc())
            return {
                "response": "I'm sorry, I encountered an error while searching for properties.",
                "properties": []
            }
    
    def chat(self, user_message: str, buyer_profile: Dict[str, Any] = None) -> str:
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
                formatted_response=""
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
            import traceback
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