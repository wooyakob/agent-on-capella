import os
import json
import logging
from typing import List, Dict, Any
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import CouchbaseException
import couchbase.search as search
from couchbase.vector_search import VectorQuery, VectorSearch
from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealEstateAgent:
    """
    A conversational real estate agent that uses Amazon Bedrock Models and Couchbase Capella Operational Database and Vector Search
    to help users find their dream properties.
    """
    
    def __init__(self):
        """Initialize the Real Estate Agent with AWS Bedrock and Couchbase Capella."""
        self.setup_llm()
        self.setup_embeddings()
        self.setup_vector_search()
        
    def setup_llm(self):
        """Set up the ChatBedrock LLM for conversation."""
        try:
            model_kwargs = {
                "temperature": 0.7,
                "max_tokens": 800
            }
            
            self.chat_model = ChatBedrock(
                model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
                region_name=os.getenv("AWS_REGION", "us-east-2"),
                model_kwargs=model_kwargs
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def setup_embeddings(self):
        """Set up the Titan embeddings model."""
        try:
            self.embeddings_model = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v2:0",
                region_name=os.getenv("AWS_REGION", "us-east-2")
            )
            logger.info("Embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def setup_vector_search(self):
        """Set up Couchbase vector search connection."""
        try:
            pa = PasswordAuthenticator(os.getenv("CB_USERNAME"), os.getenv("CB_PASSWORD"))
            self.cluster = Cluster(os.getenv("CB_HOSTNAME"), ClusterOptions(pa))
            
            self.properties_bucket = self.cluster.bucket("properties")
            self.properties_scope = self.properties_bucket.scope("2025-listings")
            self.properties_collection = self.properties_scope.collection("united-states")
            self.search_index = "properties-index"
             
            self.profiles_bucket = self.cluster.bucket("profiles")
            self.buyers_scope = self.profiles_bucket.scope("buyers")
            self.buyers_collection = self.buyers_scope.collection("2025")
            
            logger.info("Vector search and profiles database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            raise
    
    def get_llm_response(self, user_message: str, system_prompt: str = None) -> str:
        """Get a response from the LLM."""
        try:
            messages = []
            
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            messages.append(HumanMessage(content=user_message))
            
            response = self.chat_model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Failed to get LLM response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
    
    def get_buyer_profile(self, buyer_name: str) -> Dict[str, Any]:
        """Retrieve buyer profile from Couchbase."""
        try:
            
            query = f"SELECT META().id, * FROM `profiles`.`buyers`.`2025` WHERE LOWER(buyer) LIKE LOWER('%{buyer_name}%')"
            result = self.cluster.query(query)
            
            for row in result:
                profile_data = dict(row)
                logger.info(f"Raw profile data: {profile_data}")
                
                if '2025' in profile_data:
                    actual_profile = profile_data['2025']
                    logger.info(f"Found profile for {buyer_name}: {actual_profile}")
                    return actual_profile
                
                if 'id' in profile_data:
                    del profile_data['id']
                    return profile_data
            
            logger.warning(f"No profile found for {buyer_name}")
            return {}
        except Exception as e:
            logger.error(f"Failed to retrieve buyer profile: {e}")
            return {}
    
    def enhance_search_with_profile(self, query_text: str, profile: Dict[str, Any]) -> str:
        """Enhance the search query with profile information."""
        if not profile:
            return query_text
        
        enhanced_query = query_text
        
        if 'budget' in profile:
            budget = profile['budget']
            if 'min' in budget and 'max' in budget:
                enhanced_query += f" budget range ${budget['min']:,} to ${budget['max']:,}"
        
        if 'location' in profile:
            enhanced_query += f" location {profile['location']}"
        
        if 'bedrooms' in profile:
            enhanced_query += f" {profile['bedrooms']} bedroom"
        if 'bathrooms' in profile:
            enhanced_query += f" {profile['bathrooms']} bathroom"
        
        logger.info(f"Enhanced query: {enhanced_query}")
        return enhanced_query
    
    def search_properties(self, query_text: str, num_results: int = 3, buyer_profile: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for properties using vector embeddings, optionally filtered by buyer profile."""
        try:
            if buyer_profile:
                enhanced_query = self.enhance_search_with_profile(query_text, buyer_profile)
            else:
                enhanced_query = query_text
            
            vector = self.embeddings_model.embed_query(enhanced_query)
            
            search_req = search.SearchRequest.create(search.MatchNoneQuery()).with_vector_search(
                VectorSearch.from_vector_query(VectorQuery('embedding', vector, num_candidates=20))
            )
            
            result = self.properties_scope.search(
                self.search_index,
                search_req,
                SearchOptions(limit=20, fields=["name", "description", "price", "bedrooms", "bathrooms", "address"])
            )
            
            properties = []
            for row in result.rows():
                property_data = {
                    "name": row.fields.get('name', 'N/A'),
                    "description": row.fields.get('description', 'N/A'),
                    "price": row.fields.get('price', 'N/A'),
                    "bedrooms": row.fields.get('bedrooms', 'N/A'),
                    "bathrooms": row.fields.get('bathrooms', 'N/A'),
                    "address": row.fields.get('address', 'N/A')
                }
                properties.append(property_data)
            
            if buyer_profile and 'budget' in buyer_profile:
                properties = self.filter_properties_by_budget(properties, buyer_profile['budget'])
            
            properties = properties[:num_results]
            
            logger.info(f"Found {len(properties)} matching properties after filtering")
            return properties
            
        except Exception as e:
            logger.error(f"Failed to search properties: {e}")
            return []
    
    def filter_properties_by_budget(self, properties: List[Dict[str, Any]], budget: Dict[str, int]) -> List[Dict[str, Any]]:
        """Filter properties based on budget constraints using actual price data."""
        min_budget = budget.get('min', 0)
        max_budget = budget.get('max', float('inf'))
        
        logger.info(f"Filtering properties by budget: ${min_budget:,} - ${max_budget:,}")
        
        filtered_properties = []
        
        for property_data in properties:
            price_str = property_data.get('price', '')
            
            if price_str and price_str != 'N/A':
                try:
                    clean_price_str = price_str.replace('$', '').replace(',', '')
                    property_price = float(clean_price_str)
                    
                    if min_budget <= property_price <= max_budget:
                        filtered_properties.append(property_data)
                        logger.info(f"✅ Included: {property_data['name']} - ${property_price:,.0f} (within budget)")
                    else:
                        logger.info(f"❌ Excluded: {property_data['name']} - ${property_price:,.0f} (outside budget ${min_budget:,}-${max_budget:,})")
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse price '{price_str}' for {property_data['name']}: {e}")
                    filtered_properties.append(property_data)
            else:
                logger.info(f"⚠️  No price data for {property_data['name']}, including anyway")
                filtered_properties.append(property_data)
        
        logger.info(f"Budget filtering: {len(properties)} -> {len(filtered_properties)} properties")
        return filtered_properties
    
    def format_property_results(self, properties: List[Dict[str, Any]], buyer_profile: Dict[str, Any] = None) -> str:
        """Format property search results for presentation, with profile context."""
        if not properties:
            return "I couldn't find any properties matching your description and budget. Let me know if you'd like to try a different search or modify your criteria."
        
        profile_context = ""
        if buyer_profile:
            budget = buyer_profile.get('budget', {})
            profile_context = f"Based on your profile (budget: ${budget.get('min', 0):,} - ${budget.get('max', 0):,}, {buyer_profile.get('bedrooms', 'any')} bed, {buyer_profile.get('bathrooms', 'any')} bath, {buyer_profile.get('location', 'any location')}), "
        
        result_text = f"{profile_context}I found {len(properties)} properties within your budget:\n\n"
        
        for i, prop in enumerate(properties, 1):
            result_text += f"**Property {i}: {prop['name']}**\n"
            
            if prop.get('price') and prop.get('price') != 'N/A':
                result_text += f"Price: {prop['price']}\n"
            
            if prop.get('bedrooms') and prop.get('bedrooms') != 'N/A':
                result_text += f"Bedrooms: {prop['bedrooms']}, "
            if prop.get('bathrooms') and prop.get('bathrooms') != 'N/A':
                result_text += f"Bathrooms: {prop['bathrooms']}\n"
            else:
                result_text += "\n"
            
            if prop.get('address') and prop.get('address') != 'N/A':
                result_text += f"Address: {prop['address']}\n"
            
            result_text += f"Description: {prop['description'][:200]}{'...' if len(prop['description']) > 200 else ''}\n"
            result_text += "-" * 50 + "\n\n"
        
        return result_text
    
    def is_property_search_query(self, user_input: str) -> bool:
        """Determine if user input is a property search query or just conversation."""
    
        property_keywords = [
            'house', 'home', 'property', 'bedroom', 'bathroom', 'kitchen', 'garage',
            'yard', 'garden', 'pool', 'gated', 'community', 'neighborhood', 'location',
            'modern', 'traditional', 'luxury', 'apartment', 'condo', 'townhouse',
            'square feet', 'acres', 'fireplace', 'balcony', 'patio', 'view', 'ocean',
            'mountain', 'downtown', 'suburb', 'quiet', 'family', 'school', 'park'
        ]
        
       
        greeting_keywords = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'nice to meet you', 'thanks', 'thank you', 'yes', 'no',
            'okay', 'ok', 'sure', 'maybe', 'tell me more', 'interesting'
        ]
        
        user_lower = user_input.lower()
      
        if any(greeting in user_lower for greeting in greeting_keywords):
            return False
            
        if any(keyword in user_lower for keyword in property_keywords):
            return True
            
        if len(user_input.split()) > 10:
            return True
            
        return False

    def start_conversation(self):
        """Start the conversational interface."""
        print("🏠 Welcome to Your AI Real Estate Agent! 🏠")
        print("=" * 60)
        
        system_prompt = """You are a friendly, professional real estate agent. 
        You help clients find their dream properties by understanding their needs and preferences.
        Be conversational, warm, and helpful. Ask follow-up questions to better understand what they're looking for.
        Keep responses concise but engaging. Don't mention technical details about embeddings or vector search."""
        
        greeting = self.get_llm_response(
            "Introduce yourself and ask the user to describe their dream property.",
            system_prompt
        )
        
        print(f"\n🤖 Real Estate Agent: {greeting}\n")
        
        while True:
            try:
                user_input = input("🏡 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    farewell = self.get_llm_response(
                        "Say a professional goodbye to a real estate client.",
                        system_prompt
                    )
                    print(f"\n🤖 Real Estate Agent: {farewell}")
                    break
                
                if not user_input:
                    continue
                
           
                if self.is_property_search_query(user_input):
                   
                    print("\n🔍 Searching for matching properties...")
                    properties = self.search_properties(user_input)
                    
                    property_results = self.format_property_results(properties)
                    
                    llm_prompt = f"""The client described their dream property as: "{user_input}"
                    
                    Here are the search results:
                    {property_results}
                    
                    Provide a brief, enthusiastic response about these properties and ask if they'd like more details 
                    about any specific property or if they'd like to refine their search."""
                    
                    agent_response = self.get_llm_response(llm_prompt, system_prompt)
                    
                    print(f"\n🤖 Real Estate Agent: {agent_response}")
                    
                   
                    print(f"\n📋 Property Details:\n{property_results}")
                else:
          
                    agent_response = self.get_llm_response(user_input, system_prompt)
                    print(f"\n🤖 Real Estate Agent: {agent_response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using our real estate service. Have a great day!")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                print("\n❌ I encountered an issue. Let me try to help you in a different way.")

def main():
    """Main function to run the conversational real estate agent."""
    try:
        agent = RealEstateAgent()
        agent.start_conversation()
    except Exception as e:
        print(f"❌ Failed to start the real estate agent: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
