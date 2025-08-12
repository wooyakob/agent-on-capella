class ChatApp {
    constructor() {
        this.sessionId = null;
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.buyerNameInput = document.getElementById('buyerNameInput');
        this.profileStatus = document.getElementById('profileStatus');
        this.sendButton = document.getElementById('sendButton');
        this.chatForm = document.getElementById('chatForm');
        this.initialLoading = document.getElementById('initialLoading');
        this.clearHistoryButton = document.getElementById('clearHistoryButton');
        
        this.init();
    }
    
    async init() {
        try {
            await this.startSession();
            this.setupEventListeners();
            this.showGraphInfo();
        } catch (error) {
            console.error('Failed to initialize chat app:', error);
            this.showError('Failed to initialize. Please refresh the page.');
        }
    }
    
    async startSession() {
        const response = await fetch('/api/start_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            this.sessionId = data.session_id;
            this.initialLoading.style.display = 'none';

            this.addMessage('agent', data.greeting);
            
            this.messageInput.disabled = false;
            this.sendButton.disabled = false;
            this.messageInput.focus();
        } else {
            throw new Error(data.error || 'Failed to start session');
        }
    }
    
    async showGraphInfo() {
        try {
            const response = await fetch('/api/graph_info');
            const data = await response.json();
            
            if (data.success) {
                const infoDiv = document.createElement('div');
                infoDiv.className = 'graph-info';
                infoDiv.innerHTML = `
                    <div class="info-header"><strong>Real Estate Agent Active</strong></div>
                    <div class="info-details">
                        <div><strong>Property Search:</strong> Couchbase vector search for dream properties</div>
                        <div><strong>Market Research:</strong> Tavily web search for current information</div>
                        <div><strong>Expert Advice:</strong> Bedrock LLM for real estate guidance</div>
                    </div>
                `;
                this.chatMessages.appendChild(infoDiv);
            }
        } catch (error) {
            console.log('Graph info not available');
        }
    }
    
    setupEventListeners() {
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.buyerNameInput.addEventListener('input', () => {
            const buyerName = this.buyerNameInput.value.trim();
            if (buyerName) {
                this.profileStatus.textContent = `🔍 Will search for profile: ${buyerName}`;
                this.profileStatus.style.color = '#2196f3';
            } else {
                this.profileStatus.textContent = '';
            }
        });
        
        this.clearHistoryButton.addEventListener('click', () => {
            this.clearConversationHistory();
        });
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        const buyerName = this.buyerNameInput.value.trim();
        
        if (!message) return;
        
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.setLoading(true);
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    buyer_name: buyerName
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
         
                if (data.buyer_profile && Object.keys(data.buyer_profile).length > 0) {
                    const profile = data.buyer_profile;
                    this.profileStatus.innerHTML = `
                        ✅ Profile: ${profile.buyer} | Budget: $${profile.budget?.min?.toLocaleString()}-$${profile.budget?.max?.toLocaleString()} | 
                        ${profile.bedrooms}bd/${profile.bathrooms}ba in ${profile.location}
                    `;
                    this.profileStatus.style.color = '#4caf50';
                }
                
                
                this.addMessage('agent', data.response, data.properties || []);
                
            } else {
                this.showError(data.error || 'An error occurred');
            }
            
        } catch (error) {
            console.error('Send message error:', error);
            this.showError('Failed to send message. Please try again.');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(sender, content, properties = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'agent' ? '🏠' : '👤';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (sender === 'agent') {
            const formattedContent = content
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\n/g, '<br>');
            contentDiv.innerHTML = formattedContent;
        } else {
            contentDiv.textContent = content;
        }
        
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timestampDiv);
        
        if (properties && properties.length > 0) {
            const propertiesDiv = document.createElement('div');
            propertiesDiv.className = 'properties-container';
            
            properties.slice(0, 3).forEach((property, index) => {
                const propertyCard = document.createElement('div');
                propertyCard.className = 'property-card';
                propertyCard.innerHTML = `
                    <div class="property-header">
                        <h4>${property.name}</h4>
                        <span class="property-price">${property.price}</span>
                    </div>
                    <div class="property-details">
                        <div class="property-address">📍 ${property.address}</div>
                        <div class="property-specs">
                            🛏️ ${property.bedrooms}bd/${property.bathrooms}ba • ${property.house_sqft} sqft
                        </div>
                        <div class="property-description">${property.description.substring(0, 150)}...</div>
                        <div class="property-score">🎯 Match: ${(property.similarity_score * 100).toFixed(1)}%</div>
                    </div>
                `;
                propertiesDiv.appendChild(propertyCard);
            });
            
            messageDiv.appendChild(propertiesDiv);
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    setLoading(loading) {
        const loadingDiv = document.getElementById('loadingIndicator') || this.createLoadingIndicator();
        loadingDiv.style.display = loading ? 'flex' : 'none';
        this.sendButton.disabled = loading;
        this.messageInput.disabled = loading;
        
        if (loading) {
            this.scrollToBottom();
        }
    }
    
    createLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'loadingIndicator';
        loadingDiv.className = 'loading-indicator';
        loadingDiv.innerHTML = `
            <div class="message agent">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                        <span class="loading-text">Your Real Estate Agent has received your message</span>
                    </div>
                </div>
            </div>
        `;
        this.chatMessages.appendChild(loadingDiv);
        return loadingDiv;
    }
    
    showError(message) {
        this.addMessage('agent', `${message}`);
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    async clearConversationHistory() {
        try {
          
            this.clearHistoryButton.disabled = true;
            this.clearHistoryButton.textContent = '🗑️ Clearing...';
            
            const response = await fetch('/api/clear_history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                
                const graphInfo = this.chatMessages.querySelector('.graph-info');
                this.chatMessages.innerHTML = '';
                if (graphInfo) {
                    this.chatMessages.appendChild(graphInfo);
                }
                
             
                this.addMessage('agent', 'Conversation history cleared! How can I help you today?');
                
                this.profileStatus.textContent = 'History cleared successfully';
                this.profileStatus.style.color = '#059669';
                
                setTimeout(() => {
                    this.profileStatus.textContent = '';
                }, 3000);
            } else {
                this.showError(data.error || 'Failed to clear history');
            }
        } catch (error) {
            console.error('Failed to clear history:', error);
            this.showError('Failed to clear conversation history');
        } finally {
            this.clearHistoryButton.disabled = false;
            this.clearHistoryButton.textContent = '🗑️ Clear History';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});
