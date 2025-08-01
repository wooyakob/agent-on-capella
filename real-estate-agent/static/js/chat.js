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
        
        this.init();
    }
    
    async init() {
        try {
            await this.startSession();
            this.setupEventListeners();
        } catch (error) {
            this.showError('Failed to initialize chat session. Please refresh the page.');
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
            throw new Error(data.error);
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
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        const buyerName = this.buyerNameInput.value.trim();
        if (!message) return;
        
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.setLoading(true);
        
        try {
            const requestBody = { message: message };
            if (buyerName) {
                requestBody.buyer_name = buyerName;
            }
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.addMessage('agent', data.response, data.properties);
                
                if (data.profile_used) {
                    this.profileStatus.textContent = '✓ Profile Active';
                    this.profileStatus.style.color = '#059669';
                } else if (buyerName) {
                    this.profileStatus.textContent = '⚠ Profile Not Found';
                    this.profileStatus.style.color = '#dc2626';
                }
            } else {
                this.showError(data.error);
            }
        } catch (error) {
            this.showError('Network error. Please try again.');
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
        contentDiv.textContent = content;
        
        if (sender === 'agent') {
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
        } else {
            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(avatar);
        }
        
        if (properties && properties.length > 0) {
            const propertiesSection = document.createElement('div');
            propertiesSection.className = 'properties-section';
            
            const title = document.createElement('h4');
            title.textContent = '🏠 Matching Properties:';
            title.style.marginBottom = '10px';
            title.style.color = '#dc2626';
            propertiesSection.appendChild(title);
            
            properties.forEach(property => {
                const propertyCard = document.createElement('div');
                propertyCard.className = 'property-card';
                
                const propertyName = document.createElement('div');
                propertyName.className = 'property-name';
                propertyName.textContent = property.name;
                
                let propertyInfo = '';
                if (property.price && property.price !== 'N/A') {
                    propertyInfo += `💰 ${property.price}`;
                }
                if (property.bedrooms && property.bedrooms !== 'N/A') {
                    propertyInfo += ` • 🛏️ ${property.bedrooms} bed`;
                }
                if (property.bathrooms && property.bathrooms !== 'N/A') {
                    propertyInfo += ` • 🚿 ${property.bathrooms} bath`;
                }
                
                if (propertyInfo) {
                    const propertyDetails = document.createElement('div');
                    propertyDetails.style.cssText = 'font-size: 13px; color: #dc2626; font-weight: bold; margin-bottom: 8px;';
                    propertyDetails.textContent = propertyInfo;
                    propertyCard.appendChild(propertyName);
                    propertyCard.appendChild(propertyDetails);
                } else {
                    propertyCard.appendChild(propertyName);
                }
                
                if (property.address && property.address !== 'N/A') {
                    const propertyAddress = document.createElement('div');
                    propertyAddress.style.cssText = 'font-size: 12px; color: #666; margin-bottom: 8px; font-style: italic;';
                    propertyAddress.textContent = `📍 ${property.address}`;
                    propertyCard.appendChild(propertyAddress);
                }
                
                const propertyDesc = document.createElement('div');
                propertyDesc.className = 'property-description';
                propertyDesc.textContent = property.description.length > 150 ? 
                    property.description.substring(0, 150) + '...' : 
                    property.description;
                
                propertyCard.appendChild(propertyDesc);
                propertiesSection.appendChild(propertyCard);
            });
            
            contentDiv.appendChild(propertiesSection);
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    setLoading(loading) {
        this.sendButton.disabled = loading;
        this.messageInput.disabled = loading;
        
        if (loading) {
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'loading show';
            loadingDiv.id = 'responseLoading';
            loadingDiv.textContent = '🏠 Searching properties...';
            this.chatMessages.appendChild(loadingDiv);
            this.scrollToBottom();
        } else {
            const loadingDiv = document.getElementById('responseLoading');
            if (loadingDiv) {
                loadingDiv.remove();
            }
            this.messageInput.focus();
        }
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        this.chatMessages.appendChild(errorDiv);
        this.scrollToBottom();
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});
