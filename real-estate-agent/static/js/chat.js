class ChatApp {
    constructor() {
        this.sessionId = null;
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.buyerNameInput = document.getElementById('buyerNameInput');
        this.buyerNamesDatalist = document.getElementById('buyerNames');
        this.profileStatus = document.getElementById('profileStatus');
        this.sendButton = document.getElementById('sendButton');
        this.chatForm = document.getElementById('chatForm');
        this.initialLoading = document.getElementById('initialLoading');
        this.clearHistoryButton = document.getElementById('clearHistoryButton');
        this.quickPrompts = document.getElementById('quickPrompts');
        this.tabMatches = document.getElementById('tabMatches');
        this.tabSaved = document.getElementById('tabSaved');
        this.propertiesPanel = document.getElementById('propertiesPanel');
        this.propertiesList = document.getElementById('propertiesList');
        this.propertiesMeta = document.getElementById('propertiesMeta');
        this.currentTab = 'matches'; // 'matches' | 'saved'
        
        this.init();
    }

    formatPrice(val) {
        if (val == null) return 'Price not available';
        // Accept either number or strings like "$950,000"
        if (typeof val === 'number') return `$${val.toLocaleString()}`;
        const num = Number(String(val).replace(/[^0-9.]/g, ''));
        return Number.isFinite(num) && num > 0 ? `$${num.toLocaleString()}` : String(val);
    }

    buildBadges(property) {
        const badges = [];
        if (property?.bedrooms) badges.push(`🛏️ ${property.bedrooms} bd`);
        if (property?.bathrooms) badges.push(`🛁 ${property.bathrooms} ba`);
        if (property?.house_sqft) badges.push(`📐 ${property.house_sqft} sqft`);
        // Light heuristic: high similarity highlights
        const score = typeof property?.similarity_score === 'number' ? property.similarity_score : null;
        if (score != null && score >= 0.75) badges.push('⭐ Top match');
        return badges;
    }

    async loadSavedProperties() {
        try {
            const res = await fetch('/api/saved_properties');
            const data = await res.json();
            if (!data.success) {
                this.propertiesMeta && (this.propertiesMeta.textContent = '');
                this.renderProperties([]);
                return;
            }
            const saved = data.properties || [];
            this.propertiesMeta && (this.propertiesMeta.textContent = `${saved.length} saved`);
            this.renderProperties(saved);
        } catch (e) {
            console.debug('Failed to load saved properties', e);
            this.renderProperties([]);
        }
    }

    setupTabs() {
        if (!this.tabMatches || !this.tabSaved) return;
        const activate = (tab) => {
            this.currentTab = tab;
            this.tabMatches.classList.toggle('active', tab === 'matches');
            this.tabMatches.setAttribute('aria-selected', tab === 'matches' ? 'true' : 'false');
            this.tabSaved.classList.toggle('active', tab === 'saved');
            this.tabSaved.setAttribute('aria-selected', tab === 'saved' ? 'true' : 'false');
        };
        this.tabMatches.addEventListener('click', () => {
            activate('matches');
            // Expect the next agent response to populate matches; leave current list as-is
        });
        this.tabSaved.addEventListener('click', () => {
            activate('saved');
            this.loadSavedProperties();
        });
    }

    renderProperties(properties) {
        const panel = this.propertiesPanel;
        const list = this.propertiesList;
        const meta = this.propertiesMeta;
        if (!panel || !list) return;
        if (!properties || properties.length === 0) {
            list.innerHTML = '';
            meta && (meta.textContent = this.currentTab === 'matches' ? '' : meta.textContent);
            panel.style.display = 'none';
            return;
        }
        panel.style.display = 'block';
        list.innerHTML = '';
        if (this.currentTab === 'matches') {
            meta && (meta.textContent = `Showing ${Math.min(3, properties.length)} of ${properties.length}`);
        }
        properties.slice(0, 3).forEach((property) => {
            const propertyCard = document.createElement('div');
            propertyCard.className = 'property-card';
            const matchScore = (typeof property.similarity_score === 'number')
                ? property.similarity_score.toFixed(2)
                : (property.similarity_score || 'N/A');
            const badges = this.buildBadges(property);
            const priceLabel = this.formatPrice(property.price);
            const mapUrl = property?.location?.embedUrl || null;
            const propPayload = encodeURIComponent(JSON.stringify(property));
            propertyCard.innerHTML = `
                <div class="property-header">
                    <h4>${property.name}</h4>
                    <span class="property-price">${priceLabel}</span>
                </div>
                <div class="property-details">
                    <div class="property-address">📍 ${property.address ?? ''}</div>
                    <div class="property-specs">
                        ${property.bedrooms ? `🛏️ ${property.bedrooms}bd/` : ''}${property.bathrooms ? `${property.bathrooms}ba` : ''} ${property.house_sqft ? `• ${property.house_sqft} sqft` : ''}
                    </div>
                    ${badges.length ? `<div class="property-badges">${badges.map(b => `<span class=\"property-badge\">${b}</span>`).join(' ')}</div>` : ''}
                    <div class="property-description">${(property.description || '').substring(0, 180)}...</div>
                    ${this.currentTab === 'matches' ? `<div class=\"property-score\">🎯 Match score: ${matchScore}</div>` : ''}
                    ${mapUrl ? `<div class=\"property-map\"><iframe loading=\"lazy\" allowfullscreen src=\"${mapUrl}\"></iframe></div>` : ''}
                    <div class="property-actions">
                        <button class="property-action-btn" data-action="save" data-prop="${propPayload}">Save</button>
                        <button class="property-action-btn" data-action="hide" data-prop="${propPayload}">Hide</button>
                        <button class="property-action-btn" data-action="tour" data-prop="${propPayload}">Request tour</button>
                    </div>
                </div>
            `;
            list.appendChild(propertyCard);
        });

        // Delegate actions
        if (!panel.dataset.bound) {
            panel.addEventListener('click', async (e) => {
                const btn = e.target.closest('button.property-action-btn');
                if (!btn) return;
                const action = btn.dataset.action;
                const propStr = btn.dataset.prop;
                let prop;
                try { prop = JSON.parse(decodeURIComponent(propStr)); } catch { prop = null; }
                if (!prop || !action) return;
                if (action === 'save' || action === 'hide') {
                    const url = action === 'save' ? '/api/save_property' : '/api/hide_property';
                    try {
                        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ property: prop }) });
                        const data = await res.json();
                        if (!data.success) {
                            console.warn('Action failed', data.error);
                        } else {
                            btn.textContent = action === 'save' ? 'Saved ✓' : 'Hidden ✓';
                            btn.disabled = true;
                            if (this.currentTab === 'saved' && action === 'hide') {
                                // If hiding from saved view, remove it from the list visually
                                const card = btn.closest('.property-card');
                                card && card.remove();
                            }
                        }
                    } catch (err) { console.error(err); }
                } else if (action === 'tour') {
                    alert('Tour request placeholder — we can wire this to an email/CRM next.');
                }
            });
            panel.dataset.bound = '1';
        }
    }

    async loadBuyerNames() {
        try {
            const res = await fetch('/api/buyers');
            const data = await res.json();
            if (!data.success) return;
            const names = (data.buyers || []).map(b => b.buyer).filter(Boolean);
            // Clear existing options
            this.buyerNamesDatalist.innerHTML = '';
            names.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                this.buyerNamesDatalist.appendChild(opt);
            });
        } catch (e) {
            // Non-blocking
            console.debug('Failed to load buyer names', e);
        }
    }
    
    async init() {
        try {
            await this.startSession();
            this.loadBuyerNames();
            this.setupEventListeners();
            this.setupTabs();
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
                        <div><strong>Property Search:</strong> Vector search for dream properties</div>
                        <div><strong>Market Research:</strong> Web search for current information</div>
                        <div><strong>Expert Advice:</strong> LLM for real estate guidance</div>
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

        this.messageInput.addEventListener('input', () => {
            const hasText = this.messageInput.value.trim().length > 0;
            this.sendButton.disabled = !hasText;
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

        if (this.quickPrompts) {
            this.quickPrompts.addEventListener('click', (e) => {
                const btn = e.target.closest('.quick-prompt');
                if (!btn) return;
                const text = btn.getAttribute('data-text') || '';
                if (!text) return;
                this.messageInput.value = text;
                this.sendButton.disabled = false;
                this.sendMessage();
            });
        }
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
        
        this.chatMessages.appendChild(messageDiv);

        // Render properties only if the Matches tab is active
        if (this.currentTab === 'matches') {
            if (properties && properties.length > 0) {
                this.renderProperties(properties);
            } else {
                // hide panel if no properties for this turn
                this.renderProperties([]);
            }
        }
        
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
