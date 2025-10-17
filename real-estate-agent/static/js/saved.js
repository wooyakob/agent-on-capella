class SavedPage {
  constructor() {
    this.buyerNameInput = document.getElementById('buyerNameInput');
    this.buyerNamesDatalist = document.getElementById('buyerNames');
    this.propertiesList = document.getElementById('propertiesList');
    this.propertiesMeta = document.getElementById('propertiesMeta');
    this.refreshBtn = document.getElementById('refreshBtn');

    this.bind();
    this.loadBuyerNames();
  }


  bind() {
    if (this.refreshBtn) {
      this.refreshBtn.addEventListener('click', () => this.loadBuyerSaved());
    }
    if (this.buyerNameInput) {
      this.buyerNameInput.addEventListener('input', () => {
        if (this.buyerNameInput.value.trim()) this.loadBuyerSaved();
      });
    }
    if (!this.propertiesList.dataset.bound) {
      this.propertiesList.addEventListener('click', async (e) => {
        const btn = e.target.closest('button[data-action="delete"]');
        if (!btn) return;
        const key = btn.dataset.key || '';
        const buyerName = (this.buyerNameInput?.value || '').trim();
        if (!buyerName || !key) return;
        try {
          const res = await fetch('/api/buyer_saved', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ buyer_name: buyerName, key })
          });
          const data = await res.json();
          if (data.success && data.deleted) {
            btn.closest('.property-card')?.remove();
            const m = this.propertiesMeta?.textContent || '';
            const match = m.match(/^(\d+) saved/);
            if (match) {
              const count = Math.max(0, (parseInt(match[1], 10) || 1) - 1);
              this.propertiesMeta.textContent = `${count} saved for ${buyerName}`;
            }
          }
        } catch (err) { console.error('Delete failed', err); }
      });
      this.propertiesList.dataset.bound = '1';
    }
  }

  async loadBuyerNames() {
    try {
      const res = await fetch('/api/buyers');
      const data = await res.json();
      if (!data.success) return;
      const names = (data.buyers || []).map(b => b.buyer).filter(Boolean);
      this.buyerNamesDatalist.innerHTML = '';
      names.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        this.buyerNamesDatalist.appendChild(opt);
      });
    } catch {}
  }

  async loadBuyerSaved() {
    try {
      const buyerName = (this.buyerNameInput?.value || '').trim();
      if (!buyerName) {
        this.propertiesMeta.textContent = 'Select a buyer to view saved properties';
        this.propertiesList.innerHTML = '';
        return;
      }
      const res = await fetch(`/api/buyer_saved?buyer_name=${encodeURIComponent(buyerName)}`);
      const data = await res.json();
      if (!data.success) {
        this.propertiesMeta.textContent = '';
        this.propertiesList.innerHTML = '';
        return;
      }
      const properties = data.properties || [];
      this.propertiesMeta.textContent = `${properties.length} saved for ${buyerName}`;
      this.render(properties);
    } catch (e) {
      console.debug('Load buyer saved failed', e);
      this.propertiesList.innerHTML = '';
    }
  }

  formatPrice(val) {
    if (val == null) return 'Price not available';
    if (typeof val === 'number') return `$${val.toLocaleString()}`;
    const num = Number(String(val).replace(/[^0-9.]/g, ''));
    return Number.isFinite(num) && num > 0 ? `$${num.toLocaleString()}` : String(val);
  }

  render(properties) {
    this.propertiesList.innerHTML = '';
    if (!properties || properties.length === 0) {
      this.propertiesList.innerHTML = '<div class="empty">No saved properties</div>';
      return;
    }
    properties.forEach((p) => {
      const priceLabel = this.formatPrice(p.price);
      const key = p.dedupe_key || p.id || `${p.address ?? ''}|${p.name ?? ''}`;
      const card = document.createElement('div');
      card.className = 'property-card';
      card.innerHTML = `
        <div class="property-header">
          <h4>${p.name}</h4>
          <span class="property-price">${priceLabel}</span>
        </div>
        <div class="property-details">
          <div class="property-address">📍 ${p.address ?? ''}</div>
          <div class="property-specs">
            ${p.bedrooms ? `🛏️ ${p.bedrooms}bd/` : ''}${p.bathrooms ? `${p.bathrooms}ba` : ''} ${p.house_sqft ? `• ${p.house_sqft} sqft` : ''}
          </div>
          <div class="nearby" data-nearby="1"></div>
          <div class="property-actions">
            <button class="btn delete-btn" data-action="delete" data-key="${encodeURIComponent(key)}">Delete</button>
          </div>
        </div>
      `;
      this.propertiesList.appendChild(card);
      window.Nearby && window.Nearby.populate(card, p);
    });
  }
}

document.addEventListener('DOMContentLoaded', () => new SavedPage());
