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

  async populateNearby(card, property) {
    try {
      const lat = property?.geo?.lat;
      const lon = property?.geo?.lon;
      const addr = property?.address || '';
      const qs = new URLSearchParams();
      if (typeof lat === 'number' && typeof lon === 'number') {
        qs.set('lat', String(lat));
        qs.set('lon', String(lon));
      } else if (addr) {
        qs.set('address', addr);
      }
      if ([...qs.keys()].length === 0) return;
      const res = await fetch(`/api/nearby?${qs.toString()}`);
      const data = await res.json();
      if (!data.success) return;
      const target = card.querySelector('.nearby');
      if (!target) return;
      const schools = data.schools || [];
      const restaurants = data.restaurants || [];
      const schoolHtml = schools.length ? `
        <div class="nearby-section">
          <div class="nearby-title">🎓 Nearby Schools</div>
          <ul class="nearby-list">
            ${schools.map(s => `<li><strong>${s.name}</strong> • ${s.rating ?? 'N/A'}⭐ • ${s.distance_km ?? '?'} km • ${s.address ?? ''}</li>`).join('')}
          </ul>
        </div>` : '';
      const restaurantHtml = restaurants.length ? `
        <div class="nearby-section">
          <div class="nearby-title">🍽️ Nearby Restaurants</div>
          <ul class="nearby-list">
            ${restaurants.map(r => `<li><strong>${r.name}</strong> • ${r.rating ?? 'N/A'}⭐ • ${r.distance_km ?? '?'} km • ${r.address ?? ''}</li>`).join('')}
          </ul>
        </div>` : '';
      target.innerHTML = schoolHtml + restaurantHtml;
    } catch (e) {
      console.debug('Nearby fetch failed', e);
    }
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
      this.populateNearby(card, p);
    });
  }
}

document.addEventListener('DOMContentLoaded', () => new SavedPage());
