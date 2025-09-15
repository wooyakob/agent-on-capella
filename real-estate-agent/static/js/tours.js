class ToursPage {
  constructor() {
    this.tourContext = document.getElementById('tourContext');
    this.toursContainer = document.getElementById('toursContainer');
    this.tourDetail = document.getElementById('tourDetail');
    this.tourProperty = document.getElementById('tourProperty');
    this.tourStatus = document.getElementById('tourStatus');
    this.tourPreferred = document.getElementById('tourPreferred');
    this.tourConfirmed = document.getElementById('tourConfirmed');
    this.preferredTimeInput = document.getElementById('preferredTimeInput');
    this.updatePreferredBtn = document.getElementById('updatePreferredBtn');
    this.buyerNameInput = document.getElementById('buyerNameInput');
    this.buyerNamesDatalist = document.getElementById('buyerNames');
    this.refreshToursBtn = document.getElementById('refreshToursBtn');
    this.deleteTourBtn = document.getElementById('deleteTourBtn');
    this.tourError = document.getElementById('tourError');
    this.confirmModal = document.getElementById('confirmModal');
    this.confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
    this.cancelDeleteBtn = document.getElementById('cancelDeleteBtn');
    this._pendingDeleteId = null;

    this.currentTour = null;
    this.init();
  }

  openConfirmModal() {
    if (!this.confirmModal) return;
    this.confirmModal.setAttribute('aria-hidden', 'false');
    this.confirmModal.style.display = 'flex';
  }

  closeConfirmModal() {
    if (!this.confirmModal) return;
    this.confirmModal.setAttribute('aria-hidden', 'true');
    this.confirmModal.style.display = 'none';
  }

  // Save confirmed time with validation (must be future and not before preferred)
  bindConfirmedTimeSave() {
    const btn = document.getElementById('updateConfirmedBtn');
    const input = document.getElementById('confirmedTimeInput');
    if (!btn || !input) return;
    btn.addEventListener('click', async () => {
      if (!this.currentTour) return;
      this.tourError.textContent = '';
      const confirmed = input.value || '';
      if (!confirmed) {
        this.tourError.textContent = 'Please select a confirmed date/time.';
        return;
      }
      const confirmedDt = new Date(confirmed);
      const now = new Date();
      if (isNaN(confirmedDt.getTime()) || confirmedDt <= now) {
        this.tourError.textContent = 'Confirmed time must be in the future.';
        return;
      }
      const preferredVal = this.preferredTimeInput?.value || '';
      if (preferredVal) {
        const preferredDt = new Date(preferredVal);
        if (!isNaN(preferredDt.getTime()) && confirmedDt < preferredDt) {
          this.tourError.textContent = 'Confirmed time cannot be earlier than preferred time.';
          return;
        }
      }
      try {
        const buyerName = (this.buyerNameInput?.value || '').trim();
        const qp = buyerName ? `?buyer_name=${encodeURIComponent(buyerName)}` : '';
        const res = await fetch(`/api/tours/${this.currentTour.id}${qp}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ confirmed_time: confirmed })
        });
        const data = await res.json();
        if (data.success) {
          await this.loadTour(this.currentTour.id);
          await this.loadToursList();
        }
      } catch (e) {
        console.error('Failed to save confirmed time', e);
      }
    });
  }

  formatPrice(val) {
    if (val == null) return 'Price not available';
    if (typeof val === 'number') return `$${val.toLocaleString()}`;
    const num = Number(String(val).replace(/[^0-9.]/g, ''));
    return Number.isFinite(num) && num > 0 ? `$${num.toLocaleString()}` : String(val);
  }

  buildBadges(property) {
    const badges = [];
    if (property?.bedrooms) badges.push(`🛏️ ${property.bedrooms} bd`);
    if (property?.bathrooms) badges.push(`🛁 ${property.bathrooms} ba`);
    if (property?.house_sqft) badges.push(`📐 ${property.house_sqft} sqft`);
    return badges;
  }

  renderPropertyCard(container, property) {
    container.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'property-card';
    const badges = this.buildBadges(property);
    const priceLabel = this.formatPrice(property.price);
    const mapLink = property?.location?.mapsLink || null;
    card.innerHTML = `
      <div class="property-header">
        <h4>${property.name || 'Property'}</h4>
        <span class="property-price">${priceLabel}</span>
      </div>
      <div class="property-details">
        <div class="property-address">📍 ${property.address ?? ''}</div>
        <div class="property-specs">
          ${property.bedrooms ? `🛏️ ${property.bedrooms}bd/` : ''}${property.bathrooms ? `${property.bathrooms}ba` : ''} ${property.house_sqft ? `• ${property.house_sqft} sqft` : ''}
        </div>
        ${badges.length ? `<div class="property-badges">${badges.map(b => `<span class=\"property-badge\">${b}</span>`).join(' ')}</div>` : ''}
        ${mapLink ? `<div class="property-map"><a href="${mapLink}" target="_blank" rel="noopener noreferrer">🗺️ View on map</a></div>` : ''}
        <div class="nearby" data-nearby="1"></div>
      </div>
    `;
    container.appendChild(card);
    this.populateNearby(card, property);
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
      // non-blocking
      console.debug('Nearby fetch failed', e);
    }
  }

  bindDetailActions() {
    if (!this.tourDetail) return;
    this.updatePreferredBtn?.addEventListener('click', async () => {
      if (!this.currentTour) return;
      const preferred = this.preferredTimeInput?.value || '';
      // Validation: must be future datetime if provided
      this.tourError.textContent = '';
      if (preferred) {
        const selected = new Date(preferred);
        const now = new Date();
        if (isNaN(selected.getTime()) || selected <= now) {
          this.tourError.textContent = 'Please select a future date/time for the preferred tour time.';
          return;
        }
      }
      try {
        const buyerName = (this.buyerNameInput?.value || '').trim();
        const qp = buyerName ? `?buyer_name=${encodeURIComponent(buyerName)}` : '';
        const res = await fetch(`/api/tours/${this.currentTour.id}${qp}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ confirmed_time: '', status: this.currentTour.status, preferred_time: preferred })
        });
        const data = await res.json();
        if (data.success) {
          await this.loadTour(this.currentTour.id);
        }
      } catch (e) { console.error(e); }
    });

    this.tourDetail?.addEventListener('click', async (e) => {
      const btn = e.target.closest('.tour-status-btn');
      if (!btn || !this.currentTour) return;
      const status = btn.getAttribute('data-status');
      try {
        const buyerName = (this.buyerNameInput?.value || '').trim();
        const qp = buyerName ? `?buyer_name=${encodeURIComponent(buyerName)}` : '';
        const res = await fetch(`/api/tours/${this.currentTour.id}${qp}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status })
        });
        const data = await res.json();
        if (data.success) {
          await this.loadTour(this.currentTour.id);
        }
      } catch (e) { console.error(e); }
    });

    // Delete button
    this.deleteTourBtn?.addEventListener('click', () => {
      if (!this.currentTour) return;
      this._pendingDeleteId = this.currentTour.id;
      this.openConfirmModal();
    });

    // Modal buttons
    this.confirmDeleteBtn?.addEventListener('click', async () => {
      const id = this._pendingDeleteId;
      if (!id) return this.closeConfirmModal();
      try {
        const buyerName = (this.buyerNameInput?.value || '').trim();
        const qp = buyerName ? `?buyer_name=${encodeURIComponent(buyerName)}` : '';
        const res = await fetch(`/api/tours/${id}${qp}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
          if (this.currentTour && this.currentTour.id === id) {
            this.currentTour = null;
            this.tourDetail.style.display = 'none';
          }
          await this.loadToursList();
        }
      } catch (e) {
        console.error('Failed to delete tour', e);
      } finally {
        this._pendingDeleteId = null;
        this.closeConfirmModal();
      }
    });
    this.cancelDeleteBtn?.addEventListener('click', () => {
      this._pendingDeleteId = null;
      this.closeConfirmModal();
    });
  }

  async loadTour(tourId) {
    try {
      const buyerName = (this.buyerNameInput?.value || '').trim();
      const qp = buyerName ? `?buyer_name=${encodeURIComponent(buyerName)}` : '';
      const res = await fetch(`/api/tours/${encodeURIComponent(tourId)}${qp}`);
      const data = await res.json();
      if (!data.success) return;
      const t = data.tour;
      this.currentTour = t;
      this.tourDetail.style.display = 'block';
      this.tourStatus.textContent = t.status || 'pending';
      this.tourPreferred.textContent = t.preferred_time || '-';
      this.tourConfirmed.textContent = t.confirmed_time || '-';
      this.renderPropertyCard(this.tourProperty, t.property || {});
    } catch (e) {
      console.error('Failed to load tour', e);
    }
  }

  renderToursList(tours) {
    this.toursContainer.innerHTML = '';
    if (!tours || tours.length === 0) {
      this.toursContainer.textContent = 'No tours yet.';
      return;
    }
    tours.forEach((t) => {
      const div = document.createElement('div');
      div.className = 'property-card';
      const p = t.property || {};
      const price = this.formatPrice(p.price);
      div.innerHTML = `
        <div class="property-header">
          <h4>${p.name || 'Property'}</h4>
          <span class="property-price">${price}</span>
        </div>
        <div class="property-details">
          <div class="property-address">📍 ${p.address ?? ''}</div>
          <div class="property-specs">${p.bedrooms ? `🛏️ ${p.bedrooms}bd/` : ''}${p.bathrooms ? `${p.bathrooms}ba` : ''} ${p.house_sqft ? `• ${p.house_sqft} sqft` : ''}</div>
          <div class="tour-meta">Status: <strong>${t.status}</strong> • Preferred: ${t.preferred_time || '-'} • Confirmed: ${t.confirmed_time || '-'}</div>
          <div class="property-actions">
            <a class="property-action-btn" href="/tours/${t.id}">View</a>
            <button class="property-action-btn" data-action="delete" data-id="${t.id}">Delete</button>
          </div>
        </div>
      `;
      this.toursContainer.appendChild(div);
    });

    // Bind delete buttons in list
    if (!this.toursContainer.dataset.bound) {
      this.toursContainer.addEventListener('click', async (e) => {
        const btn = e.target.closest('button[data-action="delete"]');
        if (!btn) return;
        const id = btn.dataset.id;
        if (!id) return;
        this._pendingDeleteId = id;
        this.openConfirmModal();
      });
      this.toursContainer.dataset.bound = '1';
    }
  }

  async loadToursList() {
    try {
      const buyerName = (this.buyerNameInput?.value || '').trim();
      const qp = buyerName ? `?buyer_name=${encodeURIComponent(buyerName)}` : '';
      const res = await fetch(`/api/tours${qp}`);
      const data = await res.json();
      if (data.success) {
        this.renderToursList(data.tours || []);
      }
    } catch (e) {
      console.error('Failed to load tours', e);
    }
  }

  async init() {
    const tourId = this.tourContext?.dataset?.tourId || '';
    this.bindPageControls();
    this.bindDetailActions();
    if (tourId) {
      await this.loadTour(tourId);
    }
    await this.loadToursList();
  }

  bindPageControls() {
    // Load buyers into datalist
    this.loadBuyerNames();
    // Refresh list when clicking button or changing buyer
    this.refreshToursBtn?.addEventListener('click', () => this.loadToursList());
    this.buyerNameInput?.addEventListener('input', () => this.loadToursList());
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
}

document.addEventListener('DOMContentLoaded', () => {
  new ToursPage();
});
