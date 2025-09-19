'use strict';

class BuyersPage {
  constructor() {
    this.form = document.getElementById('buyerForm');
    this.status = document.getElementById('buyerFormStatus');
    this.buyersList = document.getElementById('buyersList');

    this.bind();
    this.loadBuyers();
  }

  parseNumeric(val) {
    if (val == null) return NaN;
    const cleaned = String(val).replace(/[^0-9.]/g, '');
    if (!cleaned) return NaN;
    const n = Number(cleaned);
    return Number.isFinite(n) ? n : NaN;
  }

  formatWithCommas(val) {
    const n = this.parseNumeric(val);
    if (!Number.isFinite(n)) return '';
    return Math.round(n).toLocaleString();
  }

  bind() {
    if (!this.form) return;

    // Pretty formatting on blur for budget fields
    const minEl = document.getElementById('budgetMinInput');
    const maxEl = document.getElementById('budgetMaxInput');
    [minEl, maxEl].forEach((el) => {
      if (!el) return;
      el.addEventListener('blur', () => {
        el.value = this.formatWithCommas(el.value);
      });
    });

    this.form.addEventListener('submit', async (e) => {
      e.preventDefault();
      this.status.textContent = '';

      const buyer = document.getElementById('buyerInput').value.trim();
      const location = document.getElementById('locationInput').value.trim();
      const bedroomsRaw = document.getElementById('bedroomsInput').value;
      const bathroomsRaw = document.getElementById('bathroomsInput').value;
      const minRaw = document.getElementById('budgetMinInput').value;
      const maxRaw = document.getElementById('budgetMaxInput').value;

      const bedrooms = this.parseNumeric(bedroomsRaw);
      const bathrooms = this.parseNumeric(bathroomsRaw);
      const min = this.parseNumeric(minRaw);
      const max = this.parseNumeric(maxRaw);

      const errs = [];
      if (!buyer) errs.push('Buyer is required');
      if (!location) errs.push('Location is required');
      if (!Number.isFinite(bedrooms)) errs.push('Bedrooms must be a number');
      if (!Number.isFinite(bathrooms)) errs.push('Bathrooms must be a number');
      if (!Number.isFinite(min) || !Number.isFinite(max)) errs.push('Budget min/max must be numbers');
      if (Number.isFinite(min) && Number.isFinite(max) && (min < 0 || max < 0 || min > max)) errs.push('Budget must be non-negative and min <= max');

      if (errs.length) {
        this.status.style.color = '#dc2626';
        this.status.textContent = errs.join('. ');
        return;
      }

      try {
        const res = await fetch('/api/buyers', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            buyer,
            location,
            bedrooms,
            bathrooms,
            budget: { min, max }
          })
        });
        const data = await res.json();
        if (!data.success) {
          const msg = (data.errors && data.errors.join(', ')) || data.error || 'Failed to create buyer';
          this.status.style.color = '#dc2626';
          this.status.textContent = msg;
          return;
        }
        this.status.style.color = '#16a34a';
        this.status.textContent = 'Buyer created! Redirecting to Home...';
        setTimeout(() => {
          window.location.href = '/';
        }, 800);
      } catch (e) {
        this.status.style.color = '#dc2626';
        this.status.textContent = 'Network error creating buyer';
      }
    });
  }

  async loadBuyers() {
    try {
      const res = await fetch('/api/buyers');
      const data = await res.json();
      const buyers = (data && data.success && Array.isArray(data.buyers)) ? data.buyers : [];
      if (!buyers.length) {
        this.buyersList.innerHTML = '<div class="empty">No buyers yet</div>';
        return;
      }
      this.buyersList.innerHTML = '';
      buyers.forEach((b) => {
        const item = document.createElement('div');
        item.className = 'list-item';
        const budgetStr = b.budget ? `$${Number(b.budget.min).toLocaleString()} - $${Number(b.budget.max).toLocaleString()}` : '—';
        item.innerHTML = `
          <span class="buyer-line">${b.buyer} • ${b.bedrooms}bd/${b.bathrooms}ba • ${b.location} • ${budgetStr}</span>
          <button class="btn delete-btn" data-action="delete-buyer" data-buyer="${b.buyer.replace(/"/g, '&quot;')}">Delete</button>
        `;
        this.buyersList.appendChild(item);
      });

      if (!this.buyersList.dataset.bound) {
        this.buyersList.addEventListener('click', async (e) => {
          const btn = e.target.closest('button[data-action="delete-buyer"]');
          if (!btn) return;
          const buyer = btn.getAttribute('data-buyer') || '';
          if (!buyer) return;
          if (!confirm(`Delete buyer "${buyer}"? This cannot be undone.`)) return;
          try {
            const res = await fetch('/api/buyers', {
              method: 'DELETE',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ buyer })
            });
            const data = await res.json();
            if (data && data.success) {
              btn.closest('.list-item')?.remove();
            } else {
              alert(data.error || 'Failed to delete buyer');
            }
          } catch (err) {
            alert('Network error deleting buyer');
          }
        });
        this.buyersList.dataset.bound = '1';
      }
    } catch (e) {
      this.buyersList.innerHTML = '<div class="empty">Unable to load buyers</div>';
    }
  }
}

window.addEventListener('DOMContentLoaded', () => new BuyersPage());
