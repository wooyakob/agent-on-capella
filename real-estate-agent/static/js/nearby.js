(function(){
  function formatNearbySections(data) {
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
    return schoolHtml + restaurantHtml;
  }

  async function fetchNearby(property) {
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
    if ([...qs.keys()].length === 0) return null;
    const res = await fetch(`/api/nearby?${qs.toString()}`);
    return await res.json();
  }

  async function populate(card, property) {
    try {
      const data = await fetchNearby(property);
      if (!data || !data.success) return;
      const target = card.querySelector('.nearby');
      if (!target) return;
      target.innerHTML = formatNearbySections(data);
    } catch (e) {
      console.debug('Nearby fetch failed', e);
    }
  }

  window.Nearby = { populate };
})();
