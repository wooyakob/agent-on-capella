document.addEventListener('DOMContentLoaded', () => {
  try {
    const currentPath = window.location.pathname.replace(/\/+$/, '') || '/';
    const links = document.querySelectorAll('.top-nav a[href]');
    const topNav = document.querySelector('.top-nav');
    links.forEach((link) => {
      const url = new URL(link.getAttribute('href'), window.location.origin);
      const linkPath = url.pathname.replace(/\/+$/, '') || '/';
      const isHome = linkPath === '/' && currentPath === '/';
      const isSaved = linkPath.startsWith('/saved') && currentPath.startsWith('/saved');
      const isTours = linkPath.startsWith('/tours') && currentPath.startsWith('/tours');
      const isBuyers = linkPath.startsWith('/buyers') && currentPath.startsWith('/buyers');
      const exactMatch = linkPath === currentPath;
      if (isHome || isSaved || isTours || isBuyers || exactMatch) {
        link.setAttribute('aria-current', 'page');
        link.classList.add('active');
      } else {
        link.removeAttribute('aria-current');
        link.classList.remove('active');
      }
    });

    // Toggle stronger nav elevation when scrolled
    const onScroll = () => {
      if (!topNav) return;
      const scrolled = window.scrollY > 4;
      topNav.classList.toggle('scrolled', scrolled);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  } catch (e) {
    // no-op: nav highlighting is a progressive enhancement
    console.debug('nav active highlight skipped:', e);
  }
});
