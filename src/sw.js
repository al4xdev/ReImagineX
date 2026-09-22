const CACHE_NAME = 'reimaginex-cache-v3';
const ASSETS = [
  '/',
  '/icon.svg',
  '/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS).catch((err) => {
        console.warn('Failed to pre-cache some assets:', err);
      });
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

async function fetchAndCache(request) {
  const response = await fetch(request);
  if (response && response.status === 200 && response.type === 'basic') {
    const cache = await caches.open(CACHE_NAME);
    await cache.put(request, response.clone());
  }
  return response;
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') {
    return;
  }
  if (request.url.includes('/api/') || request.url.includes('/ws')) {
    return;
  }

  // The app shell and the document itself must never be served stale, otherwise
  // frontend fixes would never reach an installed PWA. Everything else stays
  // cache-first.
  const freshFirst = request.mode === 'navigate' || new URL(request.url).pathname === '/';

  event.respondWith((async () => {
    if (!freshFirst) {
      const cached = await caches.match(request);
      if (cached) {
        return cached;
      }
    }
    try {
      return await fetchAndCache(request);
    } catch (err) {
      const cached = await caches.match(request);
      // respondWith must always resolve to a Response, never undefined.
      return cached || new Response('Offline', { status: 503, statusText: 'Offline' });
    }
  })());
});
