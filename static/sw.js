const CACHE='eidomira-shell-v3';
const SHELL=[
  '/',
  '/static/fonts.css',
  '/static/tokens.css',
  '/static/landing.css',
  '/static/studio.css',
  '/static/landing.js',
  '/static/app.js',
  '/static/manifest.webmanifest',
  '/static/favicon-32.png',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/apple-touch-icon.png',
  '/static/hero-dark.jpg',
  '/static/hero-identity.jpg',
  '/static/persona-01.jpg',
  '/static/persona-02.jpg',
  '/static/persona-03.jpg',
  '/static/persona-04.jpg',
  '/static/fonts/inter-tight-latin-wght-normal.woff2',
  '/static/fonts/instrument-serif-latin-400-normal.woff2',
  '/static/fonts/instrument-serif-latin-400-italic.woff2'
];
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL).catch(()=>{})).then(()=>self.skipWaiting())));
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',e=>{
  const url=new URL(e.request.url);
  if(e.request.method!=='GET'||url.origin!==self.location.origin||url.pathname.startsWith('/api/'))return;
  // Fonts are content-addressed: cache-first. Everything else: network-first with offline fallback.
  if(url.pathname.startsWith('/static/fonts/')){
    e.respondWith(caches.match(e.request).then(hit=>hit||fetch(e.request).then(r=>{const copy=r.clone();caches.open(CACHE).then(c=>c.put(e.request,copy));return r})));
    return;
  }
  e.respondWith(fetch(e.request).then(r=>{const copy=r.clone();caches.open(CACHE).then(c=>c.put(e.request,copy));return r}).catch(()=>caches.match(e.request)));
});
