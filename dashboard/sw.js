// Simple PWA service worker: cache the shell, never cache API answers.
const CACHE = "foodflow-dash-v1";
const SHELL = ["/", "/index.html", "/manifest.webmanifest", "/icon.png"];
self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(
    ks.filter(k => k !== CACHE).map(k => caches.delete(k)))));
});
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (u.pathname.startsWith("/api/")) return;              // live data: network only
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});
