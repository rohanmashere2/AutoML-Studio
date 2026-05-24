/**
 * AutoML Studio — Frontend Auth Helper
 * Injects Firebase ID token into all /api/ fetch requests.
 * 
 * Include this script AFTER Firebase SDK and BEFORE app.js / dashboard.js.
 * It monkey-patches window.fetch so every existing fetch() call automatically
 * gets the Authorization header — no changes needed in other JS files.
 */

(function () {
    'use strict';

    const _originalFetch = window.fetch;

    /**
     * Get the current Firebase user's ID token (cached for 5 min).
     * Returns null if not signed in or Firebase not loaded.
     */
    let _cachedToken = null;
    let _tokenExpiry = 0;

    async function getIdToken() {
        const now = Date.now();
        if (_cachedToken && now < _tokenExpiry) {
            return _cachedToken;
        }

        try {
            // Firebase Auth should already be initialized by login.html
            const { getAuth } = await import('https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js');
            const auth = getAuth();
            const user = auth.currentUser;

            if (!user) {
                // Not signed in — check localStorage fallback
                const stored = JSON.parse(localStorage.getItem('automl_user') || '{}');
                if (!stored.uid || stored.uid === 'anonymous') {
                    return null;
                }
                // User data exists but no Firebase auth state — might still be initializing
                // Wait briefly for auth state
                return new Promise((resolve) => {
                    const unsubscribe = auth.onAuthStateChanged((u) => {
                        unsubscribe();
                        if (u) {
                            u.getIdToken().then(token => {
                                _cachedToken = token;
                                _tokenExpiry = now + 5 * 60 * 1000; // Cache 5 min
                                resolve(token);
                            }).catch(() => resolve(null));
                        } else {
                            resolve(null);
                        }
                    });
                    // Timeout after 3 seconds
                    setTimeout(() => resolve(null), 3000);
                });
            }

            _cachedToken = await user.getIdToken();
            _tokenExpiry = now + 5 * 60 * 1000;
            return _cachedToken;
        } catch (e) {
            console.debug('[auth] Token fetch failed:', e.message);
            return null;
        }
    }

    /**
     * Patched fetch that injects Authorization header on /api/ requests.
     */
    window.fetch = async function (input, init) {
        const url = typeof input === 'string' ? input : (input instanceof Request ? input.url : String(input));

        // Only inject auth on API routes
        if (url.startsWith('/api/') || url.startsWith(window.location.origin + '/api/')) {
            const token = await getIdToken();
            if (token) {
                init = init || {};
                const headers = new Headers(init.headers || {});
                if (!headers.has('Authorization')) {
                    headers.set('Authorization', 'Bearer ' + token);
                }
                init.headers = headers;
            }
        }

        return _originalFetch.call(window, input, init);
    };

    // Expose for manual use if needed
    window.automlGetIdToken = getIdToken;

    /**
     * Clear cached token on sign-out.
     */
    window.addEventListener('storage', (e) => {
        if (e.key === 'automl_user' && !e.newValue) {
            _cachedToken = null;
            _tokenExpiry = 0;
        }
    });

    console.debug('[auth] Fetch interceptor installed — /api/ requests will include Firebase ID token');
})();
