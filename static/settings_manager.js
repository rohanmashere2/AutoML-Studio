/**
 * AutoML Studio — Settings Manager
 * Reads the user's profile settings from localStorage and applies them to the DOM.
 */

(function () {
    'use strict';

    function applySettings() {
        try {
            const raw = localStorage.getItem('automl_profile_settings');
            const s = raw ? JSON.parse(raw) : {};

            // Expose globally
            window.automlSettings = {
                webTheme: s.webTheme || 'auto',
                language: s.language || 'en',
                autoRefresh: s.autoRefresh !== false,
                reduceMotion: !!s.reduceMotion,
                newTabDownloads: s.newTabDownloads !== false,
                darkTheme: !!s.darkTheme,
                emailNotifications: !!s.emailNotifications,
                compactMode: !!s.compactMode,
                autoSave: s.autoSave !== false
            };

            // 1. Apply Theme
            // webTheme overrides darkTheme toggle if set specifically
            let useDark = window.automlSettings.darkTheme;
            if (window.automlSettings.webTheme === 'dark') useDark = true;
            if (window.automlSettings.webTheme === 'light') useDark = false;
            
            if (useDark) {
                document.documentElement.style.setProperty('--bg0', '#0F172A');
                document.documentElement.style.setProperty('--bg1', '#1E293B');
                document.documentElement.style.setProperty('--bg2', '#0A0F1E');
                document.documentElement.style.setProperty('--txt', '#F1F5F9');
                document.documentElement.style.setProperty('--txt2', '#94A3B8');
                document.documentElement.style.setProperty('--bdr', '#1E293B');
                document.documentElement.style.setProperty('--bdr2', '#334155');
            } else {
                document.documentElement.style.setProperty('--bg0', '#ffffff');
                document.documentElement.style.setProperty('--bg1', '#F8FAFC');
                document.documentElement.style.setProperty('--bg2', '#F1F5F9');
                document.documentElement.style.setProperty('--txt', '#0F172A');
                document.documentElement.style.setProperty('--txt2', '#64748B');
                document.documentElement.style.setProperty('--bdr', '#E2E8F0');
                document.documentElement.style.setProperty('--bdr2', '#CBD5E1');
            }

            // 2. Compact Mode
            if (window.automlSettings.compactMode) {
                document.documentElement.classList.add('compact-mode');
            } else {
                document.documentElement.classList.remove('compact-mode');
            }

            // 3. Reduce Motion
            if (window.automlSettings.reduceMotion) {
                document.documentElement.classList.add('reduce-motion');
            } else {
                document.documentElement.classList.remove('reduce-motion');
            }
            
        } catch (e) {
            console.error('[Settings] Error applying settings:', e);
            window.automlSettings = { autoRefresh: true, newTabDownloads: true };
        }
    }

    // Apply immediately
    applySettings();

    // 4. New Tab Downloads interceptor
    document.addEventListener('DOMContentLoaded', () => {
        document.body.addEventListener('click', (e) => {
            const target = e.target.closest('a');
            if (target && target.href && window.automlSettings && window.automlSettings.newTabDownloads) {
                // If it looks like a download link (b2, aws, or api/download)
                if (target.href.includes('b2') || target.href.includes('api/download') || target.hasAttribute('download')) {
                    target.target = '_blank';
                }
            }
        });
    });

    // Expose reload function
    window.applyAutoMLSettings = applySettings;
})();
