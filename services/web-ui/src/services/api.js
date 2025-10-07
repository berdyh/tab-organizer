import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// URL Management API
export const urlAPI = {
  validate: (url) => api.post('/api/url-input-service/input/urls', [url]),
  batch: (file, format, enrich = true) => {
    const formData = new FormData();
    formData.append('file', file);
    const endpoint = format === 'text' ? 'text' : 
                    format === 'json' ? 'json' : 
                    format === 'csv' ? 'csv' : 'excel';
    return api.post(`/api/url-input-service/input/upload/${endpoint}`, formData, {
      params: { enrich },
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  getMetadata: (id) => api.get(`/api/url-input-service/input/${id}`),
  list: (params) => api.get('/api/url-input-service/input/list', { params }),
  delete: (id) => api.delete(`/api/url-input-service/input/${id}`),
  update: (id, data) => api.put(`/api/url-input-service/input/${id}`, data),
};

// Scraping API
export const scrapingAPI = {
  scrapeUrl: (url, sessionId, options) => 
    api.post('/api/scraper-service/scrape/url', { url, session_id: sessionId, options }),
  batchScrape: (urls, sessionId, parallelAuth = true) =>
    api.post('/api/scraper-service/scrape/batch', { urls, session_id: sessionId, parallel_auth: parallelAuth }),
  getStatus: (jobId) => api.get(`/api/scraper-service/scrape/status/${jobId}`),
  getJobs: () => api.get('/api/scraper-service/scrape/jobs'),
};

// Search API
export const searchAPI = {
  search: (query, searchType = 'semantic', filters = {}) =>
    api.get('/api/analyzer-service/search', { params: { query, search_type: searchType, ...filters } }),
  getSuggestions: (query) => api.get('/api/analyzer-service/search/suggestions', { params: { query } }),
};

// Session API
export const sessionAPI = {
  list: () => api.get('/api/session-service/sessions'),
  create: (name, description) => api.post('/api/session-service/sessions', { name, description }),
  get: (id) => api.get(`/api/session-service/sessions/${id}`),
  update: (id, data) => api.put(`/api/session-service/sessions/${id}`, data),
  delete: (id) => api.delete(`/api/session-service/sessions/${id}`),
  compare: (sessionIds) => api.post('/api/session-service/sessions/compare', { session_ids: sessionIds }),
};

// Export API
export const exportAPI = {
  export: (sessionId, format, options) =>
    api.post('/api/export-service/export', { session_id: sessionId, format, options }),
  getTemplates: (format) => api.get(`/api/export-service/export/templates/${format}`),
  getStatus: (jobId) => api.get(`/api/export-service/export/status/${jobId}`),
};

// Clustering API
export const clusteringAPI = {
  getClusters: (sessionId) => api.get(`/api/clustering-service/clusters/${sessionId}`),
  getClusterDetails: (clusterId) => api.get(`/api/clustering-service/clusters/details/${clusterId}`),
  visualize: (sessionId, type = '2d') => api.get(`/api/clustering-service/clusters/${sessionId}/visualize`, { params: { type } }),
};

// Chatbot API
export const chatbotAPI = {
  sendMessage: (sessionId, message, context) =>
    api.post('/api/chatbot-service/chat/message', { session_id: sessionId, message, context }),
  getConversationHistory: (sessionId) => api.get(`/api/chatbot-service/chat/history/${sessionId}`),
  clearHistory: (sessionId) => api.delete(`/api/chatbot-service/chat/history/${sessionId}`),
  provideFeedback: (messageId, feedback) =>
    api.post('/api/chatbot-service/chat/feedback', { message_id: messageId, feedback }),
};

export default api;