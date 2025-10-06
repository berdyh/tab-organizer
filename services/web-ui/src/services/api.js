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
  validate: (url) => api.post('/urls/validate', { url }),
  batch: (file, format) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('format', format);
    return api.post('/urls/batch', formData);
  },
  getMetadata: (id) => api.get(`/urls/${id}/metadata`),
  list: (params) => api.get('/urls', { params }),
  delete: (id) => api.delete(`/urls/${id}`),
  update: (id, data) => api.put(`/urls/${id}`, data),
};

// Scraping API
export const scrapingAPI = {
  scrapeUrl: (url, sessionId, options) => 
    api.post('/scrape/url', { url, session_id: sessionId, options }),
  batchScrape: (urls, sessionId, parallelAuth = true) =>
    api.post('/scrape/batch', { urls, session_id: sessionId, parallel_auth: parallelAuth }),
  getStatus: (jobId) => api.get(`/scrape/status/${jobId}`),
  getJobs: () => api.get('/scrape/jobs'),
};

// Search API
export const searchAPI = {
  search: (query, searchType = 'semantic', filters = {}) =>
    api.get('/search', { params: { query, search_type: searchType, ...filters } }),
  getSuggestions: (query) => api.get('/search/suggestions', { params: { query } }),
};

// Session API
export const sessionAPI = {
  list: () => api.get('/sessions'),
  create: (name, description) => api.post('/sessions', { name, description }),
  get: (id) => api.get(`/sessions/${id}`),
  update: (id, data) => api.put(`/sessions/${id}`, data),
  delete: (id) => api.delete(`/sessions/${id}`),
  compare: (sessionIds) => api.post('/sessions/compare', { session_ids: sessionIds }),
};

// Export API
export const exportAPI = {
  export: (sessionId, format, options) =>
    api.post('/export', { session_id: sessionId, format, options }),
  getTemplates: (format) => api.get(`/export/templates/${format}`),
  getStatus: (jobId) => api.get(`/export/status/${jobId}`),
};

// Clustering API
export const clusteringAPI = {
  getClusters: (sessionId) => api.get(`/clusters/${sessionId}`),
  getClusterDetails: (clusterId) => api.get(`/clusters/details/${clusterId}`),
  visualize: (sessionId, type = '2d') => api.get(`/clusters/${sessionId}/visualize`, { params: { type } }),
};

// Chatbot API
export const chatbotAPI = {
  sendMessage: (sessionId, message, context) =>
    api.post('/chat/message', { session_id: sessionId, message, context }),
  getConversationHistory: (sessionId) => api.get(`/chat/history/${sessionId}`),
  clearHistory: (sessionId) => api.delete(`/chat/history/${sessionId}`),
  provideFeedback: (messageId, feedback) =>
    api.post('/chat/feedback', { message_id: messageId, feedback }),
};

export default api;