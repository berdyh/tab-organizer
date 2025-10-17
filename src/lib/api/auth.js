import apiClient from './client';

export const authApi = {
  listDomains() {
    return apiClient.get('/api/auth/domains');
  },

  storeCredentials(payload) {
    return apiClient.post('/api/auth/store-credentials', payload);
  },

  getCredentials(domain) {
    return apiClient.get(`/api/auth/credentials/${domain}`);
  },

  deleteCredentials(domain) {
    return apiClient.delete(`/api/auth/credentials/${domain}`);
  },

  learnDomain(payload) {
    return apiClient.post('/api/auth/learn-domain', payload);
  },

  markSuccess(domain) {
    return apiClient.post(`/api/auth/mark-success/${domain}`);
  },

  markFailure(domain) {
    return apiClient.post(`/api/auth/mark-failure/${domain}`);
  },

  getDomainMapping(domain) {
    return apiClient.get(`/api/auth/domain-mapping/${domain}`);
  },

  startInteractive(payload) {
    return apiClient.post('/api/auth/interactive', payload);
  },

  getTaskStatus(taskId) {
    return apiClient.get(`/api/auth/task/${taskId}`);
  },

  getQueueStatus() {
    return apiClient.get('/api/auth/queue/status');
  },

  listSessions() {
    return apiClient.get('/api/auth/sessions');
  },

  cleanupSessions() {
    return apiClient.post('/api/auth/sessions/cleanup');
  },

  getSession(sessionId) {
    return apiClient.get(`/api/auth/session/${sessionId}`);
  },

  renewSession(sessionId) {
    return apiClient.post(`/api/auth/session/${sessionId}/renew`);
  },

  invalidateSession(sessionId) {
    return apiClient.delete(`/api/auth/session/${sessionId}`);
  },
};

export default authApi;
