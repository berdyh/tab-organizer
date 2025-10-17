import apiClient from './client';

export const authAPI = {
  listDomains: () =>
    apiClient.get('/api/auth/domains'),
  storeCredentials: (payload) =>
    apiClient.post('/api/auth/store-credentials', payload),
  getCredentials: (domain) =>
    apiClient.get(`/api/auth/credentials/${domain}`),
  deleteCredentials: (domain) =>
    apiClient.delete(`/api/auth/credentials/${domain}`),
  learnDomain: (payload) =>
    apiClient.post('/api/auth/learn-domain', payload),
  markSuccess: (domain) =>
    apiClient.post(`/api/auth/mark-success/${domain}`),
  markFailure: (domain) =>
    apiClient.post(`/api/auth/mark-failure/${domain}`),
  getDomainMapping: (domain) =>
    apiClient.get(`/api/auth/domain-mapping/${domain}`),
  startInteractive: (payload) =>
    apiClient.post('/api/auth/interactive', payload),
  getTaskStatus: (taskId) =>
    apiClient.get(`/api/auth/task/${taskId}`),
  getQueueStatus: () =>
    apiClient.get('/api/auth/queue/status'),
  listSessions: () =>
    apiClient.get('/api/auth/sessions'),
  cleanupSessions: () =>
    apiClient.post('/api/auth/sessions/cleanup'),
  getSession: (sessionId) =>
    apiClient.get(`/api/auth/session/${sessionId}`),
  renewSession: (sessionId) =>
    apiClient.post(`/api/auth/session/${sessionId}/renew`),
  invalidateSession: (sessionId) =>
    apiClient.delete(`/api/auth/session/${sessionId}`),
};

export default authAPI;
