import apiClient from './client';

export const sessionAPI = {
  list: () => apiClient.get('/api/session-service/sessions'),
  create: (name, description) =>
    apiClient.post('/api/session-service/sessions', { name, description }),
  get: (id) => apiClient.get(`/api/session-service/sessions/${id}`),
  update: (id, data) =>
    apiClient.put(`/api/session-service/sessions/${id}`, data),
  delete: (id) =>
    apiClient.delete(`/api/session-service/sessions/${id}`),
  compare: (sessionIds) =>
    apiClient.post('/api/session-service/sessions/compare', {
      session_ids: sessionIds,
    }),
  merge: (sourceSessionIds, payload = {}) =>
    apiClient.post('/api/session-service/sessions/merge', {
      source_session_ids: sourceSessionIds,
      ...payload,
    }),
  split: (sessionId, payload) =>
    apiClient.post(`/api/session-service/sessions/${sessionId}/split`, payload),
};

export default sessionAPI;
