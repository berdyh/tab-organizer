import apiClient from './client';

export const sessionApi = {
  list() {
    return apiClient.get('/api/session-service/sessions');
  },

  create(name, description) {
    return apiClient.post('/api/session-service/sessions', { name, description });
  },

  get(id) {
    return apiClient.get(`/api/session-service/sessions/${id}`);
  },

  update(id, data) {
    return apiClient.put(`/api/session-service/sessions/${id}`, data);
  },

  delete(id) {
    return apiClient.delete(`/api/session-service/sessions/${id}`);
  },

  compare(sessionIds) {
    return apiClient.post('/api/session-service/sessions/compare', {
      session_ids: sessionIds,
    });
  },

  merge(sourceSessionIds, payload = {}) {
    return apiClient.post('/api/session-service/sessions/merge', {
      source_session_ids: sourceSessionIds,
      ...payload,
    });
  },

  split(sessionId, payload) {
    return apiClient.post(`/api/session-service/sessions/${sessionId}/split`, payload);
  },
};

export default sessionApi;
