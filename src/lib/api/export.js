import apiClient from './client';

export const exportApi = {
  export(payload) {
    return apiClient.post('/api/export-service/export', payload);
  },

  getTemplates(format) {
    return apiClient.get(`/api/export-service/export/templates/${format}`);
  },

  getStatus(jobId) {
    return apiClient.get(`/api/export-service/export/status/${jobId}`);
  },
};

export default exportApi;
