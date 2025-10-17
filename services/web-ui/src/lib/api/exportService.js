import apiClient from './client';

export const exportAPI = {
  export: (payload) =>
    apiClient.post('/api/export-service/export', payload),
  getTemplates: (format) =>
    apiClient.get(`/api/export-service/export/templates/${format}`),
  getStatus: (jobId) =>
    apiClient.get(`/api/export-service/export/status/${jobId}`),
};

export default exportAPI;
