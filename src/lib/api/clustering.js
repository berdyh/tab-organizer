import apiClient from './client';

export const clusteringApi = {
  getClusters(sessionId) {
    return apiClient.get(`/api/clustering-service/clusters/${sessionId}`);
  },

  getClusterDetails(clusterId) {
    return apiClient.get(`/api/clustering-service/clusters/details/${clusterId}`);
  },

  visualize(sessionId, type = '2d') {
    return apiClient.get(`/api/clustering-service/clusters/${sessionId}/visualize`, {
      params: { type },
    });
  },
};

export default clusteringApi;
