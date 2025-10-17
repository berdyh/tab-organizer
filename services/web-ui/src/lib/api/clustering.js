import apiClient from './client';

export const clusteringAPI = {
  getClusters: (sessionId) =>
    apiClient.get(`/api/clustering-service/clusters/${sessionId}`),
  getClusterDetails: (clusterId) =>
    apiClient.get(`/api/clustering-service/clusters/details/${clusterId}`),
  visualize: (sessionId, type = '2d') =>
    apiClient.get(`/api/clustering-service/clusters/${sessionId}/visualize`, {
      params: { type },
    }),
};

export default clusteringAPI;
