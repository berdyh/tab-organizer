import apiClient from './client';

export const searchAPI = {
  search: (query, searchType = 'semantic', filters = {}) =>
    apiClient.get('/api/analyzer-service/search', {
      params: { query, search_type: searchType, ...filters },
    }),

  getSuggestions: (query) =>
    apiClient.get('/api/analyzer-service/search/suggestions', {
      params: { query },
    }),
};

export default searchAPI;
