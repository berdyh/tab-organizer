import apiClient from './client';

export const searchApi = {
  search(query, searchType = 'semantic', filters = {}) {
    return apiClient.get('/api/analyzer-service/search', {
      params: { query, search_type: searchType, ...filters },
    });
  },

  getSuggestions(query) {
    return apiClient.get('/api/analyzer-service/search/suggestions', {
      params: { query },
    });
  },
};

export default searchApi;
