import apiClient from './client';

export const scrapingApi = {
  scrapeUrl(url, sessionId, options) {
    return apiClient.post('/api/scraper-service/scrape/url', {
      url,
      session_id: sessionId,
      options,
    });
  },

  batchScrape(urls, sessionId, parallelAuth = true) {
    return apiClient.post('/api/scraper-service/scrape/batch', {
      urls,
      session_id: sessionId,
      parallel_auth: parallelAuth,
    });
  },

  getStatus(jobId) {
    return apiClient.get(`/api/scraper-service/scrape/status/${jobId}`);
  },

  getJobs() {
    return apiClient.get('/api/scraper-service/scrape/jobs');
  },
};

export default scrapingApi;
