import apiClient from './client';

export const scrapingAPI = {
  scrapeUrl: (url, sessionId, options) =>
    apiClient.post('/api/scraper-service/scrape/url', {
      url,
      session_id: sessionId,
      options,
    }),

  batchScrape: (urls, sessionId, parallelAuth = true) =>
    apiClient.post('/api/scraper-service/scrape/batch', {
      urls,
      session_id: sessionId,
      parallel_auth: parallelAuth,
    }),

  getStatus: (jobId) =>
    apiClient.get(`/api/scraper-service/scrape/status/${jobId}`),

  getJobs: () =>
    apiClient.get('/api/scraper-service/scrape/jobs'),
};

export default scrapingAPI;
