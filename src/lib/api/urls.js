import apiClient from './client';

const endpoints = {
  urls: '/api/input/urls',
  uploadBase: '/api/input/upload',
  uploadText: '/api/input/upload/text',
  uploadJson: '/api/input/upload/json',
  uploadCsv: '/api/input/upload/csv',
  uploadExcel: '/api/input/upload/excel',
  list: '/api/input',
};

const getUploadEndpoint = (format) => {
  if (format === 'text') return endpoints.uploadText;
  if (format === 'json') return endpoints.uploadJson;
  if (format === 'csv') return endpoints.uploadCsv;
  return endpoints.uploadExcel;
};

export const urlApi = {
  validate(url, sessionId) {
    return apiClient.post(endpoints.urls, [url], {
      params: { session_id: sessionId },
    });
  },

  batch(file, format, sessionId, enrich = true) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    return apiClient.post(getUploadEndpoint(format), formData, {
      params: { enrich },
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  uploadAuto(file, { enrich = true, format, sessionId }) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    const params = { enrich };
    if (format) {
      params.format_hint = format;
    }

    return apiClient.post(endpoints.uploadBase, formData, {
      params,
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  getMetadata(id, sessionId) {
    return apiClient.get(`/api/input/${id}`, {
      params: { session_id: sessionId },
    });
  },

  list(sessionId, params = {}) {
    return apiClient.get(endpoints.list, {
      params: { session_id: sessionId, ...params },
    });
  },

  delete(id, sessionId) {
    return apiClient.delete(`/api/input/${id}`, {
      params: { session_id: sessionId },
    });
  },

  update(id, data, sessionId) {
    return apiClient.put(`/api/input/${id}`, data, {
      params: { session_id: sessionId },
    });
  },

  async getAllUrls(sessionId) {
    const inputs = await apiClient.get(endpoints.list, {
      params: { session_id: sessionId },
    });

    const allUrls = [];
    for (const input of inputs.data.inputs) {
      try {
        const inputDetails = await apiClient.get(`/api/input/${input.input_id}`, {
          params: { session_id: sessionId },
        });

        if (!inputDetails.data.urls) continue;

        inputDetails.data.urls.forEach((url, index) => {
          allUrls.push({
            id: `${input.input_id}:${url.source_metadata?.entry_id || index}`,
            url: url.url,
            status: url.validated
              ? url.enriched
                ? 'completed'
                : 'pending'
              : 'failed',
            title: url.source_metadata?.title || null,
            domain: url.metadata?.domain || null,
            created_at: input.created_at,
            input_id: input.input_id,
            category: url.category,
            priority: url.priority,
            notes: url.notes,
            validated: url.validated,
            validation_error: url.validation_error,
            metadata: url.metadata,
            source_metadata: url.source_metadata,
          });
        });
      } catch (error) {
        // eslint-disable-next-line no-console
        console.warn(`Failed to fetch details for input ${input.input_id}:`, error);
      }
    }

    return { data: allUrls };
  },
};

export default urlApi;
