import apiClient, { createFormData } from './client';

const getUploadEndpoint = (format) => {
  if (format === 'text') return 'text';
  if (format === 'json') return 'json';
  if (format === 'csv') return 'csv';
  return 'excel';
};

export const urlAPI = {
  validate: (url, sessionId) =>
    apiClient.post('/api/input/urls', [url], { params: { session_id: sessionId } }),

  batch: (file, format, sessionId, enrich = true) => {
    const formData = createFormData([
      ['file', file],
      ['session_id', sessionId],
    ]);
    const endpoint = getUploadEndpoint(format);

    return apiClient.post(`/api/input/upload/${endpoint}`, formData, {
      params: { enrich },
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  uploadAuto: (file, { enrich = true, format, sessionId }) => {
    const formData = createFormData([
      ['file', file],
      ['session_id', sessionId],
    ]);

    const params = { enrich };
    if (format) {
      params.format_hint = format;
    }

    return apiClient.post('/api/input/upload', formData, {
      params,
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  getMetadata: (id, sessionId) =>
    apiClient.get(`/api/input/${id}`, { params: { session_id: sessionId } }),

  list: (sessionId, params = {}) =>
    apiClient.get('/api/input', { params: { session_id: sessionId, ...params } }),

  delete: (id, sessionId) =>
    apiClient.delete(`/api/input/${id}`, { params: { session_id: sessionId } }),

  update: (id, data, sessionId) =>
    apiClient.put(`/api/input/${id}`, data, { params: { session_id: sessionId } }),

  async getAllUrls(sessionId) {
    const inputsResponse = await apiClient.get('/api/input', {
      params: { session_id: sessionId },
    });
    const inputs = inputsResponse.data?.inputs ?? [];

    if (!inputs.length) {
      return { data: [] };
    }

    const detailResponses = await Promise.allSettled(
      inputs.map((input) =>
        apiClient.get(`/api/input/${input.input_id}`, {
          params: { session_id: sessionId },
        }),
      ),
    );

    const aggregated = detailResponses.flatMap((result, index) => {
      if (result.status !== 'fulfilled') {
        console.warn(`Failed to fetch details for input ${inputs[index].input_id}`, result.reason);
        return [];
      }

      const urls = result.value.data?.urls ?? [];
      return urls.map((url, entryIndex) => {
        const entryId = url.source_metadata?.entry_id ?? entryIndex;
        const status = url.validated
          ? url.enriched
            ? 'completed'
            : 'pending'
          : 'failed';

        return {
          id: `${inputs[index].input_id}:${entryId}`,
          url: url.url,
          status,
          title: url.source_metadata?.title ?? null,
          domain: url.metadata?.domain ?? null,
          created_at: inputs[index].created_at,
          input_id: inputs[index].input_id,
          category: url.category,
          priority: url.priority,
          notes: url.notes,
          validated: url.validated,
          validation_error: url.validation_error,
          metadata: url.metadata,
          source_metadata: url.source_metadata,
        };
      });
    });

    return { data: aggregated };
  },
};

export default urlAPI;
