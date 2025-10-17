import { useCallback, useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { scrapingAPI, sessionAPI, urlAPI } from '../../../lib/api';
import { getErrorMessage } from '../../../shared/utils/errors';

const useUrlManager = () => {
  const queryClient = useQueryClient();
  const [selectedSession, setSelectedSession] = useState('');
  const [selectedUrls, setSelectedUrls] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [error, setError] = useState(null);

  const sessionsQuery = useQuery('sessions', sessionAPI.list, {
    staleTime: 60000,
  });

  const sessions = sessionsQuery.data?.data ?? [];

  useEffect(() => {
    if (!selectedSession && sessions.length > 0) {
      setSelectedSession(sessions[0].id);
    }
  }, [sessions, selectedSession]);

  useEffect(() => {
    setSelectedUrls([]);
  }, [selectedSession]);

  const urlsQuery = useQuery(
    ['urls', selectedSession],
    () => urlAPI.getAllUrls(selectedSession),
    {
      enabled: Boolean(selectedSession),
      onError: (err) => setError(getErrorMessage(err, 'Failed to load URLs')),
      retry: 1,
      staleTime: 30000,
    },
  );

  const urls = urlsQuery.data?.data ?? [];

  const refreshUrls = useCallback(() => {
    if (selectedSession) {
      queryClient.invalidateQueries(['urls', selectedSession]);
    }
  }, [queryClient, selectedSession]);

  const deleteUrlMutation = useMutation(
    ({ inputId, sessionId }) => urlAPI.delete(inputId, sessionId),
    {
      onSuccess: () => {
        refreshUrls();
        setError(null);
      },
      onError: (err) => {
        setError(getErrorMessage(err, 'Failed to delete URL'));
      },
    },
  );

  const scrapeMutation = useMutation(
    ({ urls: payload, sessionId }) => scrapingAPI.batchScrape(payload, sessionId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('scraping-jobs');
        setError(null);
      },
      onError: (err) => {
        setError(getErrorMessage(err, 'Failed to start scraping'));
      },
    },
  );

  const filteredUrls = useMemo(() => {
    if (!urls.length) {
      return [];
    }

    const term = searchTerm.toLowerCase();
    return urls.filter((url) => {
      const matchesSearch =
        url.url.toLowerCase().includes(term) ||
        (url.title && url.title.toLowerCase().includes(term));
      const matchesStatus = statusFilter === 'all' || url.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [urls, searchTerm, statusFilter]);

  const toggleSelectAll = useCallback(() => {
    if (selectedUrls.length === filteredUrls.length) {
      setSelectedUrls([]);
      return;
    }
    setSelectedUrls(filteredUrls.map((url) => url.id));
  }, [filteredUrls, selectedUrls.length]);

  const toggleSelectUrl = useCallback((urlId) => {
    setSelectedUrls((prev) =>
      prev.includes(urlId) ? prev.filter((id) => id !== urlId) : [...prev, urlId],
    );
  }, []);

  const handleDeleteSelected = useCallback(async () => {
    if (!selectedSession || !selectedUrls.length) {
      return;
    }

    const uniqueInputIds = Array.from(
      new Set(
        selectedUrls.map((urlId) =>
          urlId.includes(':') ? urlId.split(':')[0] : urlId,
        ),
      ),
    );

    try {
      await Promise.all(
        uniqueInputIds.map((inputId) =>
          deleteUrlMutation.mutateAsync({ inputId, sessionId: selectedSession }),
        ),
      );
      setSelectedUrls([]);
    } catch (err) {
      // handled via mutation
    }
  }, [selectedSession, selectedUrls, deleteUrlMutation]);

  const handleScrapeSelected = useCallback(() => {
    if (!selectedSession) {
      setError('Please select a session before starting a scrape.');
      return;
    }

    if (!selectedUrls.length) {
      return;
    }

    const selectedUrlObjects = filteredUrls.filter((url) =>
      selectedUrls.includes(url.id),
    );

    if (!selectedUrlObjects.length) {
      return;
    }

    const payload = selectedUrlObjects.map((url) => {
      const entryId = url.id.includes(':') ? url.id.split(':')[1] : undefined;
      const metadata = {
        input_id: url.input_id,
        entry_id: entryId,
        domain: url.domain,
        notes: url.notes,
      };

      if (url.metadata && typeof url.metadata === 'object') {
        Object.assign(metadata, url.metadata);
      }

      if (url.source_metadata && typeof url.source_metadata === 'object') {
        metadata.source = url.source_metadata;
      }

      return {
        url: url.url,
        priority: url.priority || 1,
        metadata,
      };
    });

    scrapeMutation.mutate({ urls: payload, sessionId: selectedSession });
  }, [selectedSession, selectedUrls, filteredUrls, scrapeMutation]);

  const clearError = useCallback(() => setError(null), []);

  const isLoading =
    sessionsQuery.isLoading ||
    urlsQuery.isLoading ||
    (sessions.length > 0 && !selectedSession);

  return {
    sessions,
    selectedSession,
    setSelectedSession,
    selectedUrls,
    toggleSelectUrl,
    toggleSelectAll,
    searchTerm,
    setSearchTerm,
    statusFilter,
    setStatusFilter,
    filteredUrls,
    isLoading,
    isSessionsLoading: sessionsQuery.isLoading,
    isUrlsLoading: urlsQuery.isLoading,
    error,
    clearError,
    setError,
    deleteUrlMutation,
    scrapeMutation,
    handleDeleteSelected,
    handleScrapeSelected,
    refreshUrls,
  };
};

export default useUrlManager;
