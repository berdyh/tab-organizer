import { useCallback, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { sessionAPI } from '../../../lib/api';

const useSessionsManager = () => {
  const queryClient = useQueryClient();
  const [selectedSessions, setSelectedSessions] = useState([]);

  const sessionsQuery = useQuery('sessions', sessionAPI.list);

  const refreshSessions = useCallback(() => {
    queryClient.invalidateQueries('sessions');
  }, [queryClient]);

  const deleteSessionMutation = useMutation(sessionAPI.delete, {
    onSuccess: () => {
      refreshSessions();
    },
  });

  const mergeSessionsMutation = useMutation(
    ({ sourceSessionIds, payload }) => sessionAPI.merge(sourceSessionIds, payload),
    {
      onSuccess: () => {
        refreshSessions();
      },
    },
  );

  const splitSessionMutation = useMutation(
    ({ sessionId, payload }) => sessionAPI.split(sessionId, payload),
    {
      onSuccess: () => {
        refreshSessions();
      },
    },
  );

  const toggleSessionSelection = useCallback((sessionId) => {
    setSelectedSessions((prev) =>
      prev.includes(sessionId)
        ? prev.filter((id) => id !== sessionId)
        : [...prev, sessionId],
    );
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedSessions([]);
  }, []);

  const deleteSessions = useCallback(async (sessionIds) => {
    await Promise.all(sessionIds.map((id) => deleteSessionMutation.mutateAsync(id)));
    clearSelection();
  }, [deleteSessionMutation, clearSelection]);

  const mergeSessions = useCallback(async ({ sourceSessionIds, payload }) => {
    await mergeSessionsMutation.mutateAsync({ sourceSessionIds, payload });
    clearSelection();
  }, [mergeSessionsMutation, clearSelection]);

  const splitSession = useCallback(async ({ sessionId, payload }) => {
    await splitSessionMutation.mutateAsync({ sessionId, payload });
  }, [splitSessionMutation]);

  return {
    sessions: sessionsQuery.data?.data ?? [],
    isLoading: sessionsQuery.isLoading,
    selectedSessions,
    toggleSessionSelection,
    clearSelection,
    deleteSessions,
    mergeSessions,
    splitSession,
    deleteSessionMutation,
    mergeSessionsMutation,
    splitSessionMutation,
    refreshSessions,
  };
};

export default useSessionsManager;
