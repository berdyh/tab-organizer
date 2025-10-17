import { useCallback, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { authAPI } from '../../../lib/api';

export const defaultCredentialForm = {
  domain: '',
  loginUrl: '',
  authMethod: 'form',
  username: '',
  password: '',
};

export const defaultInteractiveForm = {
  domain: '',
  loginUrl: '',
  authMethod: 'form',
  browserType: 'playwright',
  headless: true,
  username: '',
  password: '',
};

const useAuthManager = () => {
  const queryClient = useQueryClient();
  const [credentialForm, setCredentialForm] = useState(defaultCredentialForm);
  const [interactiveForm, setInteractiveForm] = useState(defaultInteractiveForm);
  const [activeTaskId, setActiveTaskId] = useState(null);

  const refreshDomains = useCallback(() => {
    queryClient.invalidateQueries('auth.domains');
  }, [queryClient]);

  const refreshQueue = useCallback(() => {
    queryClient.invalidateQueries('auth.queue');
  }, [queryClient]);

  const refreshSessions = useCallback(() => {
    queryClient.invalidateQueries('auth.sessions');
  }, [queryClient]);

  const domainsQuery = useQuery('auth.domains', async () => {
    const response = await authAPI.listDomains();
    return response.data;
  });

  const queueQuery = useQuery(
    'auth.queue',
    async () => {
      const response = await authAPI.getQueueStatus();
      return response.data;
    },
    { refetchInterval: 5000 },
  );

  const sessionsQuery = useQuery(
    'auth.sessions',
    async () => {
      const response = await authAPI.listSessions();
      return response.data;
    },
    { refetchInterval: 10000 },
  );

  const taskQuery = useQuery(
    ['auth.task', activeTaskId],
    async () => {
      const response = await authAPI.getTaskStatus(activeTaskId);
      return response.data;
    },
    {
      enabled: Boolean(activeTaskId),
      refetchInterval: activeTaskId ? 4000 : false,
      onSuccess: (data) => {
        if (data?.status === 'completed' || data?.status === 'failed') {
          refreshSessions();
        }
      },
    },
  );

  const storeCredentialsMutation = useMutation(authAPI.storeCredentials, {
    onSuccess: () => {
      refreshDomains();
      setCredentialForm(defaultCredentialForm);
    },
  });

  const deleteCredentialsMutation = useMutation(authAPI.deleteCredentials, {
    onSuccess: () => {
      refreshDomains();
    },
  });

  const interactiveMutation = useMutation(authAPI.startInteractive, {
    onSuccess: (response) => {
      setActiveTaskId(response.data.task_id);
      refreshQueue();
    },
  });

  const cleanupSessionsMutation = useMutation(authAPI.cleanupSessions, {
    onSuccess: () => {
      refreshSessions();
    },
  });

  const domains = useMemo(() => domainsQuery.data?.domains ?? [], [domainsQuery.data]);
  const sessions = useMemo(() => sessionsQuery.data?.sessions ?? [], [sessionsQuery.data]);

  const submitCredentials = useCallback(
    async (payload) => {
      await storeCredentialsMutation.mutateAsync(payload);
    },
    [storeCredentialsMutation],
  );

  const removeCredentials = useCallback(
    async (domain) => {
      await deleteCredentialsMutation.mutateAsync(domain);
    },
    [deleteCredentialsMutation],
  );

  const startInteractive = useCallback(
    async (payload) => {
      await interactiveMutation.mutateAsync(payload);
    },
    [interactiveMutation],
  );

  const cleanupSessions = useCallback(async () => {
    await cleanupSessionsMutation.mutateAsync();
  }, [cleanupSessionsMutation]);

  const resetCredentialForm = useCallback(() => {
    setCredentialForm(defaultCredentialForm);
  }, []);

  const resetInteractiveForm = useCallback(() => {
    setInteractiveForm(defaultInteractiveForm);
  }, []);

  return {
    credentialForm,
    setCredentialForm,
    resetCredentialForm,
    interactiveForm,
    setInteractiveForm,
    resetInteractiveForm,
    activeTaskId,
    setActiveTaskId,
    domainsQuery,
    queueQuery,
    sessionsQuery,
    taskQuery,
    domains,
    sessions,
    submitCredentials,
    removeCredentials,
    startInteractive,
    cleanupSessions,
    refreshDomains,
    refreshQueue,
    refreshSessions,
    storeCredentialsMutation,
    deleteCredentialsMutation,
    interactiveMutation,
    cleanupSessionsMutation,
  };
};

export default useAuthManager;
