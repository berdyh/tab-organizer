import React, { useCallback, useState } from 'react';
import ErrorAlert from '../shared/components/ErrorAlert';
import { getErrorMessage } from '../shared/utils/errors';
import {
  AuthHeader,
  CredentialForm,
  DomainsTable,
  QueueStatusCard,
  SessionsList,
  useAuthManager,
} from '../features/auth';

const AuthManager = () => {
  const [error, setError] = useState(null);

  const {
    credentialForm,
    setCredentialForm,
    interactiveForm,
    setInteractiveForm,
    activeTaskId,
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
  } = useAuthManager();

  const updateCredentialForm = useCallback((field, value) => {
    setCredentialForm((prev) => ({ ...prev, [field]: value }));
  }, [setCredentialForm]);

  const updateInteractiveForm = useCallback((field, value) => {
    setInteractiveForm((prev) => ({ ...prev, [field]: value }));
  }, [setInteractiveForm]);

  const handleCredentialSubmit = useCallback(async () => {
    if (!credentialForm.domain || !credentialForm.loginUrl) {
      setError('Domain and login URL are required to store credentials.');
      return;
    }

    try {
      await submitCredentials({
        domain: credentialForm.domain.trim(),
        auth_method: credentialForm.authMethod,
        login_url: credentialForm.loginUrl.trim(),
        credentials: {
          username: credentialForm.username,
          password: credentialForm.password,
        },
      });
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to store credentials'));
    }
  }, [credentialForm, submitCredentials]);

  const handleInteractiveSubmit = useCallback(async () => {
    if (!interactiveForm.domain || !interactiveForm.loginUrl) {
      setError('Domain and login URL are required to start interactive authentication.');
      return;
    }

    try {
      await startInteractive({
        domain: interactiveForm.domain.trim(),
        auth_method: interactiveForm.authMethod,
        login_url: interactiveForm.loginUrl.trim(),
        browser_type: interactiveForm.browserType,
        headless: interactiveForm.headless,
        credentials: {
          username: interactiveForm.username,
          password: interactiveForm.password,
        },
      });
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to start interactive task'));
    }
  }, [interactiveForm, startInteractive]);

  const handleRemoveCredentials = useCallback(async (domain) => {
    try {
      await removeCredentials(domain);
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to remove credentials'));
    }
  }, [removeCredentials]);

  const handleCleanupSessions = useCallback(async () => {
    try {
      await cleanupSessions();
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to cleanup sessions'));
    }
  }, [cleanupSessions]);

  return (
    <div className="space-y-6">
      <ErrorAlert message={error} onClose={() => setError(null)} />

      <AuthHeader
        onRefresh={() => {
          refreshDomains();
          refreshQueue();
          refreshSessions();
        }}
      />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <CredentialForm
          form={credentialForm}
          onChange={updateCredentialForm}
          onSubmit={handleCredentialSubmit}
          isSubmitting={storeCredentialsMutation.isLoading}
        />

        <QueueStatusCard
          queueData={queueQuery.data}
          isLoading={queueQuery.isLoading}
          onRefresh={refreshQueue}
          interactiveForm={interactiveForm}
          onInteractiveChange={updateInteractiveForm}
          onInteractiveSubmit={handleInteractiveSubmit}
          isSubmitting={interactiveMutation.isLoading}
          activeTaskId={activeTaskId}
          task={taskQuery.data}
        />
      </div>

      <DomainsTable
        domains={domains}
        isLoading={domainsQuery.isLoading}
        onRemoveCredentials={handleRemoveCredentials}
        isRemoving={deleteCredentialsMutation.isLoading}
      />

      <SessionsList
        sessions={sessions}
        isLoading={sessionsQuery.isLoading}
        onCleanup={handleCleanupSessions}
        isCleaning={cleanupSessionsMutation.isLoading}
      />
    </div>
  );
};

export default AuthManager;
