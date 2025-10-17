import React, { useCallback, useState } from 'react';
import { Settings } from 'lucide-react';
import { useMutation } from 'react-query';
import { sessionAPI } from '../lib/api';
import ErrorAlert from '../shared/components/ErrorAlert';
import { getErrorMessage } from '../shared/utils/errors';
import {
  CompareSessionsModal,
  CreateSessionModal,
  MergeSessionsModal,
  SelectedSessionsBar,
  SessionsHeader,
  SessionGrid,
  SplitSessionModal,
  useSessionsManager,
} from '../features/sessions';

const SessionManager = () => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [showMergeModal, setShowMergeModal] = useState(false);
  const [showSplitModal, setShowSplitModal] = useState(false);
  const [splitTargetSession, setSplitTargetSession] = useState(null);
  const [error, setError] = useState(null);

  const {
    sessions,
    isLoading,
    selectedSessions,
    toggleSessionSelection,
    deleteSessions,
    mergeSessions,
    splitSession,
    deleteSessionMutation,
    mergeSessionsMutation,
    splitSessionMutation,
    refreshSessions,
  } = useSessionsManager();

  const createSessionMutation = useMutation(
    ({ name, description }) => sessionAPI.create(name, description),
    {
      onSuccess: () => {
        refreshSessions();
        setShowCreateModal(false);
        setError(null);
      },
      onError: (err) => {
        setError(getErrorMessage(err, 'Failed to create session'));
      },
    },
  );

  const handleDeleteSelected = useCallback(async () => {
    if (!selectedSessions.length) {
      return;
    }
    const confirmed = window.confirm(`Delete ${selectedSessions.length} selected sessions?`);
    if (!confirmed) {
      return;
    }
    try {
      await deleteSessions(selectedSessions);
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to delete sessions'));
    }
  }, [selectedSessions, deleteSessions]);

  const handleDeleteSingle = useCallback(async (session) => {
    const confirmed = window.confirm(`Delete session "${session.name}"?`);
    if (!confirmed) {
      return;
    }
    try {
      await deleteSessions([session.id]);
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to delete session'));
    }
  }, [deleteSessions]);

  const handleMergeSubmit = useCallback(async ({ sourceSessionIds, payload }) => {
    try {
      await mergeSessions({ sourceSessionIds, payload });
      setShowMergeModal(false);
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to merge sessions'));
    }
  }, [mergeSessions]);

  const handleSplitSubmit = useCallback(async ({ sessionId, payload }) => {
    try {
      await splitSession({ sessionId, payload });
      setShowSplitModal(false);
      setSplitTargetSession(null);
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to split session'));
    }
  }, [splitSession]);

  const handleSplitRequest = useCallback((session) => {
    setSplitTargetSession(session);
    setShowSplitModal(true);
  }, []);

  const handleCreateSession = useCallback(async ({ name, description }) => {
    await createSessionMutation.mutateAsync({ name, description });
  }, [createSessionMutation]);

  if (isLoading) {
    return <div className="flex justify-center items-center h-64">Loading...</div>;
  }

  if (!sessions.length) {
    return (
      <div className="space-y-6">
        <ErrorAlert message={error} onClose={() => setError(null)} />
        <div className="bg-white shadow rounded-lg p-8 text-center">
          <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">No sessions yet</h2>
          <p className="text-gray-500 mb-4">
            Create your first analysis session to get started.
          </p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
          >
            Create Session
          </button>
        </div>
        {showCreateModal && (
          <CreateSessionModal
            onClose={() => setShowCreateModal(false)}
            onCreate={handleCreateSession}
            isSubmitting={createSessionMutation.isLoading}
          />
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <ErrorAlert message={error} onClose={() => setError(null)} />

      <SessionsHeader onCreateClick={() => setShowCreateModal(true)} />

      <SelectedSessionsBar
        count={selectedSessions.length}
        onCompare={() => setShowCompareModal(true)}
        onMerge={() => setShowMergeModal(true)}
        onDelete={handleDeleteSelected}
        isDeleting={deleteSessionMutation.isLoading}
      />

      <SessionGrid
        sessions={sessions}
        selectedSessionIds={selectedSessions}
        onToggleSelection={toggleSessionSelection}
        onSplit={handleSplitRequest}
        onDelete={handleDeleteSingle}
      />

      {showCreateModal && (
        <CreateSessionModal
          onClose={() => setShowCreateModal(false)}
          onCreate={handleCreateSession}
          isSubmitting={createSessionMutation.isLoading}
        />
      )}

      {showCompareModal && selectedSessions.length >= 2 && (
        <CompareSessionsModal
          sessionIds={selectedSessions}
          onClose={() => setShowCompareModal(false)}
        />
      )}

      {showMergeModal && selectedSessions.length >= 2 && (
        <MergeSessionsModal
          sessionIds={selectedSessions}
          onClose={() => setShowMergeModal(false)}
          onMerge={handleMergeSubmit}
          isSubmitting={mergeSessionsMutation.isLoading}
        />
      )}

      {showSplitModal && splitTargetSession && (
        <SplitSessionModal
          session={splitTargetSession}
          onClose={() => {
            setShowSplitModal(false);
            setSplitTargetSession(null);
          }}
          onSplit={handleSplitSubmit}
          isSubmitting={splitSessionMutation.isLoading}
        />
      )}
    </div>
  );
};

export default SessionManager;
