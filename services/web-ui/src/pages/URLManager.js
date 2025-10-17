import React, { useCallback, useMemo, useState } from 'react';
import { Settings } from 'lucide-react';
import ErrorAlert from '../shared/components/ErrorAlert';
import {
  AddUrlModal,
  UploadModal,
  UrlBulkActions,
  UrlFilters,
  UrlManagerHeader,
  UrlTable,
  useUrlManager,
} from '../features/url-management';

const URLManager = () => {
  const [showAddModal, setShowAddModal] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);

  const {
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
    isSessionsLoading,
    error,
    clearError,
    deleteUrlMutation,
    scrapeMutation,
    handleDeleteSelected,
    handleScrapeSelected,
    refreshUrls,
  } = useUrlManager();

  const disableActions = useMemo(
    () => !selectedSession || isSessionsLoading,
    [selectedSession, isSessionsLoading],
  );

  const handleDelete = useCallback(() => {
    if (!selectedUrls.length) {
      return;
    }
    const confirmed = window.confirm(`Delete ${selectedUrls.length} selected URLs?`);
    if (confirmed) {
      void handleDeleteSelected();
    }
  }, [selectedUrls.length, handleDeleteSelected]);

  const handleScrape = useCallback(() => {
    handleScrapeSelected();
  }, [handleScrapeSelected]);

  if (isLoading) {
    return <div className="flex justify-center items-center h-64">Loading...</div>;
  }

  if (!isSessionsLoading && sessions.length === 0) {
    return (
      <div className="space-y-6">
        <ErrorAlert message={error} onClose={clearError} />
        <div className="bg-white shadow rounded-lg p-8 text-center">
          <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">No sessions available</h2>
          <p className="text-gray-500">
            Create a session first to start collecting URLs. Visit the Sessions page to add one.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <ErrorAlert message={error} onClose={clearError} />

      <UrlManagerHeader
        sessions={sessions}
        selectedSession={selectedSession}
        onSessionChange={setSelectedSession}
        onAddClick={() => setShowAddModal(true)}
        onUploadClick={() => setShowUploadModal(true)}
        actionsDisabled={disableActions}
      />

      <UrlFilters
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        statusFilter={statusFilter}
        onStatusChange={setStatusFilter}
      >
        <UrlBulkActions
          selectedCount={selectedUrls.length}
          onScrape={handleScrape}
          onDelete={handleDelete}
          isScraping={scrapeMutation.isLoading}
          isDeleting={deleteUrlMutation.isLoading}
          disabled={disableActions}
        />
      </UrlFilters>

      <UrlTable
        urls={filteredUrls}
        selectedIds={selectedUrls}
        onToggleSelectAll={toggleSelectAll}
        onToggleSelect={toggleSelectUrl}
      />

      {showAddModal && (
        <AddUrlModal
          sessionId={selectedSession}
          onClose={() => setShowAddModal(false)}
          onSuccess={() => {
            setShowAddModal(false);
            refreshUrls();
          }}
        />
      )}

      {showUploadModal && (
        <UploadModal
          sessionId={selectedSession}
          onClose={() => setShowUploadModal(false)}
          onSuccess={() => {
            setShowUploadModal(false);
            refreshUrls();
          }}
        />
      )}
    </div>
  );
};

export default URLManager;
