import React, { useState } from 'react';
import { useMutation, useQuery } from 'react-query';
import { exportAPI, sessionAPI } from '../lib/api';
import {
  ExportActions,
  ExportHeader,
  ExportHistory,
  ExportOptionsPanel,
  FormatSelector,
  PreviewModal,
  SessionSelector,
} from '../features/export';

const defaultOptions = {
  includeMetadata: true,
  includeClusters: true,
  includeEmbeddings: false,
  template: 'default',
  notionToken: '',
  notionDatabaseId: '',
};

const ExportPage = () => {
  const [selectedSession, setSelectedSession] = useState('');
  const [selectedFormat, setSelectedFormat] = useState('markdown');
  const [exportOptions, setExportOptions] = useState(defaultOptions);
  const [showPreview, setShowPreview] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const sessionsQuery = useQuery('sessions', sessionAPI.list);

  const templatesQuery = useQuery(
    ['templates', selectedFormat],
    () => exportAPI.getTemplates(selectedFormat),
    { enabled: Boolean(selectedFormat) },
  );

  const exportMutation = useMutation(exportAPI.export, {
    onSuccess: (data) => {
      const jobId = data?.data?.job_id || data?.job_id || '';
      setFeedback({
        type: 'success',
        message: jobId ? `Export job ${jobId} started successfully.` : 'Export started successfully.',
      });
    },
    onError: (error) => {
      const message =
        error?.response?.data?.detail || error?.message || 'Failed to start export.';
      setFeedback({ type: 'error', message });
    },
  });

  const handleOptionChange = (key, value) => {
    setExportOptions((prev) => ({ ...prev, [key]: value }));
  };

  const handleExport = async () => {
    if (!selectedSession) {
      setFeedback({ type: 'error', message: 'Select a session before exporting.' });
      return;
    }

    if (!selectedFormat) {
      setFeedback({ type: 'error', message: 'Choose an export format.' });
      return;
    }

    if (selectedFormat === 'notion') {
      if (!exportOptions.notionToken || !exportOptions.notionDatabaseId) {
        setFeedback({
          type: 'error',
          message: 'Notion exports require an integration token and a database ID.',
        });
        return;
      }
    }

    const payload = {
      session_id: selectedSession,
      format: selectedFormat,
      include_metadata: exportOptions.includeMetadata,
      include_clusters: exportOptions.includeClusters,
      include_embeddings: exportOptions.includeEmbeddings,
    };

    if (exportOptions.template && exportOptions.template !== 'default') {
      payload.template_name = exportOptions.template;
    }

    if (selectedFormat === 'notion') {
      payload.notion_token = exportOptions.notionToken;
      payload.notion_database_id = exportOptions.notionDatabaseId;
    }

    setFeedback(null);
    await exportMutation.mutateAsync(payload);
  };

  const availableTemplates = [
    { id: 'default', name: 'Default' },
    ...(templatesQuery.data?.data ?? []),
  ];

  return (
    <div className="space-y-6">
      <ExportHeader feedback={feedback} onDismiss={() => setFeedback(null)} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <SessionSelector
            sessions={sessionsQuery.data?.data ?? []}
            selectedSession={selectedSession}
            onSelect={setSelectedSession}
          />

          <FormatSelector selectedFormat={selectedFormat} onSelect={setSelectedFormat} />

          <ExportOptionsPanel
            options={exportOptions}
            onOptionChange={handleOptionChange}
            templates={availableTemplates}
            selectedFormat={selectedFormat}
          />

          <ExportActions
            selectedSession={selectedSession}
            selectedFormat={selectedFormat}
            onPreview={() => setShowPreview(true)}
            onExport={handleExport}
            isExporting={exportMutation.isLoading}
          />
        </div>

        <div className="space-y-6">
          <ExportHistory />
        </div>
      </div>

      {showPreview && (
        <PreviewModal
          sessionId={selectedSession}
          format={selectedFormat}
          onClose={() => setShowPreview(false)}
        />
      )}
    </div>
  );
};

export default ExportPage;
