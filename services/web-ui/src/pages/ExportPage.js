import React, { useState } from 'react';
import { useQuery, useMutation } from 'react-query';
import { 
  Download, 
  FileText, 
  Database, 
  BookOpen,
  CheckCircle,
  Clock,
  AlertCircle,
  Eye
} from 'lucide-react';
import { exportAPI, sessionAPI } from '../services/api';

const ExportPage = () => {
  const [selectedSession, setSelectedSession] = useState('');
  const [selectedFormat, setSelectedFormat] = useState('markdown');
  const [exportOptions, setExportOptions] = useState({
    includeMetadata: true,
    includeClusters: true,
    includeContent: true,
    template: 'default'
  });
  const [showPreview, setShowPreview] = useState(false);

  const { data: sessions } = useQuery('sessions', sessionAPI.list);
  const { data: templates } = useQuery(
    ['templates', selectedFormat],
    () => exportAPI.getTemplates(selectedFormat),
    { enabled: !!selectedFormat }
  );

  const exportMutation = useMutation(exportAPI.export, {
    onSuccess: (data) => {
      // Handle successful export
      console.log('Export started:', data);
    },
  });

  const handleExport = async () => {
    if (!selectedSession || !selectedFormat) return;

    await exportMutation.mutateAsync(selectedSession, selectedFormat, exportOptions);
  };

  const handleOptionChange = (key, value) => {
    setExportOptions(prev => ({ ...prev, [key]: value }));
  };

  const formatIcons = {
    markdown: FileText,
    notion: Database,
    obsidian: BookOpen,
    word: FileText,
    json: Database,
    csv: Database
  };

  const formatDescriptions = {
    markdown: 'Standard Markdown files with frontmatter metadata',
    notion: 'Structured database pages with proper formatting',
    obsidian: 'Markdown files optimized for Obsidian with internal linking',
    word: 'Microsoft Word documents with tables and formatting',
    json: 'Structured JSON data for programmatic access',
    csv: 'Comma-separated values for spreadsheet applications'
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Export Data</h1>
        <p className="mt-1 text-sm text-gray-500">
          Export your analysis results to various formats and platforms
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Export Configuration */}
        <div className="lg:col-span-2 space-y-6">
          {/* Session Selection */}
          <div className="bg-white shadow rounded-lg p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Select Session</h3>
            <select
              value={selectedSession}
              onChange={(e) => setSelectedSession(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="">Choose a session to export...</option>
              {sessions?.data?.map(session => (
                <option key={session.id} value={session.id}>
                  {session.name} ({session.url_count || 0} URLs, {session.cluster_count || 0} clusters)
                </option>
              ))}
            </select>
          </div>

          {/* Format Selection */}
          <div className="bg-white shadow rounded-lg p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Export Format</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {Object.entries(formatIcons).map(([format, Icon]) => (
                <div
                  key={format}
                  className={`relative rounded-lg border p-4 cursor-pointer hover:bg-gray-50 ${
                    selectedFormat === format
                      ? 'border-indigo-500 ring-2 ring-indigo-500'
                      : 'border-gray-300'
                  }`}
                  onClick={() => setSelectedFormat(format)}
                >
                  <div className="flex items-center">
                    <Icon className="h-6 w-6 text-gray-400 mr-3" />
                    <div>
                      <div className="text-sm font-medium text-gray-900 capitalize">
                        {format}
                      </div>
                      <div className="text-xs text-gray-500">
                        {formatDescriptions[format]}
                      </div>
                    </div>
                  </div>
                  {selectedFormat === format && (
                    <CheckCircle className="absolute top-2 right-2 h-5 w-5 text-indigo-500" />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Export Options */}
          <div className="bg-white shadow rounded-lg p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Export Options</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-900">
                    Include Metadata
                  </label>
                  <p className="text-xs text-gray-500">
                    URL metadata, timestamps, and processing information
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={exportOptions.includeMetadata}
                  onChange={(e) => handleOptionChange('includeMetadata', e.target.checked)}
                  className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-900">
                    Include Clusters
                  </label>
                  <p className="text-xs text-gray-500">
                    Cluster assignments and labels
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={exportOptions.includeClusters}
                  onChange={(e) => handleOptionChange('includeClusters', e.target.checked)}
                  className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-900">
                    Include Full Content
                  </label>
                  <p className="text-xs text-gray-500">
                    Complete scraped content (may result in large files)
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={exportOptions.includeContent}
                  onChange={(e) => handleOptionChange('includeContent', e.target.checked)}
                  className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                />
              </div>

              {templates?.data && (
                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-2">
                    Template
                  </label>
                  <select
                    value={exportOptions.template}
                    onChange={(e) => handleOptionChange('template', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    {templates.data.map(template => (
                      <option key={template.id} value={template.id}>
                        {template.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          </div>

          {/* Export Actions */}
          <div className="bg-white shadow rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-gray-900">Ready to Export</h3>
                <p className="text-sm text-gray-500">
                  {selectedSession && selectedFormat
                    ? `Export session to ${selectedFormat.toUpperCase()} format`
                    : 'Select a session and format to continue'
                  }
                </p>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={() => setShowPreview(true)}
                  disabled={!selectedSession || !selectedFormat}
                  className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
                >
                  <Eye className="h-4 w-4 mr-2" />
                  Preview
                </button>
                <button
                  onClick={handleExport}
                  disabled={!selectedSession || !selectedFormat || exportMutation.isLoading}
                  className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50"
                >
                  <Download className="h-4 w-4 mr-2" />
                  {exportMutation.isLoading ? 'Exporting...' : 'Export'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Export History */}
        <div className="space-y-6">
          <ExportHistory />
        </div>
      </div>

      {/* Preview Modal */}
      {showPreview && (
        <PreviewModal
          sessionId={selectedSession}
          format={selectedFormat}
          options={exportOptions}
          onClose={() => setShowPreview(false)}
        />
      )}
    </div>
  );
};

const ExportHistory = () => {
  const { data: exportJobs } = useQuery('export-jobs', () => 
    // Mock data for now
    Promise.resolve({
      data: [
        {
          id: '1',
          session_name: 'AI Research Session',
          format: 'markdown',
          status: 'completed',
          created_at: new Date().toISOString(),
          file_size: '2.4 MB'
        },
        {
          id: '2',
          session_name: 'Web Dev Articles',
          format: 'notion',
          status: 'processing',
          created_at: new Date(Date.now() - 3600000).toISOString(),
          progress: 75
        },
        {
          id: '3',
          session_name: 'Tech News',
          format: 'obsidian',
          status: 'failed',
          created_at: new Date(Date.now() - 7200000).toISOString(),
          error: 'Template not found'
        }
      ]
    })
  );

  const statusIcons = {
    completed: CheckCircle,
    processing: Clock,
    failed: AlertCircle
  };

  const statusColors = {
    completed: 'text-green-500',
    processing: 'text-blue-500',
    failed: 'text-red-500'
  };

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Exports</h3>
      <div className="space-y-4">
        {exportJobs?.data?.map(job => {
          const StatusIcon = statusIcons[job.status];
          const statusColor = statusColors[job.status];

          return (
            <div key={job.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <StatusIcon className={`h-5 w-5 ${statusColor}`} />
                <div>
                  <div className="text-sm font-medium text-gray-900">
                    {job.session_name}
                  </div>
                  <div className="text-xs text-gray-500">
                    {job.format.toUpperCase()} • {new Date(job.created_at).toLocaleString()}
                  </div>
                  {job.status === 'processing' && (
                    <div className="mt-1">
                      <div className="w-32 bg-gray-200 rounded-full h-1">
                        <div 
                          className="bg-blue-500 h-1 rounded-full" 
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    </div>
                  )}
                  {job.status === 'failed' && (
                    <div className="text-xs text-red-600 mt-1">
                      {job.error}
                    </div>
                  )}
                </div>
              </div>
              {job.status === 'completed' && (
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-500">{job.file_size}</span>
                  <button className="text-indigo-600 hover:text-indigo-800 text-sm">
                    Download
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const PreviewModal = ({ sessionId, format, options, onClose }) => {
  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-10 mx-auto p-5 border w-4/5 max-w-4xl shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-medium text-gray-900">Export Preview</h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>

          <div className="bg-gray-50 rounded-lg p-4 h-96 overflow-y-auto">
            <div className="text-sm text-gray-600 font-mono">
              {/* Mock preview content */}
              <div className="mb-4">
                <div className="text-gray-800 font-bold">
                  # Exported Session Data
                </div>
                <div className="mt-2">
                  **Format:** {format.toUpperCase()}<br/>
                  **Session:** {sessionId}<br/>
                  **Generated:** {new Date().toLocaleString()}
                </div>
              </div>
              
              <div className="mb-4">
                <div className="text-gray-800 font-bold">## Clusters</div>
                <div className="mt-2">
                  - **Cluster 1:** AI and Machine Learning (15 articles)<br/>
                  - **Cluster 2:** Web Development (23 articles)<br/>
                  - **Cluster 3:** Data Science (8 articles)
                </div>
              </div>

              <div className="mb-4">
                <div className="text-gray-800 font-bold">## Sample Content</div>
                <div className="mt-2 bg-white p-2 rounded border">
                  **Title:** Introduction to React Hooks<br/>
                  **URL:** https://example.com/react-hooks<br/>
                  **Cluster:** Web Development<br/>
                  **Content:** React Hooks provide a way to use state and other React features...
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 flex justify-end">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md"
            >
              Close Preview
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExportPage;