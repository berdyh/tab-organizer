import { Download, Eye } from 'lucide-react';

const ExportActions = ({
  selectedSession,
  selectedFormat,
  onPreview,
  onExport,
  isExporting,
}) => (
  <div className="bg-white shadow rounded-lg p-6">
    <div className="flex items-center justify-between">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Ready to Export</h3>
        <p className="text-sm text-gray-500">
          {selectedSession && selectedFormat
            ? `Export session to ${selectedFormat.toUpperCase()} format`
            : 'Select a session and format to continue'}
        </p>
      </div>
      <div className="flex space-x-3">
        <button
          type="button"
          onClick={onPreview}
          disabled={!selectedSession || !selectedFormat}
          className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
        >
          <Eye className="h-4 w-4 mr-2" />
          Preview
        </button>
        <button
          type="button"
          onClick={onExport}
          disabled={!selectedSession || !selectedFormat || isExporting}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50"
        >
          <Download className="h-4 w-4 mr-2" />
          {isExporting ? 'Exporting...' : 'Export'}
        </button>
      </div>
    </div>
  </div>
);

export default ExportActions;
