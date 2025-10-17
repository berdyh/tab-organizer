import { useMemo, useState } from 'react';
import { useMutation } from 'react-query';
import { XCircle } from 'lucide-react';
import { urlAPI } from '../../../lib/api';
import { getErrorMessage } from '../../../shared/utils/errors';

const allowedExtensions = ['.txt', '.json', '.csv', '.tsv', '.xlsx', '.xls'];

const UploadModal = ({ sessionId, onClose, onSuccess }) => {
  const [file, setFile] = useState(null);
  const [formatHint, setFormatHint] = useState(null);
  const [error, setError] = useState(null);

  const uploadMutation = useMutation(
    ({ selectedFile, detectedFormat }) =>
      urlAPI.uploadAuto(selectedFile, {
        format: detectedFormat,
        sessionId,
      }),
    {
      onSuccess: (data) => {
        setError(null);
        onSuccess?.(data);
      },
      onError: (err) => {
        setError(getErrorMessage(err, 'Failed to upload file'));
      },
    },
  );

  const detectFormatFromName = useMemo(
    () => (name) => {
      if (!name) return null;
      const lower = name.toLowerCase();
      if (lower.endsWith('.json')) return 'json';
      if (lower.endsWith('.csv')) return 'csv';
      if (lower.endsWith('.tsv')) return 'tsv';
      if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) return 'excel';
      if (lower.endsWith('.txt')) return 'text';
      return null;
    },
    [],
  );

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0] ?? null;
    setFile(selectedFile);
    setFormatHint(selectedFile ? detectFormatFromName(selectedFile.name) : null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!sessionId) {
      setError('Please select a session before uploading.');
      return;
    }

    if (!file) {
      setError('Please select a file to upload.');
      return;
    }

    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
      setError('Unsupported file type. Please select a TXT, JSON, CSV, TSV, or Excel file.');
      return;
    }

    setError(null);
    await uploadMutation.mutateAsync({ selectedFile: file, detectedFormat: formatHint });
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Upload URL File</h3>

          {error && (
            <div className="mb-4 bg-red-50 border border-red-200 rounded-md p-3">
              <div className="flex">
                <div className="flex-shrink-0">
                  <XCircle className="h-5 w-5 text-red-400" />
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                File
              </label>
              <input
                type="file"
                onChange={handleFileChange}
                accept={allowedExtensions.join(',')}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                required
              />
              {file && (
                <p className="mt-2 text-xs text-gray-500">
                  Detected format: {formatHint ? formatHint.toUpperCase() : 'Auto'}
                </p>
              )}
            </div>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={uploadMutation.isLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-md disabled:opacity-50"
              >
                {uploadMutation.isLoading ? 'Uploading...' : 'Upload'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default UploadModal;
