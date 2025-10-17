import { useState } from 'react';
import { useMutation } from 'react-query';
import { XCircle } from 'lucide-react';
import { urlAPI } from '../../../lib/api';
import { getErrorMessage } from '../../../shared/utils/errors';

const AddUrlModal = ({ sessionId, onClose, onSuccess }) => {
  const [url, setUrl] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState(null);

  const validateMutation = useMutation(
    (value) => urlAPI.validate(value, sessionId),
    {
      onSuccess: () => {
        setError(null);
        onSuccess?.();
      },
      onError: (err) => {
        setError(getErrorMessage(err, 'Failed to validate URL'));
      },
      onSettled: () => {
        setIsValidating(false);
      },
    },
  );

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!sessionId) {
      setError('Please select a session before adding a URL.');
      return;
    }

    if (!url) {
      setError('URL is required');
      return;
    }

    try {
      new URL(url);
    } catch (e) {
      setError('Please provide a valid URL');
      return;
    }

    setIsValidating(true);
    await validateMutation.mutateAsync(url);
    setUrl('');
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Add URL</h3>

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
                URL
              </label>
              <input
                type="url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://example.com"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                required
              />
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
                disabled={isValidating || validateMutation.isLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-md disabled:opacity-50"
              >
                {isValidating ? 'Validating...' : 'Add URL'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default AddUrlModal;
