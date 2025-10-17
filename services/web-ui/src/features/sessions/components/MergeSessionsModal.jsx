import { useState } from 'react';

const MergeSessionsModal = ({ sessionIds, onClose, onMerge, isSubmitting }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [archiveSources, setArchiveSources] = useState(true);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (sessionIds.length < 2) {
      return;
    }

    await onMerge({
      sourceSessionIds: sessionIds,
      payload: {
        target_name: name || undefined,
        target_description: description || undefined,
        archive_sources: archiveSources,
      },
    });
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Merge Sessions</h3>
          <p className="text-sm text-gray-600 mb-4">
            {sessionIds.length} sessions will be merged into a single workspace. A new session will be created and the source sessions can be archived automatically.
          </p>
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                New Session Name (optional)
              </label>
              <input
                type="text"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Merged Session"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (optional)
              </label>
              <textarea
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
            <div className="mb-4 flex items-center">
              <input
                id="archive-sources"
                type="checkbox"
                checked={archiveSources}
                onChange={(event) => setArchiveSources(event.target.checked)}
                className="h-4 w-4 text-indigo-600 border-gray-300 rounded mr-2"
              />
              <label htmlFor="archive-sources" className="text-sm text-gray-700">
                Archive source sessions after merge
              </label>
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
                disabled={isSubmitting}
                className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-md disabled:opacity-50"
              >
                {isSubmitting ? 'Merging...' : 'Merge Sessions'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default MergeSessionsModal;
