import { useState } from 'react';

const emptyPart = { name: '', pointIds: '', description: '', tags: '' };

const SplitSessionModal = ({ session, onClose, onSplit, isSubmitting }) => {
  const [parts, setParts] = useState([{ ...emptyPart }]);
  const [removePoints, setRemovePoints] = useState(true);
  const [archiveOriginal, setArchiveOriginal] = useState(false);

  const updatePart = (index, key, value) => {
    setParts((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], [key]: value };
      return next;
    });
  };

  const addPart = () => {
    setParts((prev) => [...prev, { ...emptyPart }]);
  };

  const resetForm = () => {
    setParts([{ ...emptyPart }]);
    setRemovePoints(true);
    setArchiveOriginal(false);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formattedParts = parts
      .map((part) => ({
        name: part.name.trim(),
        point_ids: part.pointIds
          .split(/[\s,]+/)
          .map((value) => value.trim())
          .filter(Boolean),
        description: part.description.trim() || undefined,
        tags: part.tags
          .split(',')
          .map((value) => value.trim())
          .filter(Boolean),
      }))
      .filter((part) => part.name && part.point_ids.length);

    if (!formattedParts.length) {
      return;
    }

    await onSplit({
      sessionId: session.id,
      payload: {
        parts: formattedParts,
        remove_points: removePoints,
        archive_original: archiveOriginal,
      },
    });

    resetForm();
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-10 mx-auto p-5 border w-full max-w-3xl shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-medium text-gray-900">Split Session: {session.name}</h3>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600">×</button>
          </div>
          <p className="text-sm text-gray-600 mb-4">
            Provide one or more new session definitions. Point IDs should match the underlying Qdrant point identifiers associated with this session.
          </p>
          <form onSubmit={handleSubmit} className="space-y-6">
            {parts.map((part, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      New Session Name
                    </label>
                    <input
                      type="text"
                      value={part.name}
                      onChange={(event) => updatePart(index, 'name', event.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Tags (comma separated)
                    </label>
                    <input
                      type="text"
                      value={part.tags}
                      onChange={(event) => updatePart(index, 'tags', event.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                      placeholder="research, sprint"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Point IDs (comma or whitespace separated)
                  </label>
                  <textarea
                    value={part.pointIds}
                    onChange={(event) => updatePart(index, 'pointIds', event.target.value)}
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Description (optional)
                  </label>
                  <textarea
                    value={part.description}
                    onChange={(event) => updatePart(index, 'description', event.target.value)}
                    rows={2}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
              </div>
            ))}

            <button
              type="button"
              onClick={addPart}
              className="inline-flex items-center px-3 py-1 border border-dashed border-gray-300 rounded-md text-sm text-gray-600 hover:bg-gray-50"
            >
              + Add Another Split Definition
            </button>

            <div className="flex items-center space-x-3">
              <label className="flex items-center text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={removePoints}
                  onChange={(event) => setRemovePoints(event.target.checked)}
                  className="h-4 w-4 text-indigo-600 border-gray-300 rounded mr-2"
                />
                Remove selected points from the original session
              </label>
              <label className="flex items-center text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={archiveOriginal}
                  onChange={(event) => setArchiveOriginal(event.target.checked)}
                  className="h-4 w-4 text-indigo-600 border-gray-300 rounded mr-2"
                />
                Archive original session after split
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
                {isSubmitting ? 'Splitting...' : 'Split Session'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default SplitSessionModal;
