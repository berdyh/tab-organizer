import { Plus, Upload } from 'lucide-react';

const UrlManagerHeader = ({
  sessions,
  selectedSession,
  onSessionChange,
  onAddClick,
  onUploadClick,
  actionsDisabled,
}) => (
  <div className="flex justify-between items-center">
    <div>
      <h1 className="text-2xl font-bold text-gray-900">URL Manager</h1>
      <p className="mt-1 text-sm text-gray-500">
        Manage your URLs and organize them into collections
      </p>
      {sessions.length > 0 && (
        <div className="mt-3">
          <label htmlFor="session-select" className="block text-sm font-medium text-gray-700 mb-1">
            Active Session
          </label>
          <select
            id="session-select"
            value={selectedSession}
            onChange={(event) => onSessionChange(event.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
          >
            {sessions.map((session) => (
              <option key={session.id} value={session.id}>
                {session.name || session.id}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
    <div className="flex space-x-3">
      <button
        onClick={onUploadClick}
        disabled={actionsDisabled}
        className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
      >
        <Upload className="h-4 w-4 mr-2" />
        Upload File
      </button>
      <button
        onClick={onAddClick}
        disabled={actionsDisabled}
        className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50"
      >
        <Plus className="h-4 w-4 mr-2" />
        Add URL
      </button>
    </div>
  </div>
);

export default UrlManagerHeader;
