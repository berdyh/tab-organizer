const SessionPicker = ({ sessions, selectedSession, onSelect }) => (
  <div className="bg-white shadow rounded-lg p-4">
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label htmlFor="session-select" className="block text-sm font-medium text-gray-700 mb-1">
          Session
        </label>
        <select
          id="session-select"
          value={selectedSession}
          onChange={(event) => onSelect(event.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
        >
          <option value="">Select a session</option>
          {sessions.map((session) => (
            <option key={session.id} value={session.id}>
              {session.name || session.id}
            </option>
          ))}
        </select>
      </div>
      <div className="flex items-end">
        <p className="text-sm text-gray-500">
          Choose which session to search and analyse. Create sessions from the Sessions page.
        </p>
      </div>
    </div>
  </div>
);

export default SessionPicker;
