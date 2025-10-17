const SessionSelector = ({ sessions, selectedSession, onSelect }) => (
  <div className="bg-white shadow rounded-lg p-6">
    <h3 className="text-lg font-medium text-gray-900 mb-4">Select Session</h3>
    <select
      value={selectedSession}
      onChange={(event) => onSelect(event.target.value)}
      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
    >
      <option value="">Choose a session to export...</option>
      {sessions?.map((session) => (
        <option key={session.id} value={session.id}>
          {session.name} ({session.url_count || 0} URLs, {session.cluster_count || 0} clusters)
        </option>
      ))}
    </select>
  </div>
);

export default SessionSelector;
