import { ShieldCheck } from 'lucide-react';

const SessionsList = ({ sessions, isLoading, onCleanup, isCleaning }) => (
  <section className="bg-white rounded-lg shadow">
    <header className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
      <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
        <ShieldCheck className="h-5 w-5 text-indigo-500" /> Active Auth Sessions
      </h2>
      <button
        onClick={onCleanup}
        disabled={isCleaning}
        className="inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium text-indigo-600 hover:text-indigo-800 disabled:opacity-50"
      >
        {isCleaning ? 'Cleaning…' : 'Cleanup Expired'}
      </button>
    </header>
    <div className="divide-y divide-gray-100">
      {sessions.map((session) => (
        <div key={session.session_id} className="px-6 py-4 flex justify-between items-center">
          <div>
            <div className="text-sm font-medium text-gray-900">{session.domain}</div>
            <div className="text-xs text-gray-500">Session: {session.session_id}</div>
            <div className="text-xs text-gray-500">
              Last used: {session.last_used ? new Date(session.last_used).toLocaleString() : '—'}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-50 text-indigo-600">
              {session.auth_method}
            </span>
          </div>
        </div>
      ))}
      {sessions.length === 0 && !isLoading && (
        <div className="px-6 py-8 text-center text-sm text-gray-500">
          No active sessions yet. Launch an interactive authentication flow to create a reusable session.
        </div>
      )}
      {isLoading && (
        <div className="px-6 py-8 text-center text-sm text-gray-500">Loading sessions…</div>
      )}
    </div>
  </section>
);

export default SessionsList;
