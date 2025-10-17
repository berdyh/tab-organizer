import SessionCard from './SessionCard';

const SessionGrid = ({ sessions, selectedSessionIds, onToggleSelection, onSplit, onDelete }) => (
  <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
    {sessions.map((session) => (
      <SessionCard
        key={session.id}
        session={session}
        isSelected={selectedSessionIds.includes(session.id)}
        onSelect={onToggleSelection}
        onSplit={onSplit}
        onDelete={onDelete}
      />
    ))}
    {sessions.length === 0 && (
      <div className="col-span-full bg-white shadow rounded-lg p-8 text-center text-gray-500">
        No sessions yet. Create your first analysis session to get started.
      </div>
    )}
  </div>
);

export default SessionGrid;
