import { Plus } from 'lucide-react';

const SessionsHeader = ({ onCreateClick }) => (
  <div className="flex justify-between items-center">
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Session Manager</h1>
      <p className="mt-1 text-sm text-gray-500">
        Create, manage, and compare analysis sessions
      </p>
    </div>
    <button
      onClick={onCreateClick}
      className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
    >
      <Plus className="h-4 w-4 mr-2" />
      New Session
    </button>
  </div>
);

export default SessionsHeader;
