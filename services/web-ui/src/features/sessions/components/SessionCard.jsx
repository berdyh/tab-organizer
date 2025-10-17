import {
  BarChart3,
  Calendar,
  GitCompare,
  Trash2,
  Users,
} from 'lucide-react';

const SessionCard = ({ session, isSelected, onSelect, onSplit, onDelete }) => (
  <div
    className={`bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow ${
      isSelected ? 'ring-2 ring-indigo-500' : ''
    }`}
  >
    <div className="p-6">
      <div className="flex items-start justify-between mb-4">
        <label className="flex items-center space-x-3 cursor-pointer">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={() => onSelect(session.id)}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <span className="text-lg font-medium text-gray-900 truncate max-w-xs">
            {session.name}
          </span>
        </label>
        <span className="text-xs font-medium text-indigo-600 bg-indigo-50 px-2.5 py-0.5 rounded-full">
          {session.type || 'Custom'}
        </span>
      </div>

      <p className="text-sm text-gray-600 mb-4 line-clamp-2">
        {session.description || 'No description provided'}
      </p>

      <dl className="space-y-3">
        <DataRow icon={Users} label="URLs Processed" value={session.url_count || 0} />
        <DataRow icon={BarChart3} label="Clusters Found" value={session.cluster_count || 0} />
        <DataRow
          icon={Calendar}
          label="Last Updated"
          value={session.updated_at ? new Date(session.updated_at).toLocaleDateString() : '—'}
        />
      </dl>

      <div className="mt-4 pt-4 border-t border-gray-200 flex items-center justify-between">
        <div className="flex items-center text-sm text-gray-500">
          <Calendar className="h-4 w-4 mr-1" />
          {session.created_at ? new Date(session.created_at).toLocaleDateString() : '—'}
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => onSplit(session)}
            className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-indigo-700 bg-indigo-100 hover:bg-indigo-200"
          >
            <GitCompare className="h-4 w-4 mr-1" />
            Split
          </button>
          <button
            onClick={() => onDelete(session)}
            className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-red-700 bg-red-100 hover:bg-red-200"
          >
            <Trash2 className="h-4 w-4 mr-1" />
            Delete
          </button>
        </div>
      </div>
    </div>
  </div>
);

const DataRow = ({ icon: Icon, label, value }) => (
  <div className="flex items-center justify-between text-sm">
    <div className="flex items-center text-gray-500">
      <Icon className="h-4 w-4 mr-2" />
      {label}
    </div>
    <span className="font-medium text-gray-900">{value}</span>
  </div>
);

export default SessionCard;
