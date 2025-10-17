import { Copy, GitCompare, Trash2 } from 'lucide-react';

const SelectedSessionsBar = ({
  count,
  onCompare,
  onMerge,
  onDelete,
  isDeleting,
}) => {
  if (count === 0) {
    return null;
  }

  return (
    <div className="bg-indigo-50 rounded-lg p-4 flex items-center justify-between">
      <span className="text-sm text-indigo-700">
        {count} {count === 1 ? 'session' : 'sessions'} selected
      </span>
      <div className="flex space-x-2">
        {count >= 2 && (
          <button
            onClick={onCompare}
            className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-indigo-700 bg-indigo-100 hover:bg-indigo-200"
          >
            <GitCompare className="h-4 w-4 mr-1" />
            Compare Sessions
          </button>
        )}
        {count >= 2 && (
          <button
            onClick={onMerge}
            className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-purple-700 bg-purple-100 hover:bg-purple-200"
          >
            <Copy className="h-4 w-4 mr-1" />
            Merge Sessions
          </button>
        )}
        <button
          onClick={onDelete}
          disabled={isDeleting}
          className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-red-700 bg-red-100 hover:bg-red-200 disabled:opacity-50"
        >
          <Trash2 className="h-4 w-4 mr-1" />
          {isDeleting ? 'Deleting...' : 'Delete Selected'}
        </button>
      </div>
    </div>
  );
};

export default SelectedSessionsBar;
