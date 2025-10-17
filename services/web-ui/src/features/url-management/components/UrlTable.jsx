import {
  AlertCircle,
  CheckCircle,
  Clock,
  Edit,
  Trash2,
  XCircle,
} from 'lucide-react';

const statusConfig = {
  pending: { icon: Clock, color: 'text-yellow-500', bg: 'bg-yellow-100' },
  scraping: { icon: AlertCircle, color: 'text-blue-500', bg: 'bg-blue-100' },
  completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100' },
  failed: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-100' },
  auth_required: { icon: AlertCircle, color: 'text-orange-500', bg: 'bg-orange-100' },
};

const UrlTable = ({ urls, selectedIds, onToggleSelectAll, onToggleSelect }) => (
  <div className="bg-white shadow overflow-hidden sm:rounded-md">
    <div className="px-4 py-5 sm:p-6">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left">
                <input
                  type="checkbox"
                  checked={selectedIds.length > 0 && selectedIds.length === urls.length}
                  onChange={() => onToggleSelectAll()}
                  className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                />
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                URL
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Title
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Added
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {urls.map((url) => (
              <UrlRow
                key={url.id}
                url={url}
                isSelected={selectedIds.includes(url.id)}
                onSelect={() => onToggleSelect(url.id)}
              />
            ))}
            {urls.length === 0 && (
              <tr>
                <td colSpan="6" className="px-6 py-4 text-center text-gray-500">
                  No URLs found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  </div>
);

const UrlRow = ({ url, isSelected, onSelect }) => {
  const config = statusConfig[url.status] || statusConfig.pending;
  const StatusIcon = config.icon;

  return (
    <tr className={isSelected ? 'bg-indigo-50' : 'hover:bg-gray-50'}>
      <td className="px-6 py-4 whitespace-nowrap">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onSelect}
          className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
        />
      </td>
      <td className="px-6 py-4">
        <div className="text-sm text-gray-900 truncate max-w-xs" title={url.url}>
          {url.url}
        </div>
        <div className="text-sm text-gray-500">{url.domain}</div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.bg} ${config.color}`}>
          <StatusIcon className="h-3 w-3 mr-1" />
          {url.status.replace('_', ' ')}
        </span>
      </td>
      <td className="px-6 py-4">
        <div className="text-sm text-gray-900 truncate max-w-xs">
          {url.title || 'No title'}
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {url.created_at ? new Date(url.created_at).toLocaleDateString() : '—'}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
        <button className="text-indigo-600 hover:text-indigo-900 mr-3" type="button" aria-label="Edit URL">
          <Edit className="h-4 w-4" />
        </button>
        <button className="text-red-600 hover:text-red-900" type="button" aria-label="Delete URL">
          <Trash2 className="h-4 w-4" />
        </button>
      </td>
    </tr>
  );
};

export default UrlTable;
