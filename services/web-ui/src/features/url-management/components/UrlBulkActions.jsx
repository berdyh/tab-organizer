import { Globe, Trash2 } from 'lucide-react';

const UrlBulkActions = ({
  selectedCount,
  onScrape,
  onDelete,
  isScraping,
  isDeleting,
  disabled,
}) => {
  if (selectedCount === 0) {
    return null;
  }

  return (
    <div className="mt-4 flex items-center justify-between bg-indigo-50 rounded-lg p-4">
      <span className="text-sm text-indigo-700">
        {selectedCount} {selectedCount === 1 ? 'URL' : 'URLs'} selected
      </span>
      <div className="flex space-x-2">
        <button
          onClick={onScrape}
          disabled={disabled || isScraping}
          className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-indigo-700 bg-indigo-100 hover:bg-indigo-200 disabled:opacity-50"
        >
          <Globe className="h-4 w-4 mr-1" />
          {isScraping ? 'Starting...' : 'Scrape Selected'}
        </button>
        <button
          onClick={onDelete}
          disabled={disabled || isDeleting}
          className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-red-700 bg-red-100 hover:bg-red-200 disabled:opacity-50"
        >
          <Trash2 className="h-4 w-4 mr-1" />
          {isDeleting ? 'Deleting...' : 'Delete Selected'}
        </button>
      </div>
    </div>
  );
};

export default UrlBulkActions;
