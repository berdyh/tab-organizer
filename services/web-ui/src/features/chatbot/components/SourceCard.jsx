import { ExternalLink } from 'lucide-react';

const SourceCard = ({ source }) => {
  if (source.url) {
    return (
      <div className="bg-white border border-gray-200 rounded p-3 text-sm">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h4 className="font-medium text-gray-900 mb-1">{source.title}</h4>
            {source.snippet && <p className="text-gray-600 text-xs mb-2">{source.snippet}</p>}
            {source.cluster && (
              <span className="inline-block bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded">
                {source.cluster}
              </span>
            )}
          </div>
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-2 text-indigo-600 hover:text-indigo-800"
          >
            <ExternalLink className="h-4 w-4" />
          </a>
        </div>
      </div>
    );
  }

  if (source.count !== undefined) {
    return (
      <div className="bg-white border border-gray-200 rounded p-3 text-sm">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-medium text-gray-900">{source.title}</h4>
            <p className="text-gray-600 text-xs">{source.description}</p>
          </div>
          <span className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded font-medium">
            {source.count} articles
          </span>
        </div>
      </div>
    );
  }

  if (source.metadata) {
    return (
      <div className="bg-white border border-gray-200 rounded p-3 text-sm">
        <h4 className="font-medium text-gray-900 mb-2">{source.title}</h4>
        <p className="text-gray-600 text-xs mb-2">{source.description}</p>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(source.metadata).map(([key, value]) => (
            <div key={key} className="text-xs">
              <span className="text-gray-500">{key}:</span>
              <span className="ml-1 font-medium text-gray-900">{value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return null;
};

export default SourceCard;
