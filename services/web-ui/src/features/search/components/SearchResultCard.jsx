import { useState } from 'react';
import {
  Calendar,
  ExternalLink,
  Globe,
  Tag,
} from 'lucide-react';

const SearchResultCard = ({ result }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white shadow rounded-lg p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <Globe className="h-4 w-4 text-gray-400" />
            <span className="text-sm text-gray-500">{result.domain}</span>
            {result.cluster_label && (
              <ClusterBadge label={result.cluster_label} />
            )}
          </div>

          <h3 className="text-lg font-medium text-gray-900 mb-2">
            <a
              href={result.url}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-indigo-600 flex items-center"
            >
              {result.title || result.url}
              <ExternalLink className="h-4 w-4 ml-1" />
            </a>
          </h3>

          <p className="text-gray-600 mb-3">
            {expanded ? result.content : `${result.snippet}...`}
          </p>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              <div className="flex items-center">
                <Calendar className="h-4 w-4 mr-1" />
                {result.scraped_at ? new Date(result.scraped_at).toLocaleDateString() : '—'}
              </div>
              <div>Similarity: {(result.similarity_score * 100).toFixed(1)}%</div>
            </div>

            <button
              onClick={() => setExpanded((prev) => !prev)}
              className="text-sm text-indigo-600 hover:text-indigo-800"
            >
              {expanded ? 'Show less' : 'Show more'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const ClusterBadge = ({ label }) => (
  <span className="inline-flex items-center gap-1 text-sm text-indigo-600 bg-indigo-100 px-2 py-1 rounded">
    <Tag className="h-4 w-4" />
    {label}
  </span>
);

export default SearchResultCard;
