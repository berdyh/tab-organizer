import { Search as SearchIcon } from 'lucide-react';
import SearchResultCard from './SearchResultCard';

const SearchResultsList = ({ results }) => {
  if (!results) {
    return null;
  }

  if (results.length === 0) {
    return (
      <div className="bg-white shadow rounded-lg p-8 text-center">
        <SearchIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
        <p className="text-gray-500">
          Try adjusting your search terms or filters to find what you're looking for.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {results.map((result, index) => (
        <SearchResultCard key={`${result.id || index}`} result={result} />
      ))}
    </div>
  );
};

export default SearchResultsList;
