import { ChevronDown, ChevronUp, Filter, Search as SearchIcon } from 'lucide-react';
import SearchFilters from './SearchFilters';

const SearchForm = ({
  query,
  onQueryChange,
  searchType,
  onSearchTypeChange,
  isSearching,
  onSubmit,
  showAdvanced,
  onToggleAdvanced,
  filters,
  onFilterChange,
  clusters,
  resultSummary,
}) => (
  <div className="bg-white shadow rounded-lg p-6">
    <form onSubmit={onSubmit} className="space-y-4">
      <div className="flex gap-4">
        <div className="flex-1">
          <div className="relative">
            <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search for content, topics, or specific information..."
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              className="pl-10 pr-4 py-3 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-lg"
            />
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <select
            value={searchType}
            onChange={(event) => onSearchTypeChange(event.target.value)}
            className="px-4 py-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
            aria-label="Search type"
          >
            <option value="semantic">Semantic Search</option>
            <option value="keyword">Keyword Search</option>
            <option value="hybrid">Hybrid Search</option>
          </select>
          <button
            type="submit"
            disabled={isSearching}
            className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 font-medium"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      <div className="flex justify-between items-center">
        <button
          type="button"
          onClick={onToggleAdvanced}
          className="flex items-center text-sm text-gray-600 hover:text-gray-900"
        >
          <Filter className="h-4 w-4 mr-1" />
          Advanced Filters
          {showAdvanced ? <ChevronUp className="h-4 w-4 ml-1" /> : <ChevronDown className="h-4 w-4 ml-1" />}
        </button>
        {resultSummary && (
          <div className="text-sm text-gray-500">
            Found {resultSummary.total_count} results in {resultSummary.search_time}ms
          </div>
        )}
      </div>

      {showAdvanced && (
        <SearchFilters
          filters={filters}
          onFilterChange={onFilterChange}
          clusters={clusters}
        />
      )}
    </form>
  </div>
);

export default SearchForm;
