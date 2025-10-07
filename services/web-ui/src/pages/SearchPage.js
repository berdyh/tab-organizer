import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { 
  Search, 
  Filter, 
  Globe, 
  Calendar,
  Tag,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Bot
} from 'lucide-react';
import { searchAPI, clusteringAPI } from '../services/api';
import Chatbot from '../components/Chatbot';

const SearchPage = () => {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState('semantic');
  const [filters, setFilters] = useState({
    dateRange: 'all',
    domain: '',
    cluster: '',
    minScore: 0.5
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);

  const { data: clusters } = useQuery('clusters', () => clusteringAPI.getClusters());

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    try {
      const response = await searchAPI.search(query, searchType, filters);
      setSearchResults(response.data);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Search Content</h1>
          <p className="mt-1 text-sm text-gray-500">
            Search through your scraped content using semantic or keyword search
          </p>
        </div>
        <button
          onClick={() => setShowChatbot(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
        >
          <Bot className="h-4 w-4 mr-2" />
          Ask Assistant
        </button>
      </div>

      {/* Search Form */}
      <div className="bg-white shadow rounded-lg p-6">
        <form onSubmit={handleSearch} className="space-y-4">
          {/* Main Search */}
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search for content, topics, or specific information..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="pl-10 pr-4 py-3 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-lg"
                />
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <select
                value={searchType}
                onChange={(e) => setSearchType(e.target.value)}
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

          {/* Advanced Filters Toggle */}
          <div className="flex justify-between items-center">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center text-sm text-gray-600 hover:text-gray-900"
            >
              <Filter className="h-4 w-4 mr-1" />
              Advanced Filters
              {showAdvanced ? (
                <ChevronUp className="h-4 w-4 ml-1" />
              ) : (
                <ChevronDown className="h-4 w-4 ml-1" />
              )}
            </button>
            {searchResults && (
              <div className="text-sm text-gray-500">
                Found {searchResults.total_count} results in {searchResults.search_time}ms
              </div>
            )}
          </div>

          {/* Advanced Filters */}
          {showAdvanced && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Date Range
                </label>
                <select
                  value={filters.dateRange}
                  onChange={(e) => handleFilterChange('dateRange', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="all">All Time</option>
                  <option value="today">Today</option>
                  <option value="week">This Week</option>
                  <option value="month">This Month</option>
                  <option value="year">This Year</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Domain
                </label>
                <input
                  type="text"
                  placeholder="example.com"
                  value={filters.domain}
                  onChange={(e) => handleFilterChange('domain', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Cluster
                </label>
                <select
                  value={filters.cluster}
                  onChange={(e) => handleFilterChange('cluster', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="">All Clusters</option>
                  {clusters?.data?.map(cluster => (
                    <option key={cluster.id} value={cluster.id}>
                      {cluster.label || `Cluster ${cluster.id}`}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Min Similarity Score
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={filters.minScore}
                  onChange={(e) => handleFilterChange('minScore', parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs text-gray-500 mt-1">
                  {filters.minScore}
                </div>
              </div>
            </div>
          )}
        </form>
      </div>

      {/* Search Results */}
      {searchResults && (
        <div className="space-y-4">
          {searchResults.results.length > 0 ? (
            searchResults.results.map((result, index) => (
              <SearchResult key={index} result={result} />
            ))
          ) : (
            <div className="bg-white shadow rounded-lg p-8 text-center">
              <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
              <p className="text-gray-500">
                Try adjusting your search terms or filters to find what you're looking for.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Search Suggestions */}
      {!searchResults && (
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Search Tips</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Semantic Search</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Search by meaning and context</li>
                <li>• "articles about machine learning"</li>
                <li>• "content related to web development"</li>
                <li>• "posts discussing climate change"</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Keyword Search</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Search for exact terms</li>
                <li>• "React hooks tutorial"</li>
                <li>• "API documentation"</li>
                <li>• "best practices guide"</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Chatbot Modal */}
      {showChatbot && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-4 mx-auto p-0 border w-full max-w-4xl h-5/6 shadow-lg rounded-md bg-white">
            <Chatbot onClose={() => setShowChatbot(false)} />
          </div>
        </div>
      )}
    </div>
  );
};

const SearchResult = ({ result }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white shadow rounded-lg p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <Globe className="h-4 w-4 text-gray-400" />
            <span className="text-sm text-gray-500">{result.domain}</span>
            {result.cluster_label && (
              <>
                <Tag className="h-4 w-4 text-gray-400" />
                <span className="text-sm text-indigo-600 bg-indigo-100 px-2 py-1 rounded">
                  {result.cluster_label}
                </span>
              </>
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
                {new Date(result.scraped_at).toLocaleDateString()}
              </div>
              <div>
                Similarity: {(result.similarity_score * 100).toFixed(1)}%
              </div>
            </div>
            
            <button
              onClick={() => setExpanded(!expanded)}
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

export default SearchPage;