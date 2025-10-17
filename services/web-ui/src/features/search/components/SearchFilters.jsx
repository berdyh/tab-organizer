const SearchFilters = ({ filters, onFilterChange, clusters }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">Date Range</label>
      <select
        value={filters.dateRange}
        onChange={(event) => onFilterChange('dateRange', event.target.value)}
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
      <label className="block text-sm font-medium text-gray-700 mb-1">Domain</label>
      <input
        type="text"
        placeholder="example.com"
        value={filters.domain}
        onChange={(event) => onFilterChange('domain', event.target.value)}
        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
      />
    </div>

    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">Cluster</label>
      <select
        value={filters.cluster}
        onChange={(event) => onFilterChange('cluster', event.target.value)}
        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
      >
        <option value="">All Clusters</option>
        {clusters?.map((cluster) => (
          <option key={cluster.id} value={cluster.id}>
            {cluster.label || `Cluster ${cluster.id}`}
          </option>
        ))}
      </select>
    </div>

    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">Min Similarity Score</label>
      <input
        type="range"
        min="0"
        max="1"
        step="0.1"
        value={filters.minScore}
        onChange={(event) => onFilterChange('minScore', parseFloat(event.target.value))}
        className="w-full"
      />
      <div className="text-xs text-gray-500 mt-1">{filters.minScore}</div>
    </div>
  </div>
);

export default SearchFilters;
