import { Filter, Search } from 'lucide-react';

const UrlFilters = ({
  searchTerm,
  onSearchChange,
  statusFilter,
  onStatusChange,
  children,
}) => (
  <div className="bg-white shadow rounded-lg p-6">
    <div className="flex flex-col sm:flex-row gap-4">
      <div className="flex-1">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search URLs..."
            value={searchTerm}
            onChange={(event) => onSearchChange(event.target.value)}
            className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>
      </div>
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-gray-400" />
          <select
            value={statusFilter}
            onChange={(event) => onStatusChange(event.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-indigo-500 focus:border-indigo-500"
            aria-label="Status filter"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="scraping">Scraping</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="auth_required">Auth Required</option>
          </select>
        </div>
      </div>
    </div>
    {children && <div className="mt-4">{children}</div>}
  </div>
);

export default UrlFilters;
