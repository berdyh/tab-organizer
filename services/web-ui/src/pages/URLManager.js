import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { 
  Plus, 
  Upload, 
  Trash2, 
  Edit, 
  Globe,
  CheckCircle,
  XCircle,
  Clock,
  AlertCircle,
  Search,
  Filter
} from 'lucide-react';
import { urlAPI, scrapingAPI } from '../services/api';

const URLManager = () => {
  const [selectedUrls, setSelectedUrls] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  
  const queryClient = useQueryClient();

  const { data: urls, isLoading } = useQuery('urls', () => urlAPI.list());
  
  const deleteUrlMutation = useMutation(urlAPI.delete, {
    onSuccess: () => {
      queryClient.invalidateQueries('urls');
      setSelectedUrls([]);
    },
  });

  const scrapeMutation = useMutation(scrapingAPI.batchScrape, {
    onSuccess: () => {
      queryClient.invalidateQueries('scraping-jobs');
    },
  });

  const filteredUrls = React.useMemo(() => {
    if (!urls?.data) return [];
    
    return urls.data.filter(url => {
      const matchesSearch = url.url.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           (url.title && url.title.toLowerCase().includes(searchTerm.toLowerCase()));
      const matchesStatus = statusFilter === 'all' || url.status === statusFilter;
      
      return matchesSearch && matchesStatus;
    });
  }, [urls?.data, searchTerm, statusFilter]);

  const handleSelectAll = () => {
    if (selectedUrls.length === filteredUrls.length) {
      setSelectedUrls([]);
    } else {
      setSelectedUrls(filteredUrls.map(url => url.id));
    }
  };

  const handleSelectUrl = (urlId) => {
    setSelectedUrls(prev => 
      prev.includes(urlId) 
        ? prev.filter(id => id !== urlId)
        : [...prev, urlId]
    );
  };

  const handleDeleteSelected = () => {
    if (window.confirm(`Delete ${selectedUrls.length} selected URLs?`)) {
      selectedUrls.forEach(id => deleteUrlMutation.mutate(id));
    }
  };

  const handleScrapeSelected = () => {
    const selectedUrlObjects = filteredUrls.filter(url => selectedUrls.includes(url.id));
    const urlsToScrape = selectedUrlObjects.map(url => url.url);
    
    scrapeMutation.mutate(urlsToScrape);
  };

  if (isLoading) {
    return <div className="flex justify-center items-center h-64">Loading...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">URL Manager</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage your URLs and organize them into collections
          </p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={() => setShowUploadModal(true)}
            className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload File
          </button>
          <button
            onClick={() => setShowAddModal(true)}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add URL
          </button>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search URLs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-gray-400" />
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-indigo-500 focus:border-indigo-500"
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

        {/* Bulk Actions */}
        {selectedUrls.length > 0 && (
          <div className="mt-4 flex items-center justify-between bg-indigo-50 rounded-lg p-4">
            <span className="text-sm text-indigo-700">
              {selectedUrls.length} URLs selected
            </span>
            <div className="flex space-x-2">
              <button
                onClick={handleScrapeSelected}
                disabled={scrapeMutation.isLoading}
                className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-indigo-700 bg-indigo-100 hover:bg-indigo-200 disabled:opacity-50"
              >
                <Globe className="h-4 w-4 mr-1" />
                Scrape Selected
              </button>
              <button
                onClick={handleDeleteSelected}
                disabled={deleteUrlMutation.isLoading}
                className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-red-700 bg-red-100 hover:bg-red-200 disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4 mr-1" />
                Delete Selected
              </button>
            </div>
          </div>
        )}
      </div>

      {/* URL Table */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        <div className="px-4 py-5 sm:p-6">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left">
                    <input
                      type="checkbox"
                      checked={selectedUrls.length === filteredUrls.length && filteredUrls.length > 0}
                      onChange={handleSelectAll}
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
                {filteredUrls.map((url) => (
                  <URLRow
                    key={url.id}
                    url={url}
                    isSelected={selectedUrls.includes(url.id)}
                    onSelect={() => handleSelectUrl(url.id)}
                  />
                ))}
                {filteredUrls.length === 0 && (
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

      {/* Modals */}
      {showAddModal && (
        <AddURLModal
          onClose={() => setShowAddModal(false)}
          onSuccess={() => {
            setShowAddModal(false);
            queryClient.invalidateQueries('urls');
          }}
        />
      )}

      {showUploadModal && (
        <UploadModal
          onClose={() => setShowUploadModal(false)}
          onSuccess={() => {
            setShowUploadModal(false);
            queryClient.invalidateQueries('urls');
          }}
        />
      )}
    </div>
  );
};

const URLRow = ({ url, isSelected, onSelect }) => {
  const statusConfig = {
    pending: { icon: Clock, color: 'text-yellow-500', bg: 'bg-yellow-100' },
    scraping: { icon: AlertCircle, color: 'text-blue-500', bg: 'bg-blue-100' },
    completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100' },
    failed: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-100' },
    auth_required: { icon: AlertCircle, color: 'text-orange-500', bg: 'bg-orange-100' },
  };

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
        {new Date(url.created_at).toLocaleDateString()}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
        <button className="text-indigo-600 hover:text-indigo-900 mr-3">
          <Edit className="h-4 w-4" />
        </button>
        <button className="text-red-600 hover:text-red-900">
          <Trash2 className="h-4 w-4" />
        </button>
      </td>
    </tr>
  );
};

const AddURLModal = ({ onClose, onSuccess }) => {
  const [url, setUrl] = useState('');
  const [isValidating, setIsValidating] = useState(false);

  const validateMutation = useMutation(urlAPI.validate, {
    onSuccess: () => {
      onSuccess();
    },
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setIsValidating(true);
    try {
      await validateMutation.mutateAsync(url.trim());
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Add New URL</h3>
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                URL
              </label>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                required
              />
            </div>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isValidating || validateMutation.isLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-md disabled:opacity-50"
              >
                {isValidating ? 'Validating...' : 'Add URL'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

const UploadModal = ({ onClose, onSuccess }) => {
  const [file, setFile] = useState(null);
  const [format, setFormat] = useState('text');

  const uploadMutation = useMutation(urlAPI.batch, {
    onSuccess: () => {
      onSuccess();
    },
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    await uploadMutation.mutateAsync(file, format);
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Upload URL File</h3>
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                File Format
              </label>
              <select
                value={format}
                onChange={(e) => setFormat(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="text">Plain Text</option>
                <option value="json">JSON</option>
                <option value="csv">CSV</option>
                <option value="excel">Excel</option>
              </select>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                File
              </label>
              <input
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
                accept=".txt,.json,.csv,.xlsx,.xls"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                required
              />
            </div>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={uploadMutation.isLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-md disabled:opacity-50"
              >
                {uploadMutation.isLoading ? 'Uploading...' : 'Upload'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default URLManager;