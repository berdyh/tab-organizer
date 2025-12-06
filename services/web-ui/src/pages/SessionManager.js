import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import {
  Plus,
  Settings,
  Trash2,
  Edit,
  Copy,
  Calendar,
  BarChart3,
  Users,
  GitCompare
} from 'lucide-react';
import { sessionAPI } from '../services/api';

const SessionManager = () => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedSessions, setSelectedSessions] = useState([]);
  const [showCompareModal, setShowCompareModal] = useState(false);

  const queryClient = useQueryClient();

  const { data: sessions, isLoading } = useQuery('sessions', sessionAPI.list);

  const deleteSessionMutation = useMutation(sessionAPI.delete, {
    onSuccess: () => {
      queryClient.invalidateQueries('sessions');
      setSelectedSessions([]);
    },
  });

  const compareSessionsMutation = useMutation(sessionAPI.compare, {
    onSuccess: (data) => {
      // Handle comparison results
      // eslint-disable-next-line no-console
      console.log('Comparison results:', data);
    },
  });

  const handleSelectSession = (sessionId) => {
    setSelectedSessions(prev =>
      prev.includes(sessionId)
        ? prev.filter(id => id !== sessionId)
        : [...prev, sessionId]
    );
  };

  const handleCompare = () => {
    if (selectedSessions.length >= 2) {
      compareSessionsMutation.mutate(selectedSessions);
      setShowCompareModal(true);
    }
  };

  const handleDeleteSelected = () => {
    if (window.confirm(`Delete ${selectedSessions.length} selected sessions?`)) {
      selectedSessions.forEach(id => deleteSessionMutation.mutate(id));
    }
  };

  if (isLoading) {
    return <div className="flex justify-center items-center h-64">Loading...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Session Manager</h1>
          <p className="mt-1 text-sm text-gray-500">
            Create, manage, and compare analysis sessions
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
        >
          <Plus className="h-4 w-4 mr-2" />
          New Session
        </button>
      </div>

      {/* Bulk Actions */}
      {selectedSessions.length > 0 && (
        <div className="bg-indigo-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-indigo-700">
              {selectedSessions.length} sessions selected
            </span>
            <div className="flex space-x-2">
              {selectedSessions.length >= 2 && (
                <button
                  onClick={handleCompare}
                  className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-indigo-700 bg-indigo-100 hover:bg-indigo-200"
                >
                  <GitCompare className="h-4 w-4 mr-1" />
                  Compare Sessions
                </button>
              )}
              <button
                onClick={handleDeleteSelected}
                disabled={deleteSessionMutation.isLoading}
                className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded text-red-700 bg-red-100 hover:bg-red-200 disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4 mr-1" />
                Delete Selected
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sessions Grid */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {sessions?.data?.map((session) => (
          <SessionCard
            key={session.id}
            session={session}
            isSelected={selectedSessions.includes(session.id)}
            onSelect={() => handleSelectSession(session.id)}
          />
        ))}
        {sessions?.data?.length === 0 && (
          <div className="col-span-full bg-white shadow rounded-lg p-8 text-center">
            <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No sessions yet</h3>
            <p className="text-gray-500 mb-4">
              Create your first analysis session to get started.
            </p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
            >
              <Plus className="h-4 w-4 mr-2" />
              Create Session
            </button>
          </div>
        )}
      </div>

      {/* Modals */}
      {showCreateModal && (
        <CreateSessionModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={() => {
            setShowCreateModal(false);
            queryClient.invalidateQueries('sessions');
          }}
        />
      )}

      {showCompareModal && (
        <CompareSessionsModal
          sessionIds={selectedSessions}
          onClose={() => setShowCompareModal(false)}
        />
      )}
    </div>
  );
};

const SessionCard = ({ session, isSelected, onSelect }) => {
  const [showMenu, setShowMenu] = useState(false);
  const queryClient = useQueryClient();

  const deleteSessionMutation = useMutation(sessionAPI.delete, {
    onSuccess: () => {
      queryClient.invalidateQueries('sessions');
    },
  });

  const handleDelete = () => {
    if (window.confirm('Delete this session?')) {
      deleteSessionMutation.mutate(session.id);
    }
  };

  return (
    <div className={`bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow ${isSelected ? 'ring-2 ring-indigo-500' : ''}`}>
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <input
              type="checkbox"
              checked={isSelected}
              onChange={onSelect}
              className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded mr-3"
            />
            <h3 className="text-lg font-medium text-gray-900 truncate">
              {session.name}
            </h3>
          </div>
          <div className="relative">
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="text-gray-400 hover:text-gray-600"
            >
              <Settings className="h-5 w-5" />
            </button>
            {showMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-10 border">
                <div className="py-1">
                  <button className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left">
                    <Edit className="h-4 w-4 mr-2" />
                    Edit
                  </button>
                  <button className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left">
                    <Copy className="h-4 w-4 mr-2" />
                    Duplicate
                  </button>
                  <button
                    onClick={handleDelete}
                    className="flex items-center px-4 py-2 text-sm text-red-700 hover:bg-red-50 w-full text-left"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        <p className="text-sm text-gray-600 mb-4 line-clamp-2">
          {session.description || 'No description provided'}
        </p>

        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">URLs Processed</span>
            <span className="font-medium">{session.url_count || 0}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Clusters Found</span>
            <span className="font-medium">{session.cluster_count || 0}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Last Updated</span>
            <span className="font-medium">
              {new Date(session.updated_at).toLocaleDateString()}
            </span>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center text-sm text-gray-500">
              <Calendar className="h-4 w-4 mr-1" />
              {new Date(session.created_at).toLocaleDateString()}
            </div>
            <div className="flex space-x-2">
              <button className="text-indigo-600 hover:text-indigo-800 text-sm font-medium">
                View Details
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const CreateSessionModal = ({ onClose, onSuccess }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const createSessionMutation = useMutation(sessionAPI.create, {
    onSuccess: () => {
      onSuccess();
    },
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;

    await createSessionMutation.mutateAsync(name.trim(), description.trim());
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Create New Session</h3>
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Session Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Analysis Session"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                required
              />
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe what this session is for..."
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
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
                disabled={createSessionMutation.isLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-md disabled:opacity-50"
              >
                {createSessionMutation.isLoading ? 'Creating...' : 'Create Session'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

const CompareSessionsModal = ({ sessionIds, onClose }) => {
  const { data: comparisonData, isLoading } = useQuery(
    ['session-comparison', sessionIds],
    () => sessionAPI.compare(sessionIds),
    { enabled: sessionIds.length >= 2 }
  );

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-10 mx-auto p-5 border w-4/5 max-w-4xl shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-medium text-gray-900">Session Comparison</h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>

          {isLoading ? (
            <div className="flex justify-center items-center h-64">
              <div className="text-gray-500">Loading comparison...</div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Comparison metrics */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center">
                    <BarChart3 className="h-5 w-5 text-gray-400 mr-2" />
                    <span className="text-sm font-medium text-gray-900">
                      Common Clusters
                    </span>
                  </div>
                  <div className="mt-2 text-2xl font-bold text-gray-900">
                    {comparisonData?.data?.common_clusters || 0}
                  </div>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center">
                    <Users className="h-5 w-5 text-gray-400 mr-2" />
                    <span className="text-sm font-medium text-gray-900">
                      Unique Content
                    </span>
                  </div>
                  <div className="mt-2 text-2xl font-bold text-gray-900">
                    {comparisonData?.data?.unique_content || 0}
                  </div>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center">
                    <GitCompare className="h-5 w-5 text-gray-400 mr-2" />
                    <span className="text-sm font-medium text-gray-900">
                      Similarity Score
                    </span>
                  </div>
                  <div className="mt-2 text-2xl font-bold text-gray-900">
                    {((comparisonData?.data?.similarity_score || 0) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Detailed comparison */}
              <div className="bg-white border rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-4">Detailed Analysis</h4>
                <div className="text-sm text-gray-600">
                  {comparisonData?.data?.analysis || 'No detailed analysis available.'}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SessionManager;