import React, { useEffect, useState } from 'react';
import { useQuery } from 'react-query';
import { clusteringAPI, searchAPI, sessionAPI } from '../lib/api';
import Chatbot from '../components/Chatbot';
import {
  SearchForm,
  SearchHeader,
  SearchResultsList,
  SearchTips,
  SessionPicker,
} from '../features/search';

const SearchPage = () => {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState('semantic');
  const [filters, setFilters] = useState({
    dateRange: 'all',
    domain: '',
    cluster: '',
    minScore: 0.5,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [selectedSession, setSelectedSession] = useState('');

  const sessionsQuery = useQuery('sessions', sessionAPI.list);
  const sessions = sessionsQuery.data?.data ?? [];

  useEffect(() => {
    if (!selectedSession && sessions.length > 0) {
      setSelectedSession(sessions[0].id);
    }
  }, [sessions, selectedSession]);

  useEffect(() => {
    setFilters((prev) => ({ ...prev, cluster: '' }));
  }, [selectedSession]);

  const clustersQuery = useQuery(
    ['clusters', selectedSession],
    () => clusteringAPI.getClusters(selectedSession),
    {
      enabled: Boolean(selectedSession),
      select: (response) => response.data,
    },
  );

  const handleSearch = async (event) => {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }

    setIsSearching(true);
    try {
      const response = await searchAPI.search(query, searchType, {
        ...filters,
        session_id: selectedSession || undefined,
      });
      setSearchResults(response.data);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleFilterChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const resultSummary = searchResults
    ? { total_count: searchResults.total_count, search_time: searchResults.search_time }
    : null;

  return (
    <div className="space-y-6">
      <SearchHeader onOpenChat={() => setShowChatbot(true)} />

      <SessionPicker
        sessions={sessions}
        selectedSession={selectedSession}
        onSelect={setSelectedSession}
      />

      <SearchForm
        query={query}
        onQueryChange={setQuery}
        searchType={searchType}
        onSearchTypeChange={setSearchType}
        isSearching={isSearching}
        onSubmit={handleSearch}
        showAdvanced={showAdvanced}
        onToggleAdvanced={() => setShowAdvanced((prev) => !prev)}
        filters={filters}
        onFilterChange={handleFilterChange}
        clusters={clustersQuery.data}
        resultSummary={resultSummary}
      />

      {searchResults ? (
        <SearchResultsList results={searchResults.results} />
      ) : (
        <SearchTips />
      )}

      {showChatbot && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-4 mx-auto p-0 border w-full max-w-4xl h-5/6 shadow-lg rounded-md bg-white">
            <Chatbot sessionId={selectedSession} onClose={() => setShowChatbot(false)} />
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchPage;
