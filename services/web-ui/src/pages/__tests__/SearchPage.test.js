import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import { act } from 'react';

// Mock the API
jest.mock('../../lib/api', () => ({
  searchAPI: {
    search: jest.fn(),
  },
  clusteringAPI: {
    getClusters: jest.fn(),
  },
  sessionAPI: {
    list: jest.fn(),
    create: jest.fn(),
    get: jest.fn(),
    update: jest.fn(),
    delete: jest.fn(),
    compare: jest.fn(),
    merge: jest.fn(),
    split: jest.fn(),
  },
}));

// Mock the Chatbot component to avoid complex testing
jest.mock('../../components/Chatbot', () => {
  return function MockChatbot({ onClose }) {
    return (
      <div data-testid="chatbot-mock">
        <button onClick={onClose}>Close Chatbot</button>
      </div>
    );
  };
});

import SearchPage from '../SearchPage';
import { searchAPI, clusteringAPI, sessionAPI } from '../../lib/api';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      cacheTime: 0,
    },
  },
});

const renderWithQueryClient = (ui) => {
  const queryClient = createTestQueryClient();
  let renderResult;
  act(() => {
    renderResult = render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          {ui}
        </MemoryRouter>
      </QueryClientProvider>
    );
  });
  return renderResult;
};

const defaultSearchResponse = {
  data: {
    results: [
      {
        id: '1',
        url: 'https://example.com/article',
        domain: 'example.com',
        title: 'Test Article',
        snippet: 'This is a test article snippet...',
        content: 'This is the full content of the test article.',
        similarity_score: 0.85,
        scraped_at: '2024-01-01T00:00:00Z',
        cluster_label: 'Technology'
      }
    ],
    total_count: 1,
    search_time: 150
  }
};

const defaultClustersResponse = {
  data: [
    { id: '1', label: 'Technology' },
    { id: '2', label: 'Science' }
  ]
};

beforeEach(() => {
  searchAPI.search.mockReset().mockResolvedValue(defaultSearchResponse);
  clusteringAPI.getClusters.mockReset().mockResolvedValue(defaultClustersResponse);
  sessionAPI.list.mockReset().mockResolvedValue({
    data: [
      { id: 'session-1', name: 'Session 1' },
      { id: 'session-2', name: 'Session 2' },
    ],
  });
});

test('renders search page', async () => {
  renderWithQueryClient(<SearchPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  await waitFor(() => expect(clusteringAPI.getClusters).toHaveBeenCalled());
  expect(clusteringAPI.getClusters).toHaveBeenCalledWith('session-1');
  
  expect(await screen.findByText('Search Content', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByPlaceholderText(/Search for content, topics/, {}, { timeout: 5000 })).toBeInTheDocument();
  const sessionSelect = await screen.findByLabelText('Session', {}, { timeout: 5000 });
  expect(sessionSelect.value).toBe('session-1');
});

test('search form works', async () => {
  renderWithQueryClient(<SearchPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  await waitFor(() => expect(clusteringAPI.getClusters).toHaveBeenCalled());
  
  const searchInput = await screen.findByPlaceholderText(/Search for content, topics/, {}, { timeout: 5000 });
  const searchButton = await screen.findByText('Search', {}, { timeout: 5000 });
  
  fireEvent.change(searchInput, { target: { value: 'test query' } });
  await act(async () => {
    fireEvent.click(searchButton);
  });
  await waitFor(() => expect(searchAPI.search).toHaveBeenCalledWith(
    'test query',
    'semantic',
    expect.objectContaining({ session_id: 'session-1' })
  ));
  
  expect(searchInput.value).toBe('test query');
});

test('search type selector works', async () => {
  renderWithQueryClient(<SearchPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  await waitFor(() => expect(clusteringAPI.getClusters).toHaveBeenCalled());
  
  const searchTypeSelect = await screen.findByRole('combobox', { name: /search type/i }, { timeout: 5000 });
  expect(searchTypeSelect).toBeInTheDocument();
  
  fireEvent.change(searchTypeSelect, { target: { value: 'keyword' } });
  expect(searchTypeSelect.value).toBe('keyword');
});

test('advanced filters toggle works', async () => {
  renderWithQueryClient(<SearchPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  await waitFor(() => expect(clusteringAPI.getClusters).toHaveBeenCalled());
  
  const advancedFiltersButton = await screen.findByText('Advanced Filters', {}, { timeout: 5000 });
  fireEvent.click(advancedFiltersButton);
  
  expect(await screen.findByText('Date Range', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Domain', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Cluster', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('displays search tips when no results', async () => {
  renderWithQueryClient(<SearchPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  await waitFor(() => expect(clusteringAPI.getClusters).toHaveBeenCalled());
  
  expect(await screen.findByText('Search Tips', {}, { timeout: 5000 })).toBeInTheDocument();
  const semanticHeadings = await screen.findAllByRole('heading', { name: 'Semantic Search' }, { timeout: 5000 });
  expect(semanticHeadings.length).toBeGreaterThan(0);
  const keywordHeadings = await screen.findAllByRole('heading', { name: 'Keyword Search' }, { timeout: 5000 });
  expect(keywordHeadings.length).toBeGreaterThan(0);
});

test('filter controls work', async () => {
  renderWithQueryClient(<SearchPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  await waitFor(() => expect(clusteringAPI.getClusters).toHaveBeenCalled());
  
  // Open advanced filters
  const advancedFiltersButton = await screen.findByText('Advanced Filters', {}, { timeout: 5000 });
  
  fireEvent.click(advancedFiltersButton);
  
  // Wait for filters to appear
  await screen.findByText('Date Range', {}, { timeout: 5000 });
  
  const dateRangeSelects = screen.getAllByRole('combobox');
  const dateRangeSelect = dateRangeSelects.find(select => select.value === 'all');
  expect(dateRangeSelect).toBeInTheDocument();
  
  fireEvent.change(dateRangeSelect, { target: { value: 'week' } });
  expect(dateRangeSelect.value).toBe('week');
});
