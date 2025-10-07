import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import SearchPage from '../SearchPage';

// Mock the API
jest.mock('../../services/api', () => ({
  searchAPI: {
    search: jest.fn(() => Promise.resolve({
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
    })),
  },
  clusteringAPI: {
    getClusters: jest.fn(() => Promise.resolve({
      data: [
        { id: '1', label: 'Technology' },
        { id: '2', label: 'Science' }
      ]
    })),
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
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        {ui}
      </MemoryRouter>
    </QueryClientProvider>
  );
};

test('renders search page', async () => {
  renderWithQueryClient(<SearchPage />);
  
  expect(await screen.findByText('Search Content', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByPlaceholderText(/Search for content, topics/, {}, { timeout: 5000 })).toBeInTheDocument();
});

test('search form works', async () => {
  renderWithQueryClient(<SearchPage />);
  
  const searchInput = await screen.findByPlaceholderText(/Search for content, topics/, {}, { timeout: 5000 });
  const searchButton = await screen.findByText('Search', {}, { timeout: 5000 });
  
  fireEvent.change(searchInput, { target: { value: 'test query' } });
  fireEvent.click(searchButton);
  
  expect(searchInput.value).toBe('test query');
});

test('search type selector works', async () => {
  renderWithQueryClient(<SearchPage />);
  
  const searchTypeSelect = await screen.findByRole('combobox', { name: /search type/i }, { timeout: 5000 });
  expect(searchTypeSelect).toBeInTheDocument();
  
  fireEvent.change(searchTypeSelect, { target: { value: 'keyword' } });
  expect(searchTypeSelect.value).toBe('keyword');
});

test('advanced filters toggle works', async () => {
  renderWithQueryClient(<SearchPage />);
  
  const advancedFiltersButton = await screen.findByText('Advanced Filters', {}, { timeout: 5000 });
  fireEvent.click(advancedFiltersButton);
  
  expect(await screen.findByText('Date Range', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Domain', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Cluster', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('displays search tips when no results', async () => {
  renderWithQueryClient(<SearchPage />);
  
  expect(await screen.findByText('Search Tips', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Semantic Search', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Keyword Search', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('filter controls work', async () => {
  renderWithQueryClient(<SearchPage />);
  
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