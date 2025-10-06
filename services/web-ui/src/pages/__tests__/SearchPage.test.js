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
    },
  },
});

const renderWithQueryClient = (ui) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        {ui}
      </MemoryRouter>
    </QueryClientProvider>
  );
};

test('renders search page', () => {
  renderWithQueryClient(<SearchPage />);
  
  expect(screen.getByText('Search Content')).toBeInTheDocument();
  expect(screen.getByPlaceholderText(/Search for content, topics/)).toBeInTheDocument();
});

test('search form works', async () => {
  renderWithQueryClient(<SearchPage />);
  
  const searchInput = screen.getByPlaceholderText(/Search for content, topics/);
  const searchButton = screen.getByText('Search');
  
  fireEvent.change(searchInput, { target: { value: 'test query' } });
  fireEvent.click(searchButton);
  
  expect(searchInput.value).toBe('test query');
});

test('search type selector works', () => {
  renderWithQueryClient(<SearchPage />);
  
  const searchTypeSelect = screen.getByDisplayValue('Semantic Search');
  expect(searchTypeSelect).toBeInTheDocument();
  
  fireEvent.change(searchTypeSelect, { target: { value: 'keyword' } });
  expect(searchTypeSelect.value).toBe('keyword');
});

test('advanced filters toggle works', () => {
  renderWithQueryClient(<SearchPage />);
  
  const advancedFiltersButton = screen.getByText('Advanced Filters');
  fireEvent.click(advancedFiltersButton);
  
  expect(screen.getByText('Date Range')).toBeInTheDocument();
  expect(screen.getByText('Domain')).toBeInTheDocument();
  expect(screen.getByText('Cluster')).toBeInTheDocument();
});

test('displays search tips when no results', () => {
  renderWithQueryClient(<SearchPage />);
  
  expect(screen.getByText('Search Tips')).toBeInTheDocument();
  expect(screen.getByText('Semantic Search')).toBeInTheDocument();
  expect(screen.getByText('Keyword Search')).toBeInTheDocument();
});

test('filter controls work', async () => {
  renderWithQueryClient(<SearchPage />);
  
  // Open advanced filters
  const advancedFiltersButton = screen.getByText('Advanced Filters');
  fireEvent.click(advancedFiltersButton);
  
  await waitFor(() => {
    const dateRangeSelect = screen.getByDisplayValue('All Time');
    expect(dateRangeSelect).toBeInTheDocument();
    
    fireEvent.change(dateRangeSelect, { target: { value: 'week' } });
    expect(dateRangeSelect.value).toBe('week');
  });
});