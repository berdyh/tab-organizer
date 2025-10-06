import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import URLManager from '../URLManager';

// Mock the API
jest.mock('../../services/api', () => ({
  urlAPI: {
    list: jest.fn(() => Promise.resolve({
      data: [
        {
          id: '1',
          url: 'https://example.com',
          domain: 'example.com',
          title: 'Example Site',
          status: 'completed',
          created_at: '2024-01-01T00:00:00Z'
        }
      ]
    })),
    validate: jest.fn(() => Promise.resolve({ data: { valid: true } })),
    batch: jest.fn(() => Promise.resolve({ data: { job_id: 'test-job' } })),
    delete: jest.fn(() => Promise.resolve()),
  },
  scrapingAPI: {
    batchScrape: jest.fn(() => Promise.resolve({ data: { job_id: 'scrape-job' } })),
  },
}));

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

test('renders URL manager page', async () => {
  renderWithQueryClient(<URLManager />);
  
  expect(screen.getByText('URL Manager')).toBeInTheDocument();
  expect(screen.getByText('Manage your URLs and organize them into collections')).toBeInTheDocument();
  
  await waitFor(() => {
    expect(screen.getByText('https://example.com')).toBeInTheDocument();
  });
});

test('shows add URL button', () => {
  renderWithQueryClient(<URLManager />);
  
  expect(screen.getByText('Add URL')).toBeInTheDocument();
  expect(screen.getByText('Upload File')).toBeInTheDocument();
});

test('search functionality works', async () => {
  renderWithQueryClient(<URLManager />);
  
  const searchInput = screen.getByPlaceholderText('Search URLs...');
  expect(searchInput).toBeInTheDocument();
  
  fireEvent.change(searchInput, { target: { value: 'example' } });
  expect(searchInput.value).toBe('example');
});

test('status filter works', async () => {
  renderWithQueryClient(<URLManager />);
  
  const statusFilter = screen.getByDisplayValue('All Status');
  expect(statusFilter).toBeInTheDocument();
  
  fireEvent.change(statusFilter, { target: { value: 'completed' } });
  expect(statusFilter.value).toBe('completed');
});

test('URL selection works', async () => {
  renderWithQueryClient(<URLManager />);
  
  await waitFor(() => {
    const checkboxes = screen.getAllByRole('checkbox');
    expect(checkboxes.length).toBeGreaterThan(0);
  });
});

test('opens add URL modal', () => {
  renderWithQueryClient(<URLManager />);
  
  const addButton = screen.getByText('Add URL');
  fireEvent.click(addButton);
  
  expect(screen.getByText('Add New URL')).toBeInTheDocument();
});

test('opens upload modal', () => {
  renderWithQueryClient(<URLManager />);
  
  const uploadButton = screen.getByText('Upload File');
  fireEvent.click(uploadButton);
  
  expect(screen.getByText('Upload URL File')).toBeInTheDocument();
});