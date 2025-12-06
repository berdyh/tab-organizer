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

test('renders URL manager page', async () => {
  renderWithQueryClient(<URLManager />);

  expect(await screen.findByText('URL Manager', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Manage your URLs and organize them into collections', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('https://example.com', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('shows add URL button', async () => {
  renderWithQueryClient(<URLManager />);

  expect(await screen.findByText('Add URL', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Upload File', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('search functionality works', async () => {
  renderWithQueryClient(<URLManager />);

  const searchInput = await screen.findByPlaceholderText('Search URLs...', {}, { timeout: 5000 });
  expect(searchInput).toBeInTheDocument();

  fireEvent.change(searchInput, { target: { value: 'example' } });
  expect(searchInput.value).toBe('example');
});

test('status filter works', async () => {
  renderWithQueryClient(<URLManager />);

  const statusFilter = await screen.findByRole('combobox', { name: /status/i }, { timeout: 5000 });
  expect(statusFilter).toBeInTheDocument();

  fireEvent.change(statusFilter, { target: { value: 'completed' } });
  expect(statusFilter.value).toBe('completed');
});

test('URL selection works', async () => {
  renderWithQueryClient(<URLManager />);

  const checkboxes = await screen.findAllByRole('checkbox', {}, { timeout: 5000 });
  expect(checkboxes.length).toBeGreaterThan(0);
});

test('opens add URL modal', async () => {
  renderWithQueryClient(<URLManager />);

  const addButton = await screen.findByText('Add URL', {}, { timeout: 5000 });
  expect(addButton).toBeInTheDocument();

  fireEvent.click(addButton);

  expect(await screen.findByText('Add New URL', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('opens upload modal', async () => {
  renderWithQueryClient(<URLManager />);

  const uploadButton = await screen.findByText('Upload File', {}, { timeout: 5000 });
  expect(uploadButton).toBeInTheDocument();

  fireEvent.click(uploadButton);

  expect(await screen.findByText('Upload URL File', {}, { timeout: 5000 })).toBeInTheDocument();
});