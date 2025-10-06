import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import '@testing-library/jest-dom';
import App from './App';

// Mock the API module
jest.mock('./services/api', () => ({
  urlAPI: {
    list: jest.fn(() => Promise.resolve({ data: [] })),
  },
  sessionAPI: {
    list: jest.fn(() => Promise.resolve({ data: [] })),
  },
  scrapingAPI: {
    getJobs: jest.fn(() => Promise.resolve({ data: [] })),
  },
}));

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

const renderWithProviders = (ui) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        {ui}
      </MemoryRouter>
    </QueryClientProvider>
  );
};

test('renders web scraping tool header', () => {
  renderWithProviders(<App />);
  const headerElements = screen.getAllByText(/Web Scraping Tool/i);
  expect(headerElements.length).toBeGreaterThan(0);
});

test('renders navigation menu', () => {
  renderWithProviders(<App />);
  
  expect(screen.getAllByText('Dashboard').length).toBeGreaterThan(0);
  expect(screen.getAllByText('URL Manager').length).toBeGreaterThan(0);
  expect(screen.getAllByText('Search').length).toBeGreaterThan(0);
  expect(screen.getAllByText('Sessions').length).toBeGreaterThan(0);
  expect(screen.getAllByText('Export').length).toBeGreaterThan(0);
});

test('renders dashboard by default', () => {
  renderWithProviders(<App />);
  
  expect(screen.getAllByText('Dashboard').length).toBeGreaterThan(0);
  expect(screen.getByText('Overview of your web scraping and clustering activities')).toBeInTheDocument();
});