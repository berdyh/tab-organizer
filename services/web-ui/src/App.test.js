import { render, screen, waitFor } from '@testing-library/react';
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
      cacheTime: 0,
    },
  },
});

const renderWithProviders = (ui) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        {ui}
      </MemoryRouter>
    </QueryClientProvider>
  );
};

test('renders web scraping tool header', async () => {
  renderWithProviders(<App />);
  
  const headerElements = await screen.findAllByText(/Web Scraping Tool/i, {}, { timeout: 5000 });
  expect(headerElements.length).toBeGreaterThan(0);
});

test('renders navigation menu', async () => {
  renderWithProviders(<App />);
  
  expect((await screen.findAllByText('Dashboard', {}, { timeout: 5000 })).length).toBeGreaterThan(0);
  expect((await screen.findAllByText('URL Manager', {}, { timeout: 5000 })).length).toBeGreaterThan(0);
  expect((await screen.findAllByText('Search', {}, { timeout: 5000 })).length).toBeGreaterThan(0);
  expect((await screen.findAllByText('Sessions', {}, { timeout: 5000 })).length).toBeGreaterThan(0);
  expect((await screen.findAllByText('Export', {}, { timeout: 5000 })).length).toBeGreaterThan(0);
});

test('renders dashboard by default', async () => {
  renderWithProviders(<App />);
  
  expect((await screen.findAllByText('Dashboard', {}, { timeout: 5000 })).length).toBeGreaterThan(0);
  expect(await screen.findByText('Overview of your web scraping and clustering activities', {}, { timeout: 5000 })).toBeInTheDocument();
});