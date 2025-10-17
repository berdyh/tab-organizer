import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

jest.mock('../../lib/api', () => ({
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

import ChatbotPage from '../ChatbotPage';
import { sessionAPI } from '../../lib/api';

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

beforeEach(() => {
  sessionAPI.list.mockReset();
  sessionAPI.list.mockResolvedValue({
    data: [
      {
        id: '1',
        name: 'Test Session',
        url_count: 10,
        cluster_count: 3,
      },
    ],
  });
});

test('renders chatbot page', async () => {
  renderWithProviders(<ChatbotPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  
  expect(await screen.findByText('Content Assistant', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Ask questions about your scraped content using natural language', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('shows session selection', async () => {
  renderWithProviders(<ChatbotPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  
  expect(await screen.findByText('Select a Session to Explore', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByRole('combobox', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('shows feature cards', async () => {
  renderWithProviders(<ChatbotPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  
  expect(await screen.findByText('Content Discovery', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Cluster Exploration', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Smart Insights', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('shows quick start tips', async () => {
  renderWithProviders(<ChatbotPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  
  expect(await screen.findByText('Quick Start Tips', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Natural Language Queries', {}, { timeout: 5000 })).toBeInTheDocument();
  expect(await screen.findByText('Best Practices', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('start conversation button is disabled without session selection', async () => {
  renderWithProviders(<ChatbotPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  
  const startButton = await screen.findByText('Start Conversation', {}, { timeout: 5000 });
  expect(startButton).toBeDisabled();
});

test('enables start conversation button when session is selected', async () => {
  renderWithProviders(<ChatbotPage />);
  await waitFor(() => expect(sessionAPI.list).toHaveBeenCalled());
  
  const select = await screen.findByRole('combobox', {}, { timeout: 5000 });
  
  fireEvent.change(select, { target: { value: '1' } });
  
  const startButton = await screen.findByText('Start Conversation', {}, { timeout: 5000 });
  expect(startButton).not.toBeDisabled();
});
