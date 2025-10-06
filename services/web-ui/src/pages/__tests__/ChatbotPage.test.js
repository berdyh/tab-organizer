import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import ChatbotPage from '../ChatbotPage';

// Mock the API
jest.mock('../../services/api', () => ({
  sessionAPI: {
    list: jest.fn(() => Promise.resolve({
      data: [
        {
          id: '1',
          name: 'Test Session',
          url_count: 10,
          cluster_count: 3
        }
      ]
    })),
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

test('renders chatbot page', () => {
  renderWithProviders(<ChatbotPage />);
  
  expect(screen.getByText('Content Assistant')).toBeInTheDocument();
  expect(screen.getByText('Ask questions about your scraped content using natural language')).toBeInTheDocument();
});

test('shows session selection', () => {
  renderWithProviders(<ChatbotPage />);
  
  expect(screen.getByText('Select a Session to Explore')).toBeInTheDocument();
  expect(screen.getByText('Choose a session...')).toBeInTheDocument();
});

test('shows feature cards', () => {
  renderWithProviders(<ChatbotPage />);
  
  expect(screen.getByText('Content Discovery')).toBeInTheDocument();
  expect(screen.getByText('Cluster Exploration')).toBeInTheDocument();
  expect(screen.getByText('Smart Insights')).toBeInTheDocument();
});

test('shows quick start tips', () => {
  renderWithProviders(<ChatbotPage />);
  
  expect(screen.getByText('Quick Start Tips')).toBeInTheDocument();
  expect(screen.getByText('Natural Language Queries')).toBeInTheDocument();
  expect(screen.getByText('Best Practices')).toBeInTheDocument();
});

test('start conversation button is disabled without session selection', () => {
  renderWithProviders(<ChatbotPage />);
  
  const startButton = screen.getByText('Start Conversation');
  expect(startButton).toBeDisabled();
});

test('enables start conversation button when session is selected', async () => {
  renderWithProviders(<ChatbotPage />);
  
  const select = screen.getByDisplayValue('Choose a session...');
  fireEvent.change(select, { target: { value: '1' } });
  
  const startButton = screen.getByText('Start Conversation');
  expect(startButton).not.toBeDisabled();
});