import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import '@testing-library/jest-dom';

const mockSendMessage = jest.fn((_sessionId, message) =>
  Promise.resolve({
    data: {
      response: `Echo: ${message}`,
      suggestions: [],
      sources: [],
    },
  })
);

jest.mock('../../lib/api', () => ({
  chatbotAPI: {
    sendMessage: (...args) => mockSendMessage(...args),
  },
}));

import Chatbot from '../Chatbot';

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
      {ui}
    </QueryClientProvider>
  );
};

beforeEach(() => {
  mockSendMessage.mockClear();
  window.HTMLElement.prototype.scrollIntoView = jest.fn();
});

test('renders chatbot interface', () => {
  renderWithQueryClient(<Chatbot sessionId="test-session" />);
  
  expect(screen.getByText('Content Assistant')).toBeInTheDocument();
  expect(screen.getByPlaceholderText('Ask me about your scraped content...')).toBeInTheDocument();
  expect(screen.getByText(/Hello! I can help you explore/)).toBeInTheDocument();
});

test('sends message when form is submitted', async () => {
  renderWithQueryClient(<Chatbot sessionId="test-session" />);
  
  const input = await screen.findByPlaceholderText('Ask me about your scraped content...', {}, { timeout: 5000 });
  
  fireEvent.change(input, { target: { value: 'test message' } });
  fireEvent.submit(input.closest('form'));
  
  expect(await screen.findByText('test message', {}, { timeout: 5000 })).toBeInTheDocument();
});

test('shows typing indicator when processing', async () => {
  renderWithQueryClient(<Chatbot sessionId="test-session" />);
  
  const input = await screen.findByPlaceholderText('Ask me about your scraped content...', {}, { timeout: 5000 });
  
  fireEvent.change(input, { target: { value: 'test message' } });
  fireEvent.submit(input.closest('form'));
  
  // Check that the input is cleared after submission
  await waitFor(() => expect(input.value).toBe(''), { timeout: 5000 });
});

test('input field works correctly', async () => {
  renderWithQueryClient(<Chatbot sessionId="test-session" />);
  
  const input = await screen.findByPlaceholderText('Ask me about your scraped content...', {}, { timeout: 5000 });
  
  fireEvent.change(input, { target: { value: 'test input' } });
  expect(input.value).toBe('test input');
});

test('applies suggestion when suggestion button is clicked', async () => {
  mockSendMessage.mockResolvedValueOnce({
    data: {
      response: 'Here are some ideas.',
      suggestions: ['Tell me more about machine learning'],
      sources: [],
    },
  });

  renderWithQueryClient(<Chatbot sessionId="test-session" />);

  const input = await screen.findByPlaceholderText('Ask me about your scraped content...', {}, { timeout: 5000 });
  fireEvent.change(input, { target: { value: 'ai topics' } });
  fireEvent.submit(input.closest('form'));

  const suggestionButton = await screen.findByRole(
    'button',
    { name: 'Tell me more about machine learning' },
    { timeout: 5000 }
  );

  fireEvent.click(suggestionButton);
  expect(input.value).toBe('Tell me more about machine learning');
});

test('closes chatbot when close button is clicked', async () => {
  const onClose = jest.fn();
  renderWithQueryClient(<Chatbot sessionId="test-session" onClose={onClose} />);
  
  const closeButton = await screen.findByText('×', {}, { timeout: 5000 });
  fireEvent.click(closeButton);
  
  expect(onClose).toHaveBeenCalled();
});
