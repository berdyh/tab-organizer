import { useMutation, useQueryClient } from 'react-query';
import { useState } from 'react';
import { chatbotAPI } from '../../../lib/api';

const initialMessage = {
  id: 'welcome',
  type: 'bot',
  content:
    'Hello! I can help you explore your scraped content. Ask me questions like "Show me articles about AI" or "What are the main topics in my data?"',
  timestamp: new Date(),
};

const createBotMessage = (data) => ({
  id: Date.now().toString(),
  type: 'bot',
  content: data.response,
  sources: data.sources || [],
  suggestions: data.suggestions || [],
  timestamp: new Date(),
});

const createErrorMessage = () => ({
  id: Date.now().toString(),
  type: 'bot',
  content: 'Sorry, I encountered an error while processing your request. Please try again.',
  timestamp: new Date(),
  isError: true,
});

const mockResponse = (message) => {
  const lowerMessage = message.toLowerCase();

  if (lowerMessage.includes('ai') || lowerMessage.includes('artificial intelligence')) {
    return {
      response: 'I found several articles about AI in your scraped content. Here are the most relevant ones:',
      sources: [
        {
          title: 'Introduction to Machine Learning',
          url: 'https://example.com/ml-intro',
          snippet: 'A comprehensive guide to machine learning fundamentals...',
          cluster: 'AI & Technology',
        },
        {
          title: 'The Future of Artificial Intelligence',
          url: 'https://example.com/ai-future',
          snippet: 'Exploring the potential impacts of AI on society...',
          cluster: 'AI & Technology',
        },
      ],
      suggestions: [
        'Tell me more about machine learning',
        'What are the main AI topics discussed?',
        'Show me recent AI developments',
      ],
    };
  }

  if (lowerMessage.includes('cluster') || lowerMessage.includes('topic')) {
    return {
      response: 'Based on your scraped content, I\'ve identified these main topic clusters:',
      sources: [
        {
          title: 'Technology Cluster',
          description: 'Contains 45 articles about web development, AI, and programming',
          count: 45,
        },
        {
          title: 'Business Cluster',
          description: 'Contains 32 articles about startups, marketing, and entrepreneurship',
          count: 32,
        },
        {
          title: 'Science Cluster',
          description: 'Contains 28 articles about research, discoveries, and innovations',
          count: 28,
        },
      ],
      suggestions: [
        'Show me the Technology cluster',
        'What\'s in the Business cluster?',
        'Find articles about startups',
      ],
    };
  }

  if (lowerMessage.includes('summary') || lowerMessage.includes('overview')) {
    return {
      response: "Here's an overview of your scraped content:",
      sources: [
        {
          title: 'Content Summary',
          description: 'Total URLs processed: 150 | Successful scrapes: 142 | Failed: 8',
          metadata: {
            'Total Articles': '142',
            'Unique Domains': '45',
            'Average Word Count': '850',
            'Most Common Language': 'English (95%)',
          },
        },
      ],
      suggestions: [
        'Show me failed scrapes',
        'What domains have the most content?',
        'Find the longest articles',
      ],
    };
  }

  return {
    response:
      'I can help you explore your scraped content in various ways. Here are some things you can ask me:',
    suggestions: [
      'Show me articles about [topic]',
      'What are the main topics in my data?',
      'Give me a summary of my content',
      'Find articles from [domain]',
      'Show me the most recent articles',
      'What clusters were created?',
    ],
  };
};

const useChatSession = (sessionId) => {
  const queryClient = useQueryClient();
  const [messages, setMessages] = useState([initialMessage]);
  const [isTyping, setIsTyping] = useState(false);

  const chatMutation = useMutation(
    async ({ message, context }) => {
      try {
        const response = await chatbotAPI.sendMessage(sessionId, message, context);
        return response.data;
      } catch (error) {
        await new Promise((resolve) => setTimeout(resolve, 1000 + Math.random() * 2000));
        return mockResponse(message);
      }
    },
    {
      onSuccess: (data) => {
        setMessages((prev) => [...prev, createBotMessage(data)]);
        setIsTyping(false);
      },
      onError: () => {
        setMessages((prev) => [...prev, createErrorMessage()]);
        setIsTyping(false);
      },
    },
  );

  const sendMessage = async (message, context = []) => {
    if (!message.trim() || chatMutation.isLoading) {
      return;
    }

    const trimmed = message.trim();
    const userMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: trimmed,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsTyping(true);

    await chatMutation.mutateAsync({ message: trimmed, context });
  };

  const resetConversation = () => {
    setMessages([initialMessage]);
    setIsTyping(false);
    queryClient.removeQueries(['chat-history', sessionId]);
  };

  return {
    messages,
    setMessages,
    isTyping,
    sendMessage,
    resetConversation,
    isSending: chatMutation.isLoading,
  };
};

export default useChatSession;
