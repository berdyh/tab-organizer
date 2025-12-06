import React, { useState, useRef, useEffect } from 'react';
import { useMutation } from 'react-query';
import {
  Send,
  Bot,
  User,
  Loader,
  ExternalLink,
  Copy,
  ThumbsUp,
  ThumbsDown
} from 'lucide-react';
import { chatbotAPI } from '../services/api';

const Chatbot = ({ sessionId, onClose }) => {
  const [messages, setMessages] = useState([
    {
      id: '1',
      type: 'bot',
      content: 'Hello! I can help you explore your scraped content. Ask me questions like "Show me articles about AI" or "What are the main topics in my data?"',
      timestamp: new Date(),
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Chatbot API with fallback to mock responses
  const chatMutation = useMutation(
    async ({ message, sessionId, context }) => {
      try {
        // Try to use real API first
        const response = await chatbotAPI.sendMessage(sessionId, message, context);
        return response.data;
      } catch (error) {
        // Fallback to mock responses for development
        // eslint-disable-next-line no-console
        console.log('Using mock chatbot response');
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

        const lowerMessage = message.toLowerCase();

      if (lowerMessage.includes('ai') || lowerMessage.includes('artificial intelligence')) {
        return {
          response: "I found several articles about AI in your scraped content. Here are the most relevant ones:",
          sources: [
            {
              title: "Introduction to Machine Learning",
              url: "https://example.com/ml-intro",
              snippet: "A comprehensive guide to machine learning fundamentals...",
              cluster: "AI & Technology"
            },
            {
              title: "The Future of Artificial Intelligence",
              url: "https://example.com/ai-future",
              snippet: "Exploring the potential impacts of AI on society...",
              cluster: "AI & Technology"
            }
          ],
          suggestions: [
            "Tell me more about machine learning",
            "What are the main AI topics discussed?",
            "Show me recent AI developments"
          ]
        };
      } else if (lowerMessage.includes('cluster') || lowerMessage.includes('topic')) {
        return {
          response: "Based on your scraped content, I've identified these main topic clusters:",
          sources: [
            {
              title: "Technology Cluster",
              description: "Contains 45 articles about web development, AI, and programming",
              count: 45
            },
            {
              title: "Business Cluster",
              description: "Contains 32 articles about startups, marketing, and entrepreneurship",
              count: 32
            },
            {
              title: "Science Cluster",
              description: "Contains 28 articles about research, discoveries, and innovations",
              count: 28
            }
          ],
          suggestions: [
            "Show me the Technology cluster",
            "What's in the Business cluster?",
            "Find articles about startups"
          ]
        };
      } else if (lowerMessage.includes('summary') || lowerMessage.includes('overview')) {
        return {
          response: "Here's an overview of your scraped content:",
          sources: [
            {
              title: "Content Summary",
              description: "Total URLs processed: 150 | Successful scrapes: 142 | Failed: 8",
              metadata: {
                "Total Articles": "142",
                "Unique Domains": "45",
                "Average Word Count": "850",
                "Most Common Language": "English (95%)"
              }
            }
          ],
          suggestions: [
            "Show me failed scrapes",
            "What domains have the most content?",
            "Find the longest articles"
          ]
        };
      } else {
        return {
          response: "I can help you explore your scraped content in various ways. Here are some things you can ask me:",
          suggestions: [
            "Show me articles about [topic]",
            "What are the main topics in my data?",
            "Give me a summary of my content",
            "Find articles from [domain]",
            "Show me the most recent articles",
            "What clusters were created?"
          ]
        };
      }
      }
    },
    {
      onSuccess: (data) => {
        const botMessage = {
          id: Date.now().toString(),
          type: 'bot',
          content: data.response,
          sources: data.sources || [],
          suggestions: data.suggestions || [],
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, botMessage]);
        setIsTyping(false);
      },
      onError: () => {
        const errorMessage = {
          id: Date.now().toString(),
          type: 'bot',
          content: 'Sorry, I encountered an error while processing your request. Please try again.',
          timestamp: new Date(),
          isError: true,
        };
        setMessages(prev => [...prev, errorMessage]);
        setIsTyping(false);
      }
    }
  );

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || chatMutation.isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);

    const messageToSend = inputMessage.trim();
    setInputMessage('');

    // Send to chatbot API
    chatMutation.mutate({
      message: messageToSend,
      sessionId,
      context: messages.slice(-5) // Send last 5 messages for context
    });
  };

  const handleSuggestionClick = (suggestion) => {
    setInputMessage(suggestion);
    inputRef.current?.focus();
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const handleFeedback = (messageId, feedback) => {
    // Handle feedback (thumbs up/down)
    // eslint-disable-next-line no-console
    console.log('Feedback:', messageId, feedback);
  };

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <Bot className="h-6 w-6 text-indigo-600" />
          <h3 className="text-lg font-medium text-gray-900">Content Assistant</h3>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            ×
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onCopy={copyToClipboard}
            onFeedback={handleFeedback}
            onSuggestionClick={handleSuggestionClick}
          />
        ))}

        {isTyping && (
          <div className="flex items-center space-x-2 text-gray-500">
            <Bot className="h-5 w-5" />
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            </div>
            <span className="text-sm">Assistant is typing...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-4">
        <form onSubmit={handleSendMessage} className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask me about your scraped content..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
            disabled={chatMutation.isLoading}
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || chatMutation.isLoading}
            className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {chatMutation.isLoading ? (
              <Loader className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
};

const MessageBubble = ({ message, onCopy, onFeedback, onSuggestionClick }) => {
  const isUser = message.type === 'user';
  const isError = message.isError;

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-xs lg:max-w-md ${isUser ? 'order-2' : 'order-1'}`}>
        <div className="flex items-center space-x-2 mb-1">
          {!isUser && <Bot className="h-4 w-4 text-indigo-600" />}
          {isUser && <User className="h-4 w-4 text-gray-600" />}
          <span className="text-xs text-gray-500">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>

        <div
          className={`px-4 py-2 rounded-lg ${
            isUser
              ? 'bg-indigo-600 text-white'
              : isError
              ? 'bg-red-50 text-red-800 border border-red-200'
              : 'bg-gray-100 text-gray-900'
          }`}
        >
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>

          {/* Sources */}
          {message.sources && message.sources.length > 0 && (
            <div className="mt-3 space-y-2">
              {message.sources.map((source, index) => (
                <SourceCard key={index} source={source} />
              ))}
            </div>
          )}

          {/* Message Actions */}
          {!isUser && !isError && (
            <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-200">
              <div className="flex space-x-2">
                <button
                  onClick={() => onCopy(message.content)}
                  className="text-xs text-gray-500 hover:text-gray-700 flex items-center"
                >
                  <Copy className="h-3 w-3 mr-1" />
                  Copy
                </button>
              </div>
              <div className="flex space-x-1">
                <button
                  onClick={() => onFeedback(message.id, 'positive')}
                  className="text-gray-400 hover:text-green-600"
                >
                  <ThumbsUp className="h-3 w-3" />
                </button>
                <button
                  onClick={() => onFeedback(message.id, 'negative')}
                  className="text-gray-400 hover:text-red-600"
                >
                  <ThumbsDown className="h-3 w-3" />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Suggestions */}
        {message.suggestions && message.suggestions.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-xs text-gray-500">Suggested questions:</p>
            {message.suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => onSuggestionClick(suggestion)}
                className="block w-full text-left text-xs text-indigo-600 hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 px-2 py-1 rounded border border-indigo-200"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const SourceCard = ({ source }) => {
  if (source.url) {
    // Article source
    return (
      <div className="bg-white border border-gray-200 rounded p-3 text-sm">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h4 className="font-medium text-gray-900 mb-1">{source.title}</h4>
            {source.snippet && (
              <p className="text-gray-600 text-xs mb-2">{source.snippet}</p>
            )}
            {source.cluster && (
              <span className="inline-block bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded">
                {source.cluster}
              </span>
            )}
          </div>
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-2 text-indigo-600 hover:text-indigo-800"
          >
            <ExternalLink className="h-4 w-4" />
          </a>
        </div>
      </div>
    );
  } else if (source.count !== undefined) {
    // Cluster source
    return (
      <div className="bg-white border border-gray-200 rounded p-3 text-sm">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-medium text-gray-900">{source.title}</h4>
            <p className="text-gray-600 text-xs">{source.description}</p>
          </div>
          <span className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded font-medium">
            {source.count} articles
          </span>
        </div>
      </div>
    );
  } else if (source.metadata) {
    // Summary source
    return (
      <div className="bg-white border border-gray-200 rounded p-3 text-sm">
        <h4 className="font-medium text-gray-900 mb-2">{source.title}</h4>
        <p className="text-gray-600 text-xs mb-2">{source.description}</p>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(source.metadata).map(([key, value]) => (
            <div key={key} className="text-xs">
              <span className="text-gray-500">{key}:</span>
              <span className="ml-1 font-medium text-gray-900">{value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return null;
};

export default Chatbot;