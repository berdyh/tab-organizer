import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { Bot, MessageSquare, Zap, Search, Database } from 'lucide-react';
import Chatbot from '../components/Chatbot';
import { sessionAPI } from '../lib/api';

const ChatbotPage = () => {
  const [selectedSession, setSelectedSession] = useState('');
  const [showChatbot, setShowChatbot] = useState(false);

  const { data: sessions } = useQuery('sessions', sessionAPI.list);

  const handleStartChat = () => {
    if (selectedSession) {
      setShowChatbot(true);
    }
  };

  if (showChatbot) {
    return (
      <div className="h-screen flex flex-col">
        <div className="flex-1">
          <Chatbot 
            sessionId={selectedSession}
            onClose={() => setShowChatbot(false)}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <Bot className="h-16 w-16 text-indigo-600 mx-auto mb-4" />
        <h1 className="text-3xl font-bold text-gray-900">Content Assistant</h1>
        <p className="mt-2 text-lg text-gray-600">
          Ask questions about your scraped content using natural language
        </p>
      </div>

      {/* Session Selection */}
      <div className="bg-white shadow rounded-lg p-6 max-w-2xl mx-auto">
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Select a Session to Explore
        </h3>
        
        {sessions?.data?.length > 0 ? (
          <div className="space-y-4">
            <select
              value={selectedSession}
              onChange={(e) => setSelectedSession(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="">Choose a session...</option>
              {sessions.data.map(session => (
                <option key={session.id} value={session.id}>
                  {session.name} ({session.url_count || 0} URLs, {session.cluster_count || 0} clusters)
                </option>
              ))}
            </select>
            
            <button
              onClick={handleStartChat}
              disabled={!selectedSession}
              className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              <MessageSquare className="h-4 w-4 mr-2" />
              Start Conversation
            </button>
          </div>
        ) : (
          <div className="text-center py-8">
            <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">
              No sessions available. Create a session and scrape some content first.
            </p>
          </div>
        )}
      </div>

      {/* Features */}
      <div className="max-w-4xl mx-auto">
        <h3 className="text-xl font-semibold text-gray-900 mb-6 text-center">
          What can the assistant help you with?
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <FeatureCard
            icon={Search}
            title="Content Discovery"
            description="Find articles, topics, and information within your scraped content using natural language queries."
            examples={[
              "Show me articles about AI",
              "Find content from last week",
              "What's the most popular topic?"
            ]}
          />
          
          <FeatureCard
            icon={Bot}
            title="Cluster Exploration"
            description="Explore and understand the clusters created from your content with intelligent explanations."
            examples={[
              "What clusters were created?",
              "Explain the Technology cluster",
              "Show me outlier articles"
            ]}
          />
          
          <FeatureCard
            icon={Zap}
            title="Smart Insights"
            description="Get summaries, statistics, and insights about your scraped content and analysis results."
            examples={[
              "Give me a content summary",
              "What domains have most articles?",
              "Show me scraping statistics"
            ]}
          />
        </div>
      </div>

      {/* Quick Start Tips */}
      <div className="bg-indigo-50 rounded-lg p-6 max-w-4xl mx-auto">
        <h4 className="text-lg font-medium text-indigo-900 mb-4">Quick Start Tips</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h5 className="font-medium text-indigo-800 mb-2">Natural Language Queries</h5>
            <ul className="space-y-1 text-indigo-700">
              <li>• Ask questions in plain English</li>
              <li>• Be specific about what you're looking for</li>
              <li>• Use follow-up questions for deeper exploration</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-indigo-800 mb-2">Best Practices</h5>
            <ul className="space-y-1 text-indigo-700">
              <li>• Start with broad questions, then narrow down</li>
              <li>• Use the suggested questions for guidance</li>
              <li>• Provide feedback to improve responses</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

const FeatureCard = ({ icon: Icon, title, description, examples }) => (
  <div className="bg-white border border-gray-200 rounded-lg p-6">
    <div className="flex items-center mb-4">
      <Icon className="h-8 w-8 text-indigo-600 mr-3" />
      <h4 className="text-lg font-medium text-gray-900">{title}</h4>
    </div>
    <p className="text-gray-600 mb-4">{description}</p>
    <div>
      <h5 className="text-sm font-medium text-gray-900 mb-2">Example queries:</h5>
      <ul className="space-y-1">
        {examples.map((example, index) => (
          <li key={index} className="text-sm text-gray-600 italic">
            "{example}"
          </li>
        ))}
      </ul>
    </div>
  </div>
);

export default ChatbotPage;
