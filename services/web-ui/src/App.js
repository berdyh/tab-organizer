import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import URLManager from './pages/URLManager';
import SearchPage from './pages/SearchPage';
import SessionManager from './pages/SessionManager';
import ExportPage from './pages/ExportPage';
import ChatbotPage from './pages/ChatbotPage';
import AuthManager from './pages/AuthManager';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/urls" element={<URLManager />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/sessions" element={<SessionManager />} />
          <Route path="/export" element={<ExportPage />} />
          <Route path="/chat" element={<ChatbotPage />} />
          <Route path="/auth" element={<AuthManager />} />
        </Routes>
      </Layout>
    </QueryClientProvider>
  );
}

export default App;
