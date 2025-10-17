import apiClient from './client';

export const chatbotAPI = {
  sendMessage: (sessionId, message, context) =>
    apiClient.post('/api/chatbot-service/chat/message', {
      session_id: sessionId,
      message,
      context,
    }),
  getConversationHistory: (sessionId) =>
    apiClient.get(`/api/chatbot-service/chat/history/${sessionId}`),
  clearHistory: (sessionId) =>
    apiClient.delete(`/api/chatbot-service/chat/history/${sessionId}`),
  provideFeedback: (messageId, feedback) =>
    apiClient.post('/api/chatbot-service/chat/feedback', {
      message_id: messageId,
      feedback,
    }),
};

export default chatbotAPI;
