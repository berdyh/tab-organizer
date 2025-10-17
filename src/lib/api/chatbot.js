import apiClient from './client';

export const chatbotApi = {
  sendMessage(sessionId, message, context) {
    return apiClient.post('/api/chatbot-service/chat/message', {
      session_id: sessionId,
      message,
      context,
    });
  },

  getConversationHistory(sessionId) {
    return apiClient.get(`/api/chatbot-service/chat/history/${sessionId}`);
  },

  clearHistory(sessionId) {
    return apiClient.delete(`/api/chatbot-service/chat/history/${sessionId}`);
  },

  provideFeedback(messageId, feedback) {
    return apiClient.post('/api/chatbot-service/chat/feedback', {
      message_id: messageId,
      feedback,
    });
  },
};

export default chatbotApi;
