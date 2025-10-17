import React, { useRef, useState } from 'react';
import {
  ChatHeader,
  ChatInput,
  MessageList,
  useChatSession,
} from '../features/chatbot';

const Chatbot = ({ sessionId, onClose }) => {
  const { messages, sendMessage, isTyping, isSending } = useChatSession(sessionId);
  const [inputMessage, setInputMessage] = useState('');
  const inputRef = useRef(null);

  const handleSubmit = () => {
    if (!inputMessage.trim()) {
      return;
    }
    const context = messages.slice(-5);
    void sendMessage(inputMessage, context);
    setInputMessage('');
  };

  const handleSuggestionClick = (suggestion) => {
    setInputMessage(suggestion);
    inputRef.current?.focus();
  };

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
  };

  const handleFeedback = (messageId, feedback) => {
    // eslint-disable-next-line no-console
    console.log('Feedback:', messageId, feedback);
  };

  return (
    <div className="flex flex-col h-full bg-white">
      <ChatHeader onClose={onClose} />
      <MessageList
        messages={messages}
        isTyping={isTyping}
        onCopy={handleCopy}
        onFeedback={handleFeedback}
        onSuggestionClick={handleSuggestionClick}
      />
      <ChatInput
        ref={inputRef}
        value={inputMessage}
        onChange={setInputMessage}
        onSubmit={handleSubmit}
        canSend={Boolean(inputMessage.trim())}
        isSending={isSending}
      />
    </div>
  );
};

export default Chatbot;
