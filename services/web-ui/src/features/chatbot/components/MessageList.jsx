import { useEffect, useRef } from 'react';
import { Bot } from 'lucide-react';
import MessageBubble from './MessageBubble';

const MessageList = ({ messages, isTyping, onCopy, onFeedback, onSuggestionClick }) => {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          onCopy={onCopy}
          onFeedback={onFeedback}
          onSuggestionClick={onSuggestionClick}
        />
      ))}

      {isTyping && <TypingIndicator />}

      <div ref={endRef} />
    </div>
  );
};

const TypingIndicator = () => (
  <div className="flex items-center space-x-2 text-gray-500">
    <Bot className="h-5 w-5" />
    <div className="flex space-x-1">
      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
    </div>
    <span className="text-sm">Assistant is typing...</span>
  </div>
);

export default MessageList;
