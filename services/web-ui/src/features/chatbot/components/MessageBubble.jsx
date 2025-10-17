import { Bot, Copy, ThumbsDown, ThumbsUp, User } from 'lucide-react';
import SourceCard from './SourceCard';

const MessageBubble = ({ message, onCopy, onFeedback, onSuggestionClick }) => {
  const isUser = message.type === 'user';
  const isError = message.isError;

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-xs lg:max-w-md ${isUser ? 'order-2' : 'order-1'}`}>
        <div className="flex items-center space-x-2 mb-1">
          {!isUser && <Bot className="h-4 w-4 text-indigo-600" />}
          {isUser && <User className="h-4 w-4 text-gray-600" />}
          {message.timestamp && (
            <span className="text-xs text-gray-500">
              {message.timestamp.toLocaleTimeString()}
            </span>
          )}
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

          {message.sources?.length > 0 && (
            <div className="mt-3 space-y-2">
              {message.sources.map((source, index) => (
                <SourceCard key={index} source={source} />
              ))}
            </div>
          )}

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

        {message.suggestions?.length > 0 && (
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

export default MessageBubble;
