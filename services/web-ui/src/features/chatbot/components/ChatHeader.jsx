import { Bot } from 'lucide-react';

const ChatHeader = ({ onClose }) => (
  <div className="flex items-center justify-between p-4 border-b border-gray-200">
    <div className="flex items-center space-x-2">
      <Bot className="h-6 w-6 text-indigo-600" />
      <h3 className="text-lg font-medium text-gray-900">Content Assistant</h3>
    </div>
    {onClose && (
      <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
        ×
      </button>
    )}
  </div>
);

export default ChatHeader;
