import { Loader, Send } from 'lucide-react';
import { forwardRef } from 'react';

const ChatInput = forwardRef(({ value, onChange, onSubmit, canSend, isSending }, inputRef) => (
  <div className="border-t border-gray-200 p-4">
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      className="flex space-x-2"
    >
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Ask me about your scraped content..."
        className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
        disabled={isSending}
      />
      <button
        type="submit"
        disabled={!canSend || isSending}
        className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isSending ? <Loader className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
      </button>
    </form>
  </div>
));

export default ChatInput;
