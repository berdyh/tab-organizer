const ExportHeader = ({ feedback, onDismiss }) => (
  <div>
    <h1 className="text-2xl font-bold text-gray-900">Export Data</h1>
    <p className="mt-1 text-sm text-gray-500">
      Export your analysis results to various formats and platforms
    </p>
    {feedback && (
      <div
        className={`mt-4 rounded-md p-4 text-sm flex items-start justify-between ${
          feedback.type === 'error'
            ? 'bg-red-50 text-red-700 border border-red-200'
            : 'bg-green-50 text-green-700 border border-green-200'
        }`}
      >
        <span>{feedback.message}</span>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="ml-4 text-xs font-medium text-current underline"
          >
            Dismiss
          </button>
        )}
      </div>
    )}
  </div>
);

export default ExportHeader;
