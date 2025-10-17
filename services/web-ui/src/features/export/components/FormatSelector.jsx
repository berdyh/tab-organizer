import { CheckCircle } from 'lucide-react';
import { formatDescriptions, formatIcons } from '../constants/formats';

const FormatSelector = ({ selectedFormat, onSelect }) => (
  <div className="bg-white shadow rounded-lg p-6">
    <h3 className="text-lg font-medium text-gray-900 mb-4">Export Format</h3>
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {Object.entries(formatIcons).map(([format, Icon]) => (
        <button
          type="button"
          key={format}
          onClick={() => onSelect(format)}
          className={`relative text-left rounded-lg border p-4 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
            selectedFormat === format ? 'border-indigo-500 ring-2 ring-indigo-500' : 'border-gray-300'
          }`}
        >
          <div className="flex items-center">
            <Icon className="h-6 w-6 text-gray-400 mr-3" />
            <div>
              <div className="text-sm font-medium text-gray-900 capitalize">{format}</div>
              <div className="text-xs text-gray-500">{formatDescriptions[format]}</div>
            </div>
          </div>
          {selectedFormat === format && (
            <CheckCircle className="absolute top-2 right-2 h-5 w-5 text-indigo-500" />
          )}
        </button>
      ))}
    </div>
  </div>
);

export default FormatSelector;
