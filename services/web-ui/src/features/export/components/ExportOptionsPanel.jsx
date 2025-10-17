const ExportOptionsPanel = ({ options, onOptionChange, templates, selectedFormat }) => (
  <div className="bg-white shadow rounded-lg p-6">
    <h3 className="text-lg font-medium text-gray-900 mb-4">Export Options</h3>
    <div className="space-y-4">
      <OptionToggle
        label="Include Metadata"
        description="URL metadata, timestamps, and processing information"
        checked={options.includeMetadata}
        onChange={(value) => onOptionChange('includeMetadata', value)}
      />
      <OptionToggle
        label="Include Clusters"
        description="Cluster assignments and labels"
        checked={options.includeClusters}
        onChange={(value) => onOptionChange('includeClusters', value)}
      />
      <OptionToggle
        label="Include Embeddings"
        description="Include vector embeddings in the export payload"
        checked={options.includeEmbeddings}
        onChange={(value) => onOptionChange('includeEmbeddings', value)}
      />

      {templates?.length > 0 && (
        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">Template</label>
          <select
            value={options.template}
            onChange={(event) => onOptionChange('template', event.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
          >
            {templates.map((template) => (
              <option key={template.id} value={template.id}>
                {template.name}
              </option>
            ))}
          </select>
        </div>
      )}

      {selectedFormat === 'notion' && (
        <div className="space-y-4 pt-4 border-t border-gray-200">
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-1">
              Notion Integration Token
            </label>
            <input
              type="password"
              value={options.notionToken}
              onChange={(event) => onOptionChange('notionToken', event.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="secret_xxxxx"
            />
            <p className="text-xs text-gray-500 mt-1">
              Create an internal integration in Notion and paste the token here.
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-1">
              Notion Database ID
            </label>
            <input
              type="text"
              value={options.notionDatabaseId}
              onChange={(event) => onOptionChange('notionDatabaseId', event.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="Paste the target database ID"
            />
            <p className="text-xs text-gray-500 mt-1">
              Open the database in Notion, copy its URL, and extract the last 32-character ID.
            </p>
          </div>
        </div>
      )}
    </div>
  </div>
);

const OptionToggle = ({ label, description, checked, onChange }) => (
  <div className="flex items-center justify-between">
    <div>
      <label className="text-sm font-medium text-gray-900">{label}</label>
      <p className="text-xs text-gray-500">{description}</p>
    </div>
    <input
      type="checkbox"
      checked={checked}
      onChange={(event) => onChange(event.target.checked)}
      className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
    />
  </div>
);

export default ExportOptionsPanel;
