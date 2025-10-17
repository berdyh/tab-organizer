const SearchTips = () => (
  <div className="bg-white shadow rounded-lg p-6">
    <h3 className="text-lg font-medium text-gray-900 mb-4">Search Tips</h3>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <TipSection
        title="Semantic Search"
        tips={[
          'Search by meaning and context',
          '"articles about machine learning"',
          '"content related to web development"',
          '"posts discussing climate change"',
        ]}
      />
      <TipSection
        title="Keyword Search"
        tips={[
          'Search for exact terms',
          '"React hooks tutorial"',
          '"API documentation"',
          '"best practices guide"',
        ]}
      />
    </div>
  </div>
);

const TipSection = ({ title, tips }) => (
  <div>
    <h4 className="font-medium text-gray-900 mb-2">{title}</h4>
    <ul className="text-sm text-gray-600 space-y-1">
      {tips.map((tip, index) => (
        <li key={index}>• {tip}</li>
      ))}
    </ul>
  </div>
);

export default SearchTips;
