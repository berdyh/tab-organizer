const PreviewModal = ({ sessionId, format, onClose }) => (
  <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
    <div className="relative top-10 mx-auto p-5 border w-4/5 max-w-4xl shadow-lg rounded-md bg-white">
      <div className="mt-3">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-medium text-gray-900">Export Preview</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            ×
          </button>
        </div>

        <div className="bg-gray-50 rounded-lg p-4 h-96 overflow-y-auto">
          <div className="text-sm text-gray-600 font-mono space-y-4">
            <section>
              <div className="text-gray-800 font-bold"># Exported Session Data</div>
              <div className="mt-2">
                **Format:** {format.toUpperCase()}<br />
                **Session:** {sessionId}<br />
                **Generated:** {new Date().toLocaleString()}
              </div>
            </section>

            <section>
              <div className="text-gray-800 font-bold">## Clusters</div>
              <div className="mt-2">
                - **Cluster 1:** AI and Machine Learning (15 articles)<br />
                - **Cluster 2:** Web Development (23 articles)<br />
                - **Cluster 3:** Data Science (8 articles)
              </div>
            </section>

            <section>
              <div className="text-gray-800 font-bold">## Sample Content</div>
              <div className="mt-2 bg-white p-2 rounded border">
                **Title:** Introduction to React Hooks<br />
                **URL:** https://example.com/react-hooks<br />
                **Cluster:** Web Development<br />
                **Content:** React Hooks provide a way to use state and other React features...
              </div>
            </section>
          </div>
        </div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md"
          >
            Close Preview
          </button>
        </div>
      </div>
    </div>
  </div>
);

export default PreviewModal;
