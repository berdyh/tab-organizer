import { BarChart3, GitCompare, Users, X } from 'lucide-react';
import { useQuery } from 'react-query';
import { sessionAPI } from '../../../lib/api';

const CompareSessionsModal = ({ sessionIds, onClose }) => {
  const { data, isLoading } = useQuery(
    ['session-comparison', sessionIds],
    () => sessionAPI.compare(sessionIds),
    { enabled: sessionIds.length >= 2 },
  );

  const similarity = data?.data?.similarity_score ?? 0;
  const metrics = [
    {
      icon: BarChart3,
      label: 'Common Clusters',
      value: data?.data?.common_clusters ?? 0,
    },
    {
      icon: Users,
      label: 'Unique Content',
      value: data?.data?.unique_content ?? 0,
    },
    {
      icon: GitCompare,
      label: 'Similarity Score',
      value: `${(similarity * 100).toFixed(1)}%`,
    },
  ];

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-10 mx-auto p-5 border w-full max-w-4xl shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-medium text-gray-900">Session Comparison</h3>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
              <X className="h-5 w-5" />
            </button>
          </div>

          {isLoading ? (
            <div className="flex justify-center items-center h-64 text-gray-500">
              Loading comparison...
            </div>
          ) : (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {metrics.map((metric) => {
                  const Icon = metric.icon;
                  return (
                    <div key={metric.label} className="bg-gray-50 p-4 rounded-lg">
                    <div className="flex items-center">
                      <Icon className="h-5 w-5 text-gray-400 mr-2" />
                      <span className="text-sm font-medium text-gray-900">
                        {metric.label}
                      </span>
                    </div>
                    <div className="mt-2 text-2xl font-bold text-gray-900">
                      {metric.value}
                    </div>
                  </div>
                  );
                })}
              </div>

              <div className="bg-white border rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-4">Detailed Analysis</h4>
                <div className="text-sm text-gray-600">
                  {data?.data?.analysis || 'No detailed analysis available.'}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CompareSessionsModal;
