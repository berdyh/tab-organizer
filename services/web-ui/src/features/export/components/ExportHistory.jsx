import { useQuery } from 'react-query';
import { AlertCircle, CheckCircle, Clock } from 'lucide-react';

const statusIcons = {
  completed: CheckCircle,
  processing: Clock,
  failed: AlertCircle,
};

const statusColors = {
  completed: 'text-green-500',
  processing: 'text-blue-500',
  failed: 'text-red-500',
};

const ExportHistory = () => {
  const { data: exportJobs } = useQuery('export-jobs', () =>
    Promise.resolve({
      data: [
        {
          id: '1',
          session_name: 'AI Research Session',
          format: 'markdown',
          status: 'completed',
          created_at: new Date().toISOString(),
          file_size: '2.4 MB',
        },
        {
          id: '2',
          session_name: 'Web Dev Articles',
          format: 'notion',
          status: 'processing',
          created_at: new Date(Date.now() - 3600000).toISOString(),
          progress: 75,
        },
        {
          id: '3',
          session_name: 'Tech News',
          format: 'obsidian',
          status: 'failed',
          created_at: new Date(Date.now() - 7200000).toISOString(),
          error: 'Template not found',
        },
      ],
    }),
  );

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Exports</h3>
      <div className="space-y-4">
        {exportJobs?.data?.map((job) => {
          const StatusIcon = statusIcons[job.status];
          const statusColor = statusColors[job.status];

          return (
            <div key={job.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <StatusIcon className={`h-5 w-5 ${statusColor}`} />
                <div>
                  <div className="text-sm font-medium text-gray-900">{job.session_name}</div>
                  <div className="text-xs text-gray-500">
                    {job.format.toUpperCase()} • {new Date(job.created_at).toLocaleString()}
                  </div>
                  {job.status === 'processing' && (
                    <div className="mt-1">
                      <div className="w-32 bg-gray-200 rounded-full h-1">
                        <div
                          className="bg-blue-500 h-1 rounded-full"
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    </div>
                  )}
                  {job.status === 'failed' && (
                    <div className="text-xs text-red-600 mt-1">{job.error}</div>
                  )}
                </div>
              </div>
              {job.status === 'completed' && (
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-500">{job.file_size}</span>
                  <button className="text-indigo-600 hover:text-indigo-800 text-sm">Download</button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ExportHistory;
