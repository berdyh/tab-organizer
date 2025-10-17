import React from 'react';
import { useQuery } from 'react-query';
import { Link } from 'react-router-dom';
import { 
  Globe, 
  Database, 
  Zap, 
  TrendingUp,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';
import { scrapingAPI, sessionAPI } from '../lib/api';

const Dashboard = () => {
  const { data: jobs } = useQuery('scraping-jobs', scrapingAPI.getJobs);
  const { data: sessions } = useQuery('sessions', sessionAPI.list);

  const stats = React.useMemo(() => {
    if (!jobs?.data) return { total: 0, completed: 0, failed: 0, pending: 0 };
    
    const jobStats = jobs.data.reduce((acc, job) => {
      acc.total++;
      if (job.status === 'completed') acc.completed++;
      else if (job.status === 'failed') acc.failed++;
      else acc.pending++;
      return acc;
    }, { total: 0, completed: 0, failed: 0, pending: 0 });

    return jobStats;
  }, [jobs]);

  const recentJobs = jobs?.data?.slice(0, 5) || [];
  const recentSessions = sessions?.data?.slice(0, 3) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Overview of your web scraping and clustering activities
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total URLs"
          value={stats.total}
          icon={Globe}
          color="blue"
        />
        <StatCard
          title="Completed"
          value={stats.completed}
          icon={CheckCircle}
          color="green"
        />
        <StatCard
          title="Failed"
          value={stats.failed}
          icon={XCircle}
          color="red"
        />
        <StatCard
          title="Pending"
          value={stats.pending}
          icon={Clock}
          color="yellow"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Recent Jobs */}
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Recent Jobs
              </h3>
              <Link
                to="/urls"
                className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
              >
                View all
              </Link>
            </div>
            <div className="mt-6 flow-root">
              <ul className="-my-5 divide-y divide-gray-200">
                {recentJobs.map((job) => (
                  <JobItem key={job.id} job={job} />
                ))}
                {recentJobs.length === 0 && (
                  <li className="py-4">
                    <p className="text-sm text-gray-500">No recent jobs</p>
                  </li>
                )}
              </ul>
            </div>
          </div>
        </div>

        {/* Recent Sessions */}
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Recent Sessions
              </h3>
              <Link
                to="/sessions"
                className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
              >
                View all
              </Link>
            </div>
            <div className="mt-6 flow-root">
              <ul className="-my-5 divide-y divide-gray-200">
                {recentSessions.map((session) => (
                  <SessionItem key={session.id} session={session} />
                ))}
                {recentSessions.length === 0 && (
                  <li className="py-4">
                    <p className="text-sm text-gray-500">No recent sessions</p>
                  </li>
                )}
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white shadow rounded-lg">
        <div className="p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
            Quick Actions
          </h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <QuickActionCard
              title="Add URLs"
              description="Start a new scraping job"
              to="/urls"
              icon={Globe}
            />
            <QuickActionCard
              title="Search Content"
              description="Find scraped content"
              to="/search"
              icon={Database}
            />
            <QuickActionCard
              title="New Session"
              description="Create analysis session"
              to="/sessions"
              icon={Zap}
            />
            <QuickActionCard
              title="Export Data"
              description="Export to various formats"
              to="/export"
              icon={TrendingUp}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ title, value, icon: Icon, color }) => {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-100',
    green: 'text-green-600 bg-green-100',
    red: 'text-red-600 bg-red-100',
    yellow: 'text-yellow-600 bg-yellow-100',
  };

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="p-5">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <Icon className={`h-6 w-6 ${colorClasses[color]}`} />
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">
                {title}
              </dt>
              <dd className="text-lg font-medium text-gray-900">
                {value}
              </dd>
            </dl>
          </div>
        </div>
      </div>
    </div>
  );
};

const JobItem = ({ job }) => {
  const statusIcons = {
    completed: CheckCircle,
    failed: XCircle,
    pending: Clock,
    running: AlertCircle,
  };

  const statusColors = {
    completed: 'text-green-500',
    failed: 'text-red-500',
    pending: 'text-yellow-500',
    running: 'text-blue-500',
  };

  const StatusIcon = statusIcons[job.status] || Clock;

  return (
    <li className="py-4">
      <div className="flex items-center space-x-4">
        <div className="flex-shrink-0">
          <StatusIcon className={`h-5 w-5 ${statusColors[job.status]}`} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 truncate">
            {job.url || `Batch job (${job.total_urls} URLs)`}
          </p>
          <p className="text-sm text-gray-500">
            {new Date(job.created_at).toLocaleString()}
          </p>
        </div>
        <div className="flex-shrink-0 text-sm text-gray-500">
          {job.status}
        </div>
      </div>
    </li>
  );
};

const SessionItem = ({ session }) => (
  <li className="py-4">
    <div className="flex items-center space-x-4">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">
          {session.name}
        </p>
        <p className="text-sm text-gray-500">
          {session.description || 'No description'}
        </p>
      </div>
      <div className="flex-shrink-0 text-sm text-gray-500">
        {new Date(session.updated_at).toLocaleDateString()}
      </div>
    </div>
  </li>
);

const QuickActionCard = ({ title, description, to, icon: Icon }) => (
  <Link
    to={to}
    className="relative group bg-white p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-indigo-500 hover:bg-gray-50 rounded-lg border border-gray-200"
  >
    <div>
      <span className="rounded-lg inline-flex p-3 bg-indigo-50 text-indigo-600 group-hover:bg-indigo-100">
        <Icon className="h-6 w-6" />
      </span>
    </div>
    <div className="mt-4">
      <h3 className="text-lg font-medium text-gray-900">
        {title}
      </h3>
      <p className="mt-2 text-sm text-gray-500">
        {description}
      </p>
    </div>
  </Link>
);

export default Dashboard;
