import { CheckCircle2, Globe, Trash2, XCircle } from 'lucide-react';

const DomainsTable = ({ domains, isLoading, onRemoveCredentials, isRemoving }) => (
  <section className="bg-white rounded-lg shadow">
    <header className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
      <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
        <Globe className="h-5 w-5 text-indigo-500" /> Domain Authentication Overview
      </h2>
      <span className="text-sm text-gray-500">{domains.length} domains tracked</span>
    </header>
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <HeaderCell>Domain</HeaderCell>
            <HeaderCell>Auth Method</HeaderCell>
            <HeaderCell>Requires Auth</HeaderCell>
            <HeaderCell>Credentials</HeaderCell>
            <HeaderCell>Last Verified</HeaderCell>
            <HeaderCell>Success / Failure</HeaderCell>
            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {domains.map((domain) => (
            <tr key={domain.domain}>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm font-medium text-gray-900">{domain.domain}</div>
                {domain.login_url && (
                  <div className="text-xs text-gray-500 truncate max-w-xs">{domain.login_url}</div>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{domain.auth_method}</td>
              <td className="px-6 py-4 whitespace-nowrap">
                {domain.requires_auth ? (
                  <Badge tone="warning">Requires Auth</Badge>
                ) : (
                  <Badge tone="success">Open</Badge>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                {domain.has_credentials ? (
                  <Badge tone="info">Stored</Badge>
                ) : (
                  <Badge tone="neutral">Not Stored</Badge>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {domain.last_verified ? new Date(domain.last_verified).toLocaleString() : '—'}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1 text-green-600">
                    <CheckCircle2 className="h-4 w-4" />
                    {domain.success_count || 0}
                  </span>
                  <span className="flex items-center gap-1 text-red-600">
                    <XCircle className="h-4 w-4" />
                    {domain.failure_count || 0}
                  </span>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium space-x-2">
                {domain.has_credentials && (
                  <button
                    onClick={() => onRemoveCredentials(domain.domain)}
                    disabled={isRemoving}
                    className="inline-flex items-center px-2.5 py-1.5 border border-transparent text-xs font-medium rounded text-red-600 bg-red-50 hover:bg-red-100 disabled:opacity-50"
                  >
                    <Trash2 className="h-4 w-4 mr-1" /> Remove
                  </button>
                )}
              </td>
            </tr>
          ))}
          {domains.length === 0 && !isLoading && (
            <tr>
              <td colSpan="7" className="px-6 py-8 text-center text-sm text-gray-500">
                No domains tracked yet. Add credentials or run detection to populate this table.
              </td>
            </tr>
          )}
          {isLoading && (
            <tr>
              <td colSpan="7" className="px-6 py-8 text-center text-sm text-gray-500">
                Loading domain data…
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  </section>
);

const HeaderCell = ({ children }) => (
  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
    {children}
  </th>
);

const Badge = ({ children, tone }) => {
  const styles = {
    success: 'bg-green-100 text-green-800',
    warning: 'bg-yellow-100 text-yellow-800',
    info: 'bg-indigo-100 text-indigo-800',
    neutral: 'bg-gray-100 text-gray-500',
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${styles[tone]}`}>
      {children}
    </span>
  );
};

export default DomainsTable;
