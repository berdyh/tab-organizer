import { RefreshCcw, ShieldCheck } from 'lucide-react';

const AuthHeader = ({ onRefresh }) => (
  <header className="flex items-center justify-between">
    <div>
      <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
        <ShieldCheck className="h-6 w-6 text-indigo-500" />
        Authentication Control Center
      </h1>
      <p className="mt-1 text-sm text-gray-500">
        Manage credentials, interactive auth flows, and active sessions across domains.
      </p>
    </div>
    <button
      onClick={onRefresh}
      className="inline-flex items-center px-3 py-2 rounded-md bg-white border border-gray-200 shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50"
    >
      <RefreshCcw className="h-4 w-4 mr-2" />
      Refresh Data
    </button>
  </header>
);

export default AuthHeader;
